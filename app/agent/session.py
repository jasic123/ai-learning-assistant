"""SQLite 短期会话：只提交完整成功轮次，不存系统提示或模型配置。"""
from contextlib import contextmanager, closing
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3

from app import config


def validate_session_id(session_id):
    if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session_id):
        raise ValueError("会话标识须为 1～64 位字母、数字、下划线或连字符")


@dataclass(frozen=True)
class SessionSnapshot:
    version: int
    turns: list[list[dict]]
    turn_numbers: tuple[int, ...] = ()
    summary: str = ""
    through_turn: int = 0
    sources: tuple[dict, ...] = ()


class SessionManager:
    def __init__(self, data_dir=None, *, owner_id="local"):
        # 创建对象时不连接数据库，允许临时目录隔离测试。
        self.data_dir = Path(data_dir) if data_dir is not None else config.DATA_DIR
        validate_session_id(owner_id)
        self.owner_id = owner_id

    @contextmanager
    def _connect(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.data_dir / "assistant.db", timeout=30)) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS agent_turns (
                session_id TEXT NOT NULL, turn_number INTEGER NOT NULL,
                messages_json TEXT NOT NULL, created_at TEXT NOT NULL,
                PRIMARY KEY (session_id, turn_number))""")
            db.execute("""CREATE TABLE IF NOT EXISTS agent_session_owners (
                session_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS agent_checkpoints (
                session_id TEXT PRIMARY KEY, through_turn INTEGER NOT NULL,
                summary TEXT NOT NULL, sources_json TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS agent_memories (
                owner_id TEXT NOT NULL, memory_key TEXT NOT NULL, content TEXT NOT NULL,
                updated_at TEXT NOT NULL, PRIMARY KEY (owner_id, memory_key))""")
            db.commit()
            yield db

    def _bind(self, db, session_id):
        """会话名不跨所有者复用；旧版未绑定会话归 local，避免被其他标识认领。"""
        existing = db.execute("SELECT owner_id FROM agent_session_owners WHERE session_id=?", (session_id,)).fetchone()
        if existing is None:
            legacy = db.execute("SELECT 1 FROM agent_turns WHERE session_id=? LIMIT 1", (session_id,)).fetchone()
            owner = "local" if legacy else self.owner_id
            db.execute("INSERT OR IGNORE INTO agent_session_owners VALUES (?, ?)", (session_id, owner))
            existing = db.execute("SELECT owner_id FROM agent_session_owners WHERE session_id=?", (session_id,)).fetchone()
        if existing[0] != self.owner_id:
            raise ValueError("此会话属于其他所有者，请使用新的会话标识")

    def load(self, session_id: str) -> SessionSnapshot:
        validate_session_id(session_id)
        with self._connect() as db, db:
            self._bind(db, session_id)
            checkpoint = db.execute("SELECT through_turn, summary, sources_json FROM agent_checkpoints WHERE session_id=?",
                                    (session_id,)).fetchone()
            through, summary, sources = checkpoint if checkpoint else (0, "", "[]")
            # 加载检查点后的全部轮次；原始历史只归档，不按固定轮数丢弃。
            rows = db.execute(
                "SELECT turn_number, messages_json FROM agent_turns WHERE session_id=? AND turn_number>? "
                "ORDER BY turn_number DESC", (session_id, through),
            ).fetchall()
        return SessionSnapshot(rows[0][0] if rows else through,
                               [json.loads(row[1]) for row in reversed(rows)],
                               tuple(row[0] for row in reversed(rows)), summary, through,
                               tuple(json.loads(sources)))

    def read_turn(self, session_id, turn_number):
        validate_session_id(session_id)
        if type(turn_number) is not int or turn_number < 1:
            raise ValueError("轮次须为正整数")
        with self._connect() as db, db:
            self._bind(db, session_id)
            row = db.execute("SELECT messages_json FROM agent_turns WHERE session_id=? AND turn_number=?",
                             (session_id, turn_number)).fetchone()
        if not row:
            raise ValueError("指定轮次不存在或尚未完成")
        return row[0]

    def list_sessions(self, limit=50):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("会话列表限制须为 1～100")
        with self._connect() as db:
            rows = db.execute(
                "SELECT t.session_id, MAX(t.turn_number), MAX(t.created_at), latest.messages_json "
                "FROM agent_turns AS t "
                "JOIN agent_session_owners AS o ON o.session_id=t.session_id "
                "JOIN agent_turns AS latest ON latest.session_id=t.session_id "
                "AND latest.turn_number=(SELECT MAX(x.turn_number) FROM agent_turns AS x "
                "WHERE x.session_id=t.session_id) "
                "WHERE o.owner_id=? GROUP BY t.session_id "
                "ORDER BY MAX(t.created_at) DESC LIMIT ?", (self.owner_id, limit),
            ).fetchall()
        sessions = []
        for session_id, turn_count, updated_at, messages_json in rows:
            messages = json.loads(messages_json)
            preview = next((item.get("content", "") for item in messages if item.get("role") == "user"), "")
            sessions.append({"session_id": session_id, "turn_count": turn_count,
                             "updated_at": updated_at, "preview": preview[:100]})
        return sessions

    def load_all_turns(self, session_id):
        validate_session_id(session_id)
        with self._connect() as db, db:
            self._bind(db, session_id)
            rows = db.execute(
                "SELECT turn_number, messages_json FROM agent_turns WHERE session_id=? ORDER BY turn_number",
                (session_id,),
            ).fetchall()
        return [{"turn_number": number, "messages": json.loads(payload)} for number, payload in rows]

    def delete_session(self, session_id):
        """Delete one owned conversation and all of its persisted metadata atomically."""
        validate_session_id(session_id)
        with self._connect() as db, db:
            db.execute("BEGIN IMMEDIATE")
            has_turns = db.execute("SELECT 1 FROM agent_turns WHERE session_id=? LIMIT 1",
                                   (session_id,)).fetchone()
            if not has_turns:
                raise ValueError("会话不存在")
            self._bind(db, session_id)
            db.execute("DELETE FROM agent_checkpoints WHERE session_id=?", (session_id,))
            db.execute("DELETE FROM agent_turns WHERE session_id=?", (session_id,))
            db.execute("DELETE FROM agent_session_owners WHERE session_id=?", (session_id,))
        return {"session_id": session_id, "deleted": True}

    def append_turn(self, session_id: str, messages: list[dict], *, expected_version: int, checkpoint=None):
        validate_session_id(session_id)
        if type(expected_version) is not int or expected_version < 0:
            raise ValueError("无效会话版本")
        if (not messages or messages[0].get("role") != "user"
            or messages[-1].get("role") != "assistant"
            or any(item.get("role") not in {"user", "assistant", "tool"} for item in messages)):
            raise ValueError("只允许保存完整的用户/助理对话轮次")
        payload = json.dumps(messages, ensure_ascii=False, allow_nan=False)
        with self._connect() as db, db:
            # 模型请求期间不持有锁；提交时检查版本，避免同一会话并发串话。
            db.execute("BEGIN IMMEDIATE")
            self._bind(db, session_id)
            version = db.execute(
                "SELECT COALESCE(MAX(turn_number), 0) FROM agent_turns WHERE session_id=?",
                (session_id,),
            ).fetchone()[0]
            if version != expected_version:
                raise RuntimeError("同一会话已被其他请求更新，请重试本轮")
            db.execute(
                "INSERT INTO agent_turns VALUES (?, ?, ?, ?)",
                (session_id, version + 1, payload, datetime.now(timezone.utc).isoformat()),
            )
            if checkpoint is not None:
                through, summary, sources = checkpoint
                if type(through) is not int or not 0 < through <= expected_version or not isinstance(summary, str) or not summary.strip():
                    raise ValueError("无效压缩检查点")
                # 摘要与成功轮次同事务提交；失败轮次和失败摘要不会覆盖旧检查点。
                db.execute("INSERT INTO agent_checkpoints VALUES (?, ?, ?, ?) "
                           "ON CONFLICT(session_id) DO UPDATE SET through_turn=excluded.through_turn, "
                           "summary=excluded.summary, sources_json=excluded.sources_json",
                           (session_id, through, summary, json.dumps(sources, ensure_ascii=False, allow_nan=False)))
