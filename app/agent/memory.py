"""长期记忆：用户明确记录的偏好/项目信息，按所有者跨会话持久保存。"""
from datetime import datetime, timezone
import re

from app.agent.session import SessionManager


class MemoryStore:
    def __init__(self, sessions: SessionManager):
        self.sessions = sessions

    def remember(self, key: str, content: str):
        if not isinstance(key, str) or not re.fullmatch(r"[\w-]{1,64}", key):
            raise ValueError("记忆名称须为 1～64 位中文、字母、数字、下划线或连字符")
        if not isinstance(content, str) or not content.strip() or len(content) > 500:
            raise ValueError("记忆内容须为 1～500 字符")
        with self.sessions._connect() as db, db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT COUNT(*) FROM agent_memories WHERE owner_id=? AND memory_key<>?",
                               (self.sessions.owner_id, key)).fetchone()[0]
            if count >= 20:
                raise ValueError("长期记忆最多 20 条，请先整理或删除旧记忆")
            db.execute("INSERT INTO agent_memories VALUES (?, ?, ?, ?) "
                       "ON CONFLICT(owner_id, memory_key) DO UPDATE SET content=excluded.content, updated_at=excluded.updated_at",
                       (self.sessions.owner_id, key, content.strip(), datetime.now(timezone.utc).isoformat()))

    def list(self):
        with self.sessions._connect() as db:
            rows = db.execute("SELECT memory_key, content FROM agent_memories WHERE owner_id=? ORDER BY memory_key",
                              (self.sessions.owner_id,)).fetchall()
        return [{"key": key, "content": content} for key, content in rows]

    def forget(self, key):
        if not isinstance(key, str):
            raise ValueError("记忆名称须为字符串")
        with self.sessions._connect() as db, db:
            return bool(db.execute("DELETE FROM agent_memories WHERE owner_id=? AND memory_key=?",
                                   (self.sessions.owner_id, key)).rowcount)
