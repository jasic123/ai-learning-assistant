"""Persistence for editable text mind maps."""
from contextlib import closing
from datetime import datetime, timezone
import json
import sqlite3

from app import config


class MindMapStore:
    def __init__(self, data_dir=None, owner_id="local"):
        self.data_dir = data_dir or config.DATA_DIR
        self.owner_id = owner_id

    def _connect(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.data_dir / "assistant.db", timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("""CREATE TABLE IF NOT EXISTS learning_mindmaps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id TEXT NOT NULL DEFAULT 'local',
            title TEXT NOT NULL, tree_json TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        )""")
        columns = {row[1] for row in db.execute("PRAGMA table_info(learning_mindmaps)")}
        if "owner_id" not in columns:
            db.execute("ALTER TABLE learning_mindmaps ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'local'")
        db.commit()
        return db

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _validate_tree(tree):
        if not isinstance(tree, dict) or not isinstance(tree.get("title"), str) or not tree["title"].strip():
            raise ValueError("导图必须包含非空 title")
        nodes = tree.get("children", [])
        if not isinstance(nodes, list) or len(nodes) > 30:
            raise ValueError("导图 children 必须是最多 30 个节点的列表")
        for node in nodes:
            if not isinstance(node, dict) or not isinstance(node.get("title"), str) or not node["title"].strip():
                raise ValueError("导图节点必须包含非空 title")
            if "children" in node:
                MindMapStore._validate_tree({"title": node["title"], "children": node["children"]})
        return tree

    def save(self, title, tree, map_id=None):
        if not isinstance(title, str) or not title.strip() or len(title) > 200:
            raise ValueError("导图标题须为 1～200 字符")
        self._validate_tree(tree)
        now = self._now()
        with closing(self._connect()) as db, db:
            if map_id is None:
                cursor = db.execute("INSERT INTO learning_mindmaps(owner_id, title, tree_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                                    (self.owner_id, title.strip(), json.dumps(tree, ensure_ascii=False), now, now))
                map_id = cursor.lastrowid
            else:
                if type(map_id) is not int or map_id < 1:
                    raise ValueError("导图编号无效")
                if not db.execute("SELECT 1 FROM learning_mindmaps WHERE id=? AND owner_id=?",
                                  (map_id, self.owner_id)).fetchone():
                    raise LookupError("导图不存在")
                db.execute("UPDATE learning_mindmaps SET title=?, tree_json=?, updated_at=? WHERE id=? AND owner_id=?",
                           (title.strip(), json.dumps(tree, ensure_ascii=False), now, map_id, self.owner_id))
            row = db.execute("SELECT * FROM learning_mindmaps WHERE id=? AND owner_id=?",
                             (map_id, self.owner_id)).fetchone()
        return self._record(row)

    @staticmethod
    def _record(row):
        return {"id": row["id"], "title": row["title"], "tree": json.loads(row["tree_json"]),
                "created_at": row["created_at"], "updated_at": row["updated_at"]}

    def list(self):
        with closing(self._connect()) as db:
            rows = db.execute("SELECT * FROM learning_mindmaps WHERE owner_id=? ORDER BY updated_at DESC",
                              (self.owner_id,)).fetchall()
        return [self._record(row) for row in rows]
