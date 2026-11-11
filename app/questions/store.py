"""SQLite 题库：保存生成草稿及教师审核结果。"""
from contextlib import closing
from datetime import datetime, timezone
import json
import sqlite3

from app import config
from app.questions.models import Question


class QuestionStore:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or config.DATA_DIR

    def _connect(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.data_dir / "assistant.db", timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("""CREATE TABLE IF NOT EXISTS teacher_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stem TEXT NOT NULL,
            options_json TEXT NOT NULL,
            answer TEXT NOT NULL,
            explanation TEXT NOT NULL,
            source TEXT NOT NULL,
            page INTEGER NOT NULL CHECK (page > 0),
            chunk_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('draft', 'approved', 'rejected')),
            subject TEXT NOT NULL DEFAULT '',
            topic TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""")
        columns = {row[1] for row in db.execute("PRAGMA table_info(teacher_questions)")}
        if "subject" not in columns:
            db.execute("ALTER TABLE teacher_questions ADD COLUMN subject TEXT NOT NULL DEFAULT ''")
        if "topic" not in columns:
            db.execute("ALTER TABLE teacher_questions ADD COLUMN topic TEXT NOT NULL DEFAULT ''")
        db.commit()
        return db

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _record(row):
        return {
            "id": row["id"],
            "stem": row["stem"],
            "options": json.loads(row["options_json"]),
            "answer": row["answer"],
            "explanation": row["explanation"],
            "source": row["source"],
            "page": row["page"],
            "chunk_id": row["chunk_id"],
            "status": row["status"],
            "subject": row["subject"],
            "topic": row["topic"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def save_many(self, questions):
        if any(not isinstance(question, Question) for question in questions):
            raise ValueError("只能保存 Question 题目对象")
        if not questions:
            return []
        now = self._now()
        with closing(self._connect()) as db, db:
            ids = []
            for question in questions:
                cursor = db.execute(
                    "INSERT INTO teacher_questions "
                    "(stem, options_json, answer, explanation, source, page, chunk_id, status, subject, topic, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (question.stem, json.dumps(question.to_dict()["options"], ensure_ascii=False),
                     question.answer, question.explanation, question.source, question.page,
                     question.chunk_id, question.status, question.subject, question.topic, now, now),
                )
                ids.append(cursor.lastrowid)
            rows = [db.execute("SELECT * FROM teacher_questions WHERE id=?", (question_id,)).fetchone()
                    for question_id in ids]
            return [self._record(row) for row in rows]

    def list(self, status=None, *, subject=None, topic=None):
        if status is not None and status not in {"draft", "approved", "rejected"}:
            raise ValueError("无效的题目状态")
        with closing(self._connect()) as db:
            conditions, values = [], []
            if status:
                conditions.append("status=?")
                values.append(status)
            if subject:
                conditions.append("subject=?")
                values.append(subject)
            if topic:
                conditions.append("topic LIKE ?")
                values.append(f"%{topic}%")
            where = " WHERE " + " AND ".join(conditions) if conditions else ""
            rows = db.execute("SELECT * FROM teacher_questions" + where + " ORDER BY id DESC", values).fetchall()
        return [self._record(row) for row in rows]

    def review(self, question_id, *, stem, options, answer, explanation, status, subject=None, topic=None):
        if type(question_id) is not int or question_id < 1:
            raise ValueError("题目 ID 必须是正整数")
        with closing(self._connect()) as db, db:
            row = db.execute("SELECT * FROM teacher_questions WHERE id=?", (question_id,)).fetchone()
            if row is None:
                raise LookupError("题目不存在")
            question = Question(
                stem=stem, options=tuple(options), answer=answer, explanation=explanation,
                source=row["source"], page=row["page"], chunk_id=row["chunk_id"], status=status,
                subject=subject if subject is not None else row["subject"],
                topic=topic if topic is not None else row["topic"],
            )
            db.execute(
                "UPDATE teacher_questions SET stem=?, options_json=?, answer=?, explanation=?, status=?, subject=?, topic=?, updated_at=? "
                "WHERE id=?",
                (question.stem, json.dumps(question.to_dict()["options"], ensure_ascii=False),
                 question.answer, question.explanation, question.status, question.subject, question.topic,
                 self._now(), question_id),
            )
            return self._record(db.execute("SELECT * FROM teacher_questions WHERE id=?", (question_id,)).fetchone())
