"""SQLite persistence for quizzes, answers, mistakes and review scheduling."""
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
import random
import sqlite3

from app import config


class LearningStore:
    def __init__(self, data_dir=None, owner_id="local"):
        self.data_dir = data_dir or config.DATA_DIR
        self.owner_id = owner_id

    def _connect(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.data_dir / "assistant.db", timeout=30)
        db.row_factory = sqlite3.Row
        db.executescript("""
        CREATE TABLE IF NOT EXISTS learning_quizzes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id TEXT NOT NULL DEFAULT 'local',
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            question_ids_json TEXT NOT NULL,
            quiz_type TEXT NOT NULL DEFAULT 'practice',
            review_pairs_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS learning_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            quiz_id INTEGER NOT NULL REFERENCES learning_quizzes(id),
            answers_json TEXT NOT NULL,
            results_json TEXT NOT NULL,
            correct_count INTEGER NOT NULL,
            total_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS learning_mistakes (
            question_id INTEGER NOT NULL REFERENCES teacher_questions(id),
            owner_id TEXT NOT NULL DEFAULT 'local',
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            wrong_count INTEGER NOT NULL DEFAULT 1,
            correct_count INTEGER NOT NULL DEFAULT 0,
            repetitions INTEGER NOT NULL DEFAULT 0,
            ease_factor REAL NOT NULL DEFAULT 2.5,
            interval_days INTEGER NOT NULL DEFAULT 0,
            due_at TEXT NOT NULL,
            last_reviewed_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (owner_id, question_id)
        );
        CREATE TABLE IF NOT EXISTS learning_chat_feedback (
            session_id TEXT NOT NULL,
            turn_number INTEGER NOT NULL,
            rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),
            comment TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            PRIMARY KEY (session_id, turn_number)
        );
        CREATE TABLE IF NOT EXISTS learning_error_reasons (
            attempt_id INTEGER NOT NULL REFERENCES learning_attempts(id) ON DELETE CASCADE,
            question_id INTEGER NOT NULL REFERENCES teacher_questions(id),
            owner_id TEXT NOT NULL,
            reason TEXT NOT NULL CHECK (reason IN (
                'concept_unclear', 'condition_missed', 'method_confused',
                'calculation_error', 'guessed', 'other'
            )),
            created_at TEXT NOT NULL,
            PRIMARY KEY (attempt_id, question_id)
        );
        """)
        columns = {row[1] for row in db.execute("PRAGMA table_info(learning_quizzes)")}
        if "owner_id" not in columns:
            db.execute("ALTER TABLE learning_quizzes ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'local'")
        if "quiz_type" not in columns:
            db.execute("ALTER TABLE learning_quizzes ADD COLUMN quiz_type TEXT NOT NULL DEFAULT 'practice'")
        if "review_pairs_json" not in columns:
            db.execute("ALTER TABLE learning_quizzes ADD COLUMN review_pairs_json TEXT NOT NULL DEFAULT '[]'")
        mistake_info = db.execute("PRAGMA table_info(learning_mistakes)").fetchall()
        if "owner_id" not in {row[1] for row in mistake_info} or sum(bool(row[5]) for row in mistake_info) == 1:
            db.execute("ALTER TABLE learning_mistakes RENAME TO learning_mistakes_legacy")
            db.execute("""CREATE TABLE learning_mistakes (
                question_id INTEGER NOT NULL REFERENCES teacher_questions(id), owner_id TEXT NOT NULL DEFAULT 'local',
                subject TEXT NOT NULL, topic TEXT NOT NULL, wrong_count INTEGER NOT NULL DEFAULT 1,
                correct_count INTEGER NOT NULL DEFAULT 0, repetitions INTEGER NOT NULL DEFAULT 0,
                ease_factor REAL NOT NULL DEFAULT 2.5, interval_days INTEGER NOT NULL DEFAULT 0,
                due_at TEXT NOT NULL, last_reviewed_at TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY(owner_id, question_id))""")
            legacy = {row[1] for row in db.execute("PRAGMA table_info(learning_mistakes_legacy)")}
            owner = "owner_id" if "owner_id" in legacy else "'local'"
            db.execute("INSERT OR IGNORE INTO learning_mistakes SELECT question_id, " + owner +
                       ", subject, topic, wrong_count, correct_count, repetitions, ease_factor, interval_days, "
                       "due_at, last_reviewed_at, created_at, updated_at FROM learning_mistakes_legacy")
            db.execute("DROP TABLE learning_mistakes_legacy")
        db.execute("CREATE INDEX IF NOT EXISTS idx_learning_mistakes_owner_due ON learning_mistakes(owner_id, due_at)")
        db.commit()
        return db

    @staticmethod
    def _now():
        return datetime.now(timezone.utc)

    @staticmethod
    def _iso(value):
        return value.astimezone(timezone.utc).isoformat()

    def create_quiz(self, *, subject, topic, questions, quiz_type="practice", review_pairs=(), owner_id=None):
        if not isinstance(subject, str) or not subject.strip() or len(subject) > 100:
            raise ValueError("学科须为 1～100 字符")
        if not isinstance(topic, str) or not topic.strip() or not questions:
            raise ValueError("知识点和题目不能为空")
        now = self._iso(self._now())
        ids = [item["id"] for item in questions]
        with closing(self._connect()) as db, db:
            cursor = db.execute(
                "INSERT INTO learning_quizzes(owner_id, subject, topic, question_ids_json, quiz_type, review_pairs_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (owner_id or self.owner_id, subject.strip(), topic.strip(), json.dumps(ids), quiz_type,
                 json.dumps(review_pairs), now),
            )
            quiz_id = cursor.lastrowid
        return {"id": quiz_id, "subject": subject.strip(), "topic": topic.strip(),
                "questions": [{key: item[key] for key in ("id", "stem", "options", "source", "page")}
                              for item in questions], "created_at": now}

    def submit_quiz(self, quiz_id, answers, owner_id=None):
        if type(quiz_id) is not int or quiz_id < 1 or not isinstance(answers, dict):
            raise ValueError("练习编号或答案格式无效")
        now_dt, now = self._now(), self._iso(self._now())
        with closing(self._connect()) as db, db:
            quiz = db.execute("SELECT * FROM learning_quizzes WHERE id=? AND owner_id=?",
                              (quiz_id, owner_id or self.owner_id)).fetchone()
            if quiz is None:
                raise LookupError("练习不存在")
            question_ids = json.loads(quiz["question_ids_json"])
            if set(answers) != {str(item) for item in question_ids}:
                raise ValueError("请为每道题提交一个答案")
            if any(value not in {"A", "B", "C", "D"} for value in answers.values()):
                raise ValueError("答案必须是 A、B、C 或 D")
            placeholders = ",".join("?" for _ in question_ids)
            rows = db.execute(f"SELECT * FROM teacher_questions WHERE id IN ({placeholders})", question_ids).fetchall()
            by_id = {row["id"]: row for row in rows}
            if len(by_id) != len(question_ids):
                raise LookupError("练习中有题目已不存在")
            results = []
            correct_count = 0
            review_pairs = json.loads(quiz["review_pairs_json"])
            review_ids = {question_id for pair in review_pairs for question_id in pair}
            for question_id in question_ids:
                row = by_id[question_id]
                selected = answers[str(question_id)]
                correct = selected == row["answer"]
                correct_count += correct
                result = {
                    "question_id": question_id, "stem": row["stem"], "selected": selected,
                    "correct": correct, "answer": row["answer"], "explanation": row["explanation"],
                    "source": row["source"], "page": row["page"],
                }
                results.append(result)
                existing = db.execute("SELECT question_id FROM learning_mistakes WHERE question_id=? AND owner_id=?",
                                      (question_id, owner_id or self.owner_id)).fetchone()
                if not correct and question_id not in review_ids:
                    if existing:
                        if quiz["quiz_type"] == "review":
                            db.execute("UPDATE learning_mistakes SET wrong_count=wrong_count+1, due_at=?, "
                                       "updated_at=? WHERE question_id=? AND owner_id=?",
                                       (now, now, question_id, owner_id or self.owner_id))
                        else:
                            db.execute("UPDATE learning_mistakes SET wrong_count=wrong_count+1, subject=?, topic=?, "
                                       "due_at=?, updated_at=? WHERE question_id=? AND owner_id=?",
                                       (quiz["subject"], quiz["topic"], now, now, question_id, owner_id or self.owner_id))
                    else:
                        db.execute("INSERT INTO learning_mistakes "
                                   "(question_id, owner_id, subject, topic, due_at, created_at, updated_at) "
                                   "VALUES (?, ?, ?, ?, ?, ?, ?)",
                                   (question_id, owner_id or self.owner_id, quiz["subject"], quiz["topic"], now, now, now))
                elif existing and quiz["quiz_type"] != "review":
                    db.execute("UPDATE learning_mistakes SET correct_count=correct_count+1, updated_at=? "
                                   "WHERE question_id=? AND owner_id=?", (now, question_id, owner_id or self.owner_id))
            if quiz["quiz_type"] == "review":
                for original_id, variant_id in json.loads(quiz["review_pairs_json"]):
                    if by_id[original_id]["answer"] == answers[str(original_id)] and \
                            by_id[variant_id]["answer"] == answers[str(variant_id)]:
                        db.execute("DELETE FROM learning_mistakes WHERE question_id=? AND owner_id=?",
                                   (original_id, owner_id or self.owner_id))
                    else:
                        db.execute("UPDATE learning_mistakes SET correct_count=correct_count+1, updated_at=? "
                                   "WHERE question_id=? AND owner_id=? AND ?=?",
                                   (now, original_id, owner_id or self.owner_id,
                                   by_id[original_id]["answer"], answers[str(original_id)]))
            cursor = db.execute("INSERT INTO learning_attempts "
                       "(quiz_id, answers_json, results_json, correct_count, total_count, created_at) "
                       "VALUES (?, ?, ?, ?, ?, ?)",
                       (quiz_id, json.dumps(answers), json.dumps(results, ensure_ascii=False),
                        correct_count, len(question_ids), self._iso(now_dt)))
            attempt_id = cursor.lastrowid
        return {"quiz_id": quiz_id, "correct_count": correct_count, "total_count": len(question_ids),
                "score": round(correct_count / len(question_ids) * 100), "results": results,
                "attempt_id": attempt_id}

    def replace_quiz_question(self, quiz_id, old_question_id, replacement, owner_id=None):
        if type(quiz_id) is not int or type(old_question_id) is not int or old_question_id < 1:
            raise ValueError("练习或题目编号无效")
        with closing(self._connect()) as db, db:
            quiz = db.execute("SELECT * FROM learning_quizzes WHERE id=? AND owner_id=?",
                              (quiz_id, owner_id or self.owner_id)).fetchone()
            if quiz is None:
                raise LookupError("练习不存在")
            ids = json.loads(quiz["question_ids_json"])
            try:
                index = ids.index(old_question_id)
            except ValueError:
                raise LookupError("题目不属于当前练习") from None
            ids[index] = replacement["id"]
            db.execute("UPDATE learning_quizzes SET question_ids_json=? WHERE id=? AND owner_id=?",
                       (json.dumps(ids), quiz_id, owner_id or self.owner_id))
        return index

    def save_error_reasons(self, attempt_id, reasons, owner_id=None):
        """Save optional student-reported reasons only for incorrect answers."""
        allowed = {"concept_unclear", "condition_missed", "method_confused",
                   "calculation_error", "guessed", "other"}
        if type(attempt_id) is not int or attempt_id < 1 or not isinstance(reasons, dict):
            raise ValueError("作答记录或错因格式无效")
        if any(not isinstance(key, str) or not key.isdecimal()
               or not isinstance(value, str) or value not in allowed
               for key, value in reasons.items()):
            raise ValueError("错因记录包含无效题目或类型")
        if not reasons:
            return {"attempt_id": attempt_id, "saved_count": 0}
        owner = owner_id or self.owner_id
        now = self._iso(self._now())
        with closing(self._connect()) as db, db:
            row = db.execute("""SELECT a.results_json FROM learning_attempts a
                               JOIN learning_quizzes q ON q.id=a.quiz_id
                               WHERE a.id=? AND q.owner_id=?""", (attempt_id, owner)).fetchone()
            if row is None:
                raise LookupError("作答记录不存在")
            results = json.loads(row["results_json"])
            incorrect_ids = {str(item["question_id"]) for item in results if not item["correct"]}
            if not set(reasons).issubset(incorrect_ids):
                raise ValueError("只能为答错的题目记录错因")
            for question_id, reason in reasons.items():
                db.execute("""INSERT INTO learning_error_reasons
                              (attempt_id, question_id, owner_id, reason, created_at)
                              VALUES (?, ?, ?, ?, ?)
                              ON CONFLICT(attempt_id, question_id) DO UPDATE SET
                              reason=excluded.reason, created_at=excluded.created_at""",
                           (attempt_id, int(question_id), owner, reason, now))
        return {"attempt_id": attempt_id, "saved_count": len(reasons)}

    def learning_profile(self, owner_id=None):
        """Aggregate durable answer history into evidence-labeled learning signals."""
        owner = owner_id or self.owner_id
        with closing(self._connect()) as db:
            attempts = db.execute("""SELECT a.id, a.created_at, q.subject, q.topic, a.results_json
                                    FROM learning_attempts a
                                    JOIN learning_quizzes q ON q.id=a.quiz_id
                                    WHERE q.owner_id=? ORDER BY a.created_at DESC""", (owner,)).fetchall()
            reasons = db.execute("""SELECT r.reason, r.created_at, q.subject, q.topic
                                    FROM learning_error_reasons r
                                    JOIN learning_quizzes q ON q.owner_id=r.owner_id
                                    JOIN learning_attempts a ON a.id=r.attempt_id AND a.quiz_id=q.id
                                    WHERE r.owner_id=? ORDER BY r.created_at DESC""", (owner,)).fetchall()

        topics = {}
        total_questions = correct_questions = 0
        for attempt in attempts:
            key = (attempt["subject"], attempt["topic"])
            entry = topics.setdefault(key, {"subject": key[0], "topic": key[1],
                                            "attempts": 0, "incorrect": 0,
                                            "last_seen": attempt["created_at"]})
            for result in json.loads(attempt["results_json"]):
                entry["attempts"] += 1
                total_questions += 1
                if result["correct"]:
                    correct_questions += 1
                else:
                    entry["incorrect"] += 1

        topic_rows = []
        for entry in topics.values():
            entry["accuracy"] = round((entry["attempts"] - entry["incorrect"]) /
                                      entry["attempts"] * 100)
            entry["evidence"] = "sample_limited" if entry["attempts"] < 3 else "observed"
            entry["status"] = ("needs_review" if entry["attempts"] >= 3 and entry["accuracy"] < 70
                               else "improving" if entry["attempts"] >= 3
                               else "sample_limited")
            topic_rows.append(entry)
        topic_rows.sort(key=lambda item: (item["status"] != "needs_review",
                                          -item["incorrect"], item["accuracy"], item["topic"]))

        reason_counts = {}
        reason_labels = {
            "concept_unclear": "概念没理解",
            "condition_missed": "看漏题目条件",
            "method_confused": "方法或公式混淆",
            "calculation_error": "计算失误",
            "guessed": "不确定或猜测",
            "other": "其他原因",
        }
        for row in reasons:
            item = reason_counts.setdefault(row["reason"], {"reason": row["reason"],
                                      "count": 0, "last_seen": row["created_at"]})
            item["count"] += 1
        error_patterns = sorted(reason_counts.values(), key=lambda item: (-item["count"], item["reason"]))
        for item in error_patterns:
            item["label"] = reason_labels[item["reason"]]
            item["evidence"] = "sample_limited" if item["count"] < 3 else "repeated_self_report"

        return {
            "total_attempts": len(attempts),
            "total_questions": total_questions,
            "overall_accuracy": round(correct_questions / total_questions * 100) if total_questions else None,
            "topics": topic_rows[:20],
            "error_patterns": error_patterns,
            "reason_sample_count": sum(item["count"] for item in error_patterns),
            "note": "知识点状态依据作答记录汇总；错因来自学生自选，只表示自我报告，不代表固定能力或性格判断。",
        }

    def list_mistakes(self, *, subject=None, query=None, owner_id=None):
        sql = """SELECT m.*, q.stem, q.options_json, q.answer, q.explanation, q.source, q.page
                 FROM learning_mistakes m JOIN teacher_questions q ON q.id=m.question_id"""
        filters, values = ["m.owner_id=?"], [owner_id or self.owner_id]
        if subject:
            filters.append("m.subject=?")
            values.append(subject)
        if query:
            # 关键词搜索不要求用户输入和题库完全相同的连续短语。
            # 中文没有天然空格，先移除常见检索修饰词，再按剩余词逐个匹配。
            stopwords = {"相关", "题目", "题", "错题", "答案", "解析", "关于", "一下"}
            cleaned = query.strip()
            for word in stopwords:
                cleaned = cleaned.replace(word, " ")
            terms = [term for term in cleaned.split() if term]
            if not terms:
                terms = [query.strip()]
            for term in terms:
                pattern = f"%{term}%"
                filters.append("(q.stem LIKE ? OR m.topic LIKE ? OR q.options_json LIKE ? "
                               "OR q.answer LIKE ? OR q.explanation LIKE ? OR q.source LIKE ?)")
                values.extend([pattern] * 6)
        if filters:
            sql += " WHERE " + " AND ".join(filters)
        sql += " ORDER BY m.due_at, m.updated_at DESC"
        with closing(self._connect()) as db:
            rows = db.execute(sql, values).fetchall()
        return [self._mistake(row) for row in rows]

    def add_mistake(self, question, *, subject, topic, owner_id=None):
        if not isinstance(subject, str) or not subject.strip() or len(subject) > 100:
            raise ValueError("学科须为 1～100 字符")
        if not isinstance(topic, str) or not topic.strip() or len(topic) > 2000:
            raise ValueError("知识点须为 1～2000 字符")
        now = self._iso(self._now())
        with closing(self._connect()) as db, db:
            db.execute("INSERT INTO learning_mistakes "
                       "(question_id, owner_id, subject, topic, due_at, created_at, updated_at) "
                       "VALUES (?, ?, ?, ?, ?, ?, ?) "
                       "ON CONFLICT(owner_id, question_id) DO UPDATE SET subject=excluded.subject, topic=excluded.topic, "
                       "due_at=excluded.due_at, updated_at=excluded.updated_at",
                       (question["id"], owner_id or self.owner_id, subject.strip(), topic.strip(), now, now, now))
            row = db.execute("""SELECT m.*, q.stem, q.options_json, q.answer, q.explanation, q.source, q.page
                               FROM learning_mistakes m JOIN teacher_questions q ON q.id=m.question_id
                               WHERE m.question_id=? AND m.owner_id=?""",
                               (question["id"], owner_id or self.owner_id)).fetchone()
        return self._mistake(row)

    @staticmethod
    def _mistake(row):
        return {"question_id": row["question_id"], "subject": row["subject"], "topic": row["topic"],
                "stem": row["stem"], "options": json.loads(row["options_json"]),
                "answer": row["answer"], "explanation": row["explanation"], "source": row["source"],
                "page": row["page"], "wrong_count": row["wrong_count"],
                "correct_count": row["correct_count"], "repetitions": row["repetitions"],
                "ease_factor": row["ease_factor"], "interval_days": row["interval_days"],
                "due_at": row["due_at"], "last_reviewed_at": row["last_reviewed_at"]}

    def review_mistake(self, question_id, quality, owner_id=None):
        if type(question_id) is not int or question_id < 1 or type(quality) is not int or not 0 <= quality <= 5:
            raise ValueError("题目编号无效；复习质量须为 0～5")
        now_dt, now = self._now(), self._iso(self._now())
        with closing(self._connect()) as db, db:
            row = db.execute("SELECT * FROM learning_mistakes WHERE question_id=? AND owner_id=?",
                             (question_id, owner_id or self.owner_id)).fetchone()
            if row is None:
                raise LookupError("错题不存在")
            ease = max(1.3, row["ease_factor"] + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
            repetitions = row["repetitions"] + 1 if quality >= 3 else 0
            if quality < 3:
                interval = 1
            elif repetitions == 1:
                interval = 1
            elif repetitions == 2:
                interval = 6
            else:
                interval = max(1, round(max(1, row["interval_days"]) * ease))
            due = self._iso(now_dt + timedelta(days=interval))
            db.execute("UPDATE learning_mistakes SET repetitions=?, ease_factor=?, interval_days=?, due_at=?, "
                       "last_reviewed_at=?, updated_at=? WHERE question_id=? AND owner_id=?",
                       (repetitions, ease, interval, due, now, now, question_id, owner_id or self.owner_id))
            updated = db.execute("""SELECT m.*, q.stem, q.options_json, q.answer, q.explanation, q.source, q.page
                                   FROM learning_mistakes m JOIN teacher_questions q ON q.id=m.question_id
                                   WHERE m.question_id=? AND m.owner_id=?""",
                                   (question_id, owner_id or self.owner_id)).fetchone()
        return self._mistake(updated)

    def history(self, days=7, owner_id=None):
        if type(days) is not int or days not in {7, 14, 30}:
            raise ValueError("历史范围只能是 7、14 或 30 天")
        since = self._iso(self._now() - timedelta(days=days))
        with closing(self._connect()) as db:
            rows = db.execute("""SELECT a.*, q.subject, q.topic FROM learning_attempts a
                                 JOIN learning_quizzes q ON q.id=a.quiz_id
                                 WHERE a.created_at>=? AND q.owner_id=? ORDER BY a.created_at DESC""",
                               (since, owner_id or self.owner_id)).fetchall()
            mistakes = db.execute("""SELECT subject, COUNT(*) AS count FROM learning_mistakes
                                     WHERE owner_id=? GROUP BY subject ORDER BY count DESC""",
                                   (owner_id or self.owner_id,)).fetchall()
        timeline = [{"attempt_id": row["id"], "quiz_id": row["quiz_id"], "subject": row["subject"],
                     "topic": row["topic"], "correct_count": row["correct_count"],
                     "total_count": row["total_count"],
                     "score": round(row["correct_count"] / row["total_count"] * 100),
                     "created_at": row["created_at"], "results": json.loads(row["results_json"])} for row in rows]
        by_day = {}
        for item in timeline:
            day = item["created_at"][:10]
            stats = by_day.setdefault(day, {"date": day, "attempts": 0, "correct": 0, "total": 0})
            stats["attempts"] += 1
            stats["correct"] += item["correct_count"]
            stats["total"] += item["total_count"]
        weak = {}
        for row in mistakes:
            weak[row["subject"]] = row["count"]
        return {"days": days, "timeline": timeline, "daily": list(by_day.values()), "mistakes_by_subject": weak}

    def due_reviews(self, limit=20, subject=None, randomize=False, owner_id=None):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("复习列表数量须为 1～100")
        now = self._iso(self._now())
        with closing(self._connect()) as db:
            sql = """SELECT m.*, q.stem, q.options_json, q.answer, q.explanation, q.source, q.page
                                 FROM learning_mistakes m JOIN teacher_questions q ON q.id=m.question_id
                                 WHERE m.due_at<=? AND m.owner_id=?"""
            values = [now, owner_id or self.owner_id]
            if subject:
                sql += " AND m.subject=?"
                values.append(subject)
            sql += " ORDER BY RANDOM()" if randomize else " ORDER BY m.due_at"
            sql += " LIMIT ?"
            rows = db.execute(sql, (*values, limit)).fetchall()
        return [self._mistake(row) for row in rows]

    def record_chat_feedback(self, session_id, turn_number, rating, comment=""):
        if (not isinstance(session_id, str) or not session_id or len(session_id) > 64
            or type(turn_number) is not int or turn_number < 1 or rating not in {-1, 1}
            or not isinstance(comment, str) or len(comment) > 1000):
            raise ValueError("问答反馈字段无效")
        now = self._iso(self._now())
        with closing(self._connect()) as db, db:
            db.execute("INSERT INTO learning_chat_feedback VALUES (?, ?, ?, ?, ?) "
                       "ON CONFLICT(session_id, turn_number) DO UPDATE SET rating=excluded.rating, "
                       "comment=excluded.comment, created_at=excluded.created_at",
                       (session_id, turn_number, rating, comment.strip(), now))
        return {"session_id": session_id, "turn_number": turn_number, "rating": rating,
                "comment": comment.strip(), "created_at": now}
