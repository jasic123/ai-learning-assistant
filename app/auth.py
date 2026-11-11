"""Small SQLite-backed account and browser-session store."""
from contextlib import closing
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from uuid import uuid4

from app.config import DATA_DIR


COOKIE_NAME = "study_session"
SESSION_DAYS = 14


class AuthStore:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or DATA_DIR

    def _connect(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.data_dir / "assistant.db", timeout=30)
        db.row_factory = sqlite3.Row
        db.executescript("""
        CREATE TABLE IF NOT EXISTS app_users (
            id TEXT PRIMARY KEY, username TEXT NOT NULL COLLATE NOCASE UNIQUE,
            password_hash TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS app_login_sessions (
            token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES app_users(id) ON DELETE CASCADE,
            expires_at TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_app_login_sessions_expiry ON app_login_sessions(expires_at);
        PRAGMA foreign_keys=ON;
        """)
        db.commit()
        return db

    @staticmethod
    def _now():
        return datetime.now(timezone.utc)

    @staticmethod
    def _hash_password(password, salt=None):
        salt = salt or secrets.token_bytes(16)
        result = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
        return f"{salt.hex()}:{result.hex()}"

    @classmethod
    def _verify_password(cls, password, saved):
        try:
            salt, expected = saved.split(":", 1)
            actual = cls._hash_password(password, bytes.fromhex(salt)).split(":", 1)[1]
            return hmac.compare_digest(actual, expected)
        except (ValueError, TypeError):
            return False

    def register(self, username, password):
        if not isinstance(username, str) or not re.fullmatch(r"[A-Za-z0-9_]{3,32}", username):
            raise ValueError("用户名须为 3～32 位字母、数字或下划线")
        if not isinstance(password, str) or not 8 <= len(password) <= 128:
            raise ValueError("密码长度须为 8～128 位")
        user_id = uuid4().hex
        now = self._now().isoformat()
        try:
            with closing(self._connect()) as db, db:
                db.execute("INSERT INTO app_users VALUES (?, ?, ?, ?)",
                           (user_id, username, self._hash_password(password), now))
        except sqlite3.IntegrityError:
            raise ValueError("用户名已被注册") from None
        return {"id": user_id, "username": username}

    def login(self, username, password):
        with closing(self._connect()) as db:
            user = db.execute("SELECT id, username, password_hash FROM app_users WHERE username=?",
                              (username,)).fetchone()
        if not user or not self._verify_password(password, user["password_hash"]):
            raise ValueError("用户名或密码错误")
        token = secrets.token_urlsafe(32)
        now = self._now()
        with closing(self._connect()) as db, db:
            db.execute("DELETE FROM app_login_sessions WHERE expires_at<=?", (now.isoformat(),))
            db.execute("INSERT INTO app_login_sessions VALUES (?, ?, ?, ?)",
                       (hashlib.sha256(token.encode()).hexdigest(), user["id"],
                        (now + timedelta(days=SESSION_DAYS)).isoformat(), now.isoformat()))
        return token, {"id": user["id"], "username": user["username"]}

    def user_for_token(self, token):
        if not isinstance(token, str) or len(token) > 128:
            return None
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        now = self._now().isoformat()
        with closing(self._connect()) as db, db:
            row = db.execute("SELECT u.id, u.username FROM app_login_sessions s JOIN app_users u ON u.id=s.user_id "
                             "WHERE s.token_hash=? AND s.expires_at>?", (token_hash, now)).fetchone()
        return {"id": row["id"], "username": row["username"]} if row else None

    def logout(self, token):
        if isinstance(token, str) and len(token) <= 128:
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            with closing(self._connect()) as db, db:
                db.execute("DELETE FROM app_login_sessions WHERE token_hash=?", (token_hash,))


def auth_required():
    return os.getenv("AUTH_REQUIRED", "true").strip().lower() not in {"0", "false", "no"}


def cookie_secure():
    return os.getenv("AUTH_COOKIE_SECURE", "false").strip().lower() in {"1", "true", "yes"}
