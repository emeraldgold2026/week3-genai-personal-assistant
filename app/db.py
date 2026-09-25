"""SQLite persistence for conversations and messages. No ORM."""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    system_prompt TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    params_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_conversation(
    conn: sqlite3.Connection, title: str, system_prompt: str, provider: str, model: str, params_json: str,
) -> dict:
    conv_id = str(uuid.uuid4())
    now = _now()
    conn.execute(
        "INSERT INTO conversations (id, title, system_prompt, provider, model, params_json, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (conv_id, title, system_prompt, provider, model, params_json, now, now),
    )
    conn.commit()
    return get_conversation(conn, conv_id)


def list_conversations(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, title, provider, model, updated_at FROM conversations ORDER BY updated_at DESC"
    ).fetchall()
    return [dict(row) for row in rows]


def get_conversation(conn: sqlite3.Connection, conv_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
    return dict(row) if row else None


def update_conversation(conn: sqlite3.Connection, conv_id: str, **fields) -> dict | None:
    if not fields:
        return get_conversation(conn, conv_id)
    fields["updated_at"] = _now()
    columns = ", ".join(f"{key} = ?" for key in fields)
    values = [*fields.values(), conv_id]
    conn.execute(f"UPDATE conversations SET {columns} WHERE id = ?", values)
    conn.commit()
    return get_conversation(conn, conv_id)


def delete_conversation(conn: sqlite3.Connection, conv_id: str) -> None:
    conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit()


def add_message(conn: sqlite3.Connection, conversation_id: str, role: str, content: str) -> dict:
    msg_id = str(uuid.uuid4())
    now = _now()
    conn.execute(
        "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
        (msg_id, conversation_id, role, content, now),
    )
    conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
    conn.commit()
    return {"id": msg_id, "conversation_id": conversation_id, "role": role, "content": content, "created_at": now}


def list_messages(conn: sqlite3.Connection, conversation_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
        (conversation_id,),
    ).fetchall()
    return [dict(row) for row in rows]
