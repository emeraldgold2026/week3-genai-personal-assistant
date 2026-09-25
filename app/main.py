"""FastAPI app: conversation CRUD routes and static frontend serving."""
from __future__ import annotations

import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import db
from app.models import (
    ConversationCreate,
    ConversationDetail,
    ConversationSummary,
    ConversationUpdate,
    MessageCreate,
    MessageOut,
)
from app.providers.base import ChatParams, ProviderError
from app.providers.gemini_provider import GeminiProvider
from app.providers.openai_provider import OpenAIProvider

DB_PATH = os.environ.get("ASSISTANT_DB_PATH", "assistant.db")

app = FastAPI(title="My Personal Assistant")

PROVIDER_CLASSES = {"openai": OpenAIProvider, "gemini": GeminiProvider}


def get_provider_instance(name: str):
    provider_class = PROVIDER_CLASSES.get(name)
    if provider_class is None:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {name}")
    try:
        return provider_class()
    except ProviderError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def get_db_connection():
    conn = db.get_connection(DB_PATH)
    db.init_db(conn)
    return conn


def _conversation_to_detail(row: dict, messages: list[dict]) -> ConversationDetail:
    return ConversationDetail(
        id=row["id"],
        title=row["title"],
        system_prompt=row["system_prompt"],
        provider=row["provider"],
        model=row["model"],
        params=ChatParams.model_validate_json(row["params_json"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        messages=[MessageOut(**m) for m in messages],
    )


@app.get("/api/conversations", response_model=list[ConversationSummary])
def api_list_conversations():
    conn = get_db_connection()
    try:
        return db.list_conversations(conn)
    finally:
        conn.close()


@app.post("/api/conversations", response_model=ConversationDetail)
def api_create_conversation(body: ConversationCreate):
    conn = get_db_connection()
    try:
        row = db.create_conversation(
            conn,
            title="New chat",
            system_prompt=body.system_prompt,
            provider=body.provider,
            model=body.model,
            params_json=body.params.model_dump_json(),
        )
        return _conversation_to_detail(row, [])
    finally:
        conn.close()


@app.get("/api/conversations/{conv_id}", response_model=ConversationDetail)
def api_get_conversation(conv_id: str):
    conn = get_db_connection()
    try:
        row = db.get_conversation(conn, conv_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        messages = db.list_messages(conn, conv_id)
        return _conversation_to_detail(row, messages)
    finally:
        conn.close()


@app.patch("/api/conversations/{conv_id}", response_model=ConversationDetail)
def api_update_conversation(conv_id: str, body: ConversationUpdate):
    conn = get_db_connection()
    try:
        row = db.get_conversation(conn, conv_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        fields = {}
        if body.system_prompt is not None:
            fields["system_prompt"] = body.system_prompt
        if body.provider is not None:
            fields["provider"] = body.provider
        if body.model is not None:
            fields["model"] = body.model
        if body.params is not None:
            fields["params_json"] = body.params.model_dump_json()
        row = db.update_conversation(conn, conv_id, **fields)
        messages = db.list_messages(conn, conv_id)
        return _conversation_to_detail(row, messages)
    finally:
        conn.close()


@app.delete("/api/conversations/{conv_id}", status_code=204)
def api_delete_conversation(conv_id: str):
    conn = get_db_connection()
    try:
        db.delete_conversation(conn, conv_id)
    finally:
        conn.close()
    return None


@app.post("/api/conversations/{conv_id}/messages")
def api_post_message(conv_id: str, body: MessageCreate):
    # Opens its own short-lived connections rather than a request-scoped
    # dependency, because the generator below runs after this function
    # returns the StreamingResponse — a request-scoped connection would
    # already be closed by then.
    conn = get_db_connection()
    try:
        row = db.get_conversation(conn, conv_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        db.add_message(conn, conv_id, "user", body.content)
        history = db.list_messages(conn, conv_id)
        params = ChatParams.model_validate_json(row["params_json"])
    finally:
        conn.close()

    chat_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m["role"] in ("user", "assistant")
    ]
    provider = get_provider_instance(row["provider"])

    def event_stream():
        collected: list[str] = []
        try:
            for token in provider.stream(chat_messages, row["system_prompt"], row["model"], params):
                collected.append(token)
                yield _sse("token", {"text": token})
        except ProviderError as exc:
            write_conn = get_db_connection()
            try:
                db.add_message(write_conn, conv_id, "error", str(exc))
            finally:
                write_conn.close()
            yield _sse("error", {"message": str(exc)})
            return

        full_text = "".join(collected)
        write_conn = get_db_connection()
        try:
            db.add_message(write_conn, conv_id, "assistant", full_text)
        finally:
            write_conn.close()
        yield _sse("done", {"text": full_text})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Mounted last so it never shadows the /api/* routes above.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
