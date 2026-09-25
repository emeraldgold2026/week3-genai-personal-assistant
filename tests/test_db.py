import pytest

from app import db


@pytest.fixture
def conn():
    connection = db.get_connection(":memory:")
    db.init_db(connection)
    yield connection
    connection.close()


def test_create_and_get_conversation(conn):
    created = db.create_conversation(conn, "New chat", "be nice", "openai", "gpt-5.1", '{"temperature": 1.0}')
    fetched = db.get_conversation(conn, created["id"])
    assert fetched["title"] == "New chat"
    assert fetched["system_prompt"] == "be nice"
    assert fetched["provider"] == "openai"
    assert fetched["model"] == "gpt-5.1"


def test_get_conversation_returns_none_when_missing(conn):
    assert db.get_conversation(conn, "does-not-exist") is None


def test_list_conversations_orders_by_updated_at_desc(conn):
    first = db.create_conversation(conn, "First", "", "openai", "gpt-5.1", "{}")
    second = db.create_conversation(conn, "Second", "", "openai", "gpt-5.1", "{}")
    conversations = db.list_conversations(conn)
    assert [c["id"] for c in conversations] == [second["id"], first["id"]]


def test_update_conversation(conn):
    created = db.create_conversation(conn, "Title", "", "openai", "gpt-5.1", "{}")
    updated = db.update_conversation(conn, created["id"], system_prompt="new prompt", model="gpt-5.1-mini")
    assert updated["system_prompt"] == "new prompt"
    assert updated["model"] == "gpt-5.1-mini"


def test_delete_conversation_cascades_messages(conn):
    created = db.create_conversation(conn, "Title", "", "openai", "gpt-5.1", "{}")
    db.add_message(conn, created["id"], "user", "hi")
    db.delete_conversation(conn, created["id"])
    assert db.get_conversation(conn, created["id"]) is None
    assert db.list_messages(conn, created["id"]) == []


def test_add_and_list_messages(conn):
    created = db.create_conversation(conn, "Title", "", "openai", "gpt-5.1", "{}")
    db.add_message(conn, created["id"], "user", "hi")
    db.add_message(conn, created["id"], "assistant", "hello")
    messages = db.list_messages(conn, created["id"])
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert [m["content"] for m in messages] == ["hi", "hello"]


def test_add_message_updates_conversation_updated_at(conn):
    created = db.create_conversation(conn, "Title", "", "openai", "gpt-5.1", "{}")
    original_updated_at = created["updated_at"]
    db.add_message(conn, created["id"], "user", "hi")
    fetched = db.get_conversation(conn, created["id"])
    assert fetched["updated_at"] >= original_updated_at
