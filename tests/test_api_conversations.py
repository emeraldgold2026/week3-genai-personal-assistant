import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DB_PATH", str(tmp_path / "test.db"))
    return TestClient(main.app)


def test_create_and_get_conversation(client):
    response = client.post("/api/conversations", json={})
    assert response.status_code == 200
    body = response.json()
    conv_id = body["id"]
    assert body["provider"] == "openai"
    assert body["model"] == "gpt-5.1"

    fetched = client.get(f"/api/conversations/{conv_id}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == conv_id


def test_list_conversations(client):
    client.post("/api/conversations", json={})
    client.post("/api/conversations", json={})
    response = client.get("/api/conversations")
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_update_conversation(client):
    created = client.post("/api/conversations", json={}).json()
    response = client.patch(
        f"/api/conversations/{created['id']}",
        json={"system_prompt": "new prompt", "model": "gpt-5.1-mini"},
    )
    assert response.status_code == 200
    assert response.json()["system_prompt"] == "new prompt"
    assert response.json()["model"] == "gpt-5.1-mini"


def test_delete_conversation(client):
    created = client.post("/api/conversations", json={}).json()
    response = client.delete(f"/api/conversations/{created['id']}")
    assert response.status_code == 204
    assert client.get(f"/api/conversations/{created['id']}").status_code == 404


def test_get_missing_conversation_returns_404(client):
    response = client.get("/api/conversations/does-not-exist")
    assert response.status_code == 404
