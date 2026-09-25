import pytest
from fastapi.testclient import TestClient

from app import main
from app.providers.base import ProviderError


class _FakeProvider:
    def __init__(self, *args, **kwargs):
        pass

    def stream(self, messages, system_prompt, model, params):
        yield "Hel"
        yield "lo"


class _FailingProvider:
    def __init__(self, *args, **kwargs):
        pass

    def stream(self, messages, system_prompt, model, params):
        raise ProviderError("upstream failed")
        yield  # pragma: no cover - never reached, makes this a generator


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setitem(main.PROVIDER_CLASSES, "openai", _FakeProvider)
    return TestClient(main.app)


def test_send_message_streams_tokens_and_persists(client):
    created = client.post("/api/conversations", json={}).json()
    response = client.post(f"/api/conversations/{created['id']}/messages", json={"content": "hi"})
    assert response.status_code == 200
    assert "event: token" in response.text
    assert "event: done" in response.text

    detail = client.get(f"/api/conversations/{created['id']}").json()
    roles = [m["role"] for m in detail["messages"]]
    assert roles == ["user", "assistant"]
    assert detail["messages"][1]["content"] == "Hello"


def test_send_message_persists_error_on_provider_failure(client, monkeypatch):
    monkeypatch.setitem(main.PROVIDER_CLASSES, "openai", _FailingProvider)
    created = client.post("/api/conversations", json={}).json()
    response = client.post(f"/api/conversations/{created['id']}/messages", json={"content": "hi"})
    assert "event: error" in response.text

    detail = client.get(f"/api/conversations/{created['id']}").json()
    roles = [m["role"] for m in detail["messages"]]
    assert roles == ["user", "error"]


def test_send_message_to_missing_conversation_returns_404(client):
    response = client.post("/api/conversations/does-not-exist/messages", json={"content": "hi"})
    assert response.status_code == 404
