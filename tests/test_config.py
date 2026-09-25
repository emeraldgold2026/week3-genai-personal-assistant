import pytest

from app import config


def test_get_openai_key_returns_value_when_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    assert config.get_openai_key() == "sk-test-123"


def test_get_openai_key_raises_when_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(config.MissingAPIKeyError):
        config.get_openai_key()


def test_get_gemini_key_returns_value_when_set(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gk-test-456")
    assert config.get_gemini_key() == "gk-test-456"


def test_get_gemini_key_raises_when_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(config.MissingAPIKeyError):
        config.get_gemini_key()
