import pytest

from app.providers.base import ChatParams, ProviderError
from app.providers.openai_provider import OpenAIProvider


def test_build_kwargs_maps_shared_params_and_drops_top_k():
    params = ChatParams(
        temperature=0.5, top_p=0.9, top_k=40, max_output_tokens=256,
        seed=7, stop_sequence="END, STOP",
    )
    kwargs = OpenAIProvider._build_kwargs(
        messages=[{"role": "user", "content": "hi"}],
        system_prompt="be nice",
        model="gpt-5.1",
        params=params,
    )
    assert kwargs["model"] == "gpt-5.1"
    assert kwargs["messages"] == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hi"},
    ]
    assert kwargs["temperature"] == 0.5
    assert kwargs["top_p"] == 0.9
    assert kwargs["max_completion_tokens"] == 256
    assert kwargs["seed"] == 7
    assert kwargs["stop"] == ["END", "STOP"]
    assert kwargs["stream"] is True
    assert "top_k" not in kwargs


def test_build_kwargs_omits_seed_and_stop_when_not_set():
    kwargs = OpenAIProvider._build_kwargs(
        messages=[], system_prompt="", model="gpt-5.1", params=ChatParams(),
    )
    assert "seed" not in kwargs
    assert "stop" not in kwargs


class _FakeDelta:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, chunks):
        self._chunks = chunks
        self.received_kwargs = None

    def create(self, **kwargs):
        self.received_kwargs = kwargs
        return iter(self._chunks)


class _FakeChat:
    def __init__(self, chunks):
        self.completions = _FakeCompletions(chunks)


class _FakeClient:
    def __init__(self, chunks):
        self.chat = _FakeChat(chunks)


def test_stream_yields_text_chunks():
    fake_client = _FakeClient([_FakeChunk("Hel"), _FakeChunk("lo"), _FakeChunk(None)])
    provider = OpenAIProvider(client=fake_client)
    result = list(
        provider.stream([{"role": "user", "content": "hi"}], "sys", "gpt-5.1", ChatParams())
    )
    assert result == ["Hel", "lo"]
    assert fake_client.chat.completions.received_kwargs["stream"] is True


def test_stream_wraps_upstream_errors():
    class _RaisingCompletions:
        def create(self, **kwargs):
            raise RuntimeError("boom")

    class _RaisingChat:
        completions = _RaisingCompletions()

    class _RaisingClient:
        chat = _RaisingChat()

    provider = OpenAIProvider(client=_RaisingClient())
    with pytest.raises(ProviderError):
        list(provider.stream([], "sys", "gpt-5.1", ChatParams()))


def test_constructor_raises_provider_error_when_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        OpenAIProvider()
