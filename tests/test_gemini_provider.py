import pytest

from app.providers.base import ChatParams, ProviderError
from app.providers.gemini_provider import GeminiProvider


def test_to_contents_maps_assistant_role_to_model():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert GeminiProvider._to_contents(messages) == [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"text": "hello"}]},
    ]


def test_build_config_maps_shared_params():
    params = ChatParams(
        temperature=0.3, top_p=0.8, top_k=20, max_output_tokens=512,
        seed=9, stop_sequence="END, STOP",
    )
    config = GeminiProvider._build_config("be nice", params)
    assert config.temperature == 0.3
    assert config.top_p == 0.8
    assert config.top_k == 20
    assert config.max_output_tokens == 512
    assert config.seed == 9
    assert config.stop_sequences == ["END", "STOP"]
    assert config.system_instruction == "be nice"


def test_build_config_omits_stop_sequences_when_not_set():
    config = GeminiProvider._build_config("", ChatParams())
    assert config.stop_sequences is None


class _FakeChunk:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    def __init__(self, chunks):
        self._chunks = chunks
        self.received_kwargs = None

    def generate_content_stream(self, **kwargs):
        self.received_kwargs = kwargs
        return iter(self._chunks)


class _FakeClient:
    def __init__(self, chunks):
        self.models = _FakeModels(chunks)


def test_stream_yields_text_chunks():
    fake_client = _FakeClient([_FakeChunk("Hel"), _FakeChunk("lo"), _FakeChunk(None)])
    provider = GeminiProvider(client=fake_client)
    result = list(
        provider.stream([{"role": "user", "content": "hi"}], "sys", "gemini-3-pro", ChatParams())
    )
    assert result == ["Hel", "lo"]
    assert fake_client.models.received_kwargs["model"] == "gemini-3-pro"


def test_stream_wraps_upstream_errors():
    class _RaisingModels:
        def generate_content_stream(self, **kwargs):
            raise RuntimeError("boom")

    class _RaisingClient:
        models = _RaisingModels()

    provider = GeminiProvider(client=_RaisingClient())
    with pytest.raises(ProviderError):
        list(provider.stream([], "sys", "gemini-3-pro", ChatParams()))


def test_constructor_raises_provider_error_when_key_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        GeminiProvider()
