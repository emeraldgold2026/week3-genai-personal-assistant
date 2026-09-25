# My Personal Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local web app ("My Personal Assistant") for chatting with OpenAI and Gemini models using the user's own API keys, with a Playground-style parameter panel and persistent conversation history.

**Architecture:** A FastAPI backend (`app/`) owns both API keys and provider SDK calls behind a shared `ChatProvider` interface, persists conversations/messages to SQLite, and streams model output to a static vanilla-JS frontend (`static/`) over Server-Sent Events.

**Tech Stack:** Python (this machine's environment is 3.9.6 — see Global Constraints), FastAPI, Uvicorn, Pydantic v2, `sqlite3` (stdlib, no ORM), `openai` SDK, `google-genai` SDK, `python-dotenv`, pytest, vanilla HTML/CSS/JS (no frontend build step).

**Spec:** `docs/superpowers/specs/2026-09-24-personal-assistant-design.md`

## Global Constraints

- API keys (`OPENAI_API_KEY`, `GEMINI_API_KEY`) come only from a local `.env` file, loaded server-side via `python-dotenv`; they are never sent to, or accepted from, the browser, and never logged or persisted to SQLite.
- Models offered are a curated static list: OpenAI `gpt-5.1`, `gpt-5.1-mini`; Gemini `gemini-3-pro`, `gemini-3-flash`. No dynamic model fetching in v1.
- Shared parameters are exactly: `temperature`, `top_p`, `top_k`, `max_output_tokens`, `seed`, `stop_sequence`. `top_k` is Gemini-only — the OpenAI provider must never pass it to the OpenAI API, and the frontend must hide/disable it when the OpenAI provider is selected.
- No ORM — `app/db.py` uses the stdlib `sqlite3` module directly with parameterized queries.
- No tool/function calling, JSON mode, file attachments, auth, or dynamic model listing in v1 (out of scope per spec).
- Every backend module lives under `app/`; every frontend file lives under `static/`; every test lives under `tests/`, mirroring the module it tests.
- **Environment correction (discovered during Task 2, ruled during execution):** this machine's Python is 3.9.6, not 3.11+ as originally stated. PEP 604 union syntax (`X | None`) still applies verbatim as written in every task below, but every Python file that uses it as a class-level or function-signature annotation must start with `from __future__ import annotations` (defers annotation evaluation to strings), and `requirements.txt` includes `eval_type_backport>=0.4` so Pydantic can resolve those deferred annotations on Python <3.10. `list[dict]`/`list[str]` subscripts do not need this (PEP 585 works natively on 3.9) — only actual `X | Y` unions do. Every task's code block below has already been updated with this import where needed; implementers should not need to rediscover this.

## Task Dependency Graph

```
Task 1 (scaffolding + config)
  ├── Task 2 (provider base)         ─┬── Task 3 (OpenAI provider)   ─┐
  │                                    └── Task 4 (Gemini provider)   ─┼─ Task 8 (chat streaming API)
  ├── Task 5 (db layer)              ──────────────────────────────────┤
  ├── Task 6 (API models, needs Task 2 for ChatParams) ─── Task 7 (conversation CRUD API) ─┘
  └── Task 9 (frontend HTML/CSS)     ─── Task 10 (frontend JS, needs Task 7 & 8's API contract)

Task 11 (README + end-to-end check) depends on everything above.
```

Tasks 2, 5, and 9 can start in parallel once Task 1 is committed. Tasks 3 and 4 can run in parallel once Task 2 is committed. Task 6 can run in parallel with 3/4/5 once Task 2 is committed. Task 10's code can be written in parallel with 7/8, but its manual verification step needs a running server with Tasks 7+8 complete.

---

### Task 1: Project scaffolding & API key config

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `app/__init__.py`
- Create: `app/config.py`
- Create: `static/index.html` (placeholder, replaced in Task 9)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `app.config.get_openai_key() -> str`, `app.config.get_gemini_key() -> str`, `app.config.MissingAPIKeyError` (Exception subclass). Later provider tasks import and call these.

- [ ] **Step 1: Create the non-Python project files**

`requirements.txt`:
```
fastapi>=0.115
uvicorn[standard]>=0.32
pydantic>=2.9
eval_type_backport>=0.4
python-dotenv>=1.0
openai>=1.55
google-genai>=0.3
pytest>=8.3
httpx>=0.27
```

`.env.example`:
```
# OpenAI API key - https://platform.openai.com/api-keys
OPENAI_API_KEY=

# Google Gemini API key - https://aistudio.google.com/apikey
GEMINI_API_KEY=
```

`.gitignore`:
```
__pycache__/
*.pyc
.env
assistant.db
.venv/
venv/
```

`static/index.html` (placeholder so `StaticFiles` has a directory to serve in later tasks):
```html
<!doctype html>
<html>
  <head><title>My Personal Assistant</title></head>
  <body><p>Coming soon.</p></body>
</html>
```

- [ ] **Step 2: Write the failing test for config**

`tests/test_config.py`:
```python
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
```

- [ ] **Step 3: Create `app/__init__.py` and set up a virtualenv, then run the test to verify it fails**

```bash
touch app/__init__.py
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/test_config.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.config'` (or `AttributeError`).

- [ ] **Step 4: Implement `app/config.py`**

```python
"""Loads API keys from a local .env file. Never logs or exposes key values."""
import os

from dotenv import load_dotenv

load_dotenv()


class MissingAPIKeyError(Exception):
    """Raised when a required API key is not set in the environment."""


def get_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise MissingAPIKeyError(
            "OPENAI_API_KEY not set — add it to your .env file and restart."
        )
    return key


def get_gemini_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise MissingAPIKeyError(
            "GEMINI_API_KEY not set — add it to your .env file and restart."
        )
    return key
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
pytest tests/test_config.py -v
```
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .env.example .gitignore app/__init__.py app/config.py static/index.html tests/test_config.py
git commit -m "Add project scaffolding and API key config loading"
```

---

### Task 2: ChatProvider base interface & ChatParams

**Files:**
- Create: `app/providers/__init__.py`
- Create: `app/providers/base.py`
- Test: `tests/test_providers_base.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `app.providers.base.ChatParams` (Pydantic model with fields `temperature: float = 1.0`, `top_p: float = 1.0`, `top_k: int | None = None`, `max_output_tokens: int = 1024`, `seed: int | None = None`, `stop_sequence: str | None = None`), `app.providers.base.ProviderError` (Exception), `app.providers.base.ChatProvider` (Protocol with `stream(messages: list[dict], system_prompt: str, model: str, params: ChatParams) -> Iterator[str]`). Tasks 3, 4, 6, 7, 8 all import from here.

- [ ] **Step 1: Write the failing test**

`tests/test_providers_base.py`:
```python
import pytest
from pydantic import ValidationError

from app.providers.base import ChatParams


def test_chat_params_defaults():
    params = ChatParams()
    assert params.temperature == 1.0
    assert params.top_p == 1.0
    assert params.top_k is None
    assert params.max_output_tokens == 1024
    assert params.seed is None
    assert params.stop_sequence is None


def test_chat_params_rejects_temperature_out_of_range():
    with pytest.raises(ValidationError):
        ChatParams(temperature=2.5)


def test_chat_params_rejects_top_p_out_of_range():
    with pytest.raises(ValidationError):
        ChatParams(top_p=1.5)


def test_chat_params_rejects_top_k_below_one():
    with pytest.raises(ValidationError):
        ChatParams(top_k=0)


def test_chat_params_rejects_non_positive_max_output_tokens():
    with pytest.raises(ValidationError):
        ChatParams(max_output_tokens=0)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
mkdir -p app/providers && touch app/providers/__init__.py
pytest tests/test_providers_base.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.providers.base'`.

- [ ] **Step 3: Implement `app/providers/base.py`**

```python
"""Shared provider interface and parameters for chat model calls."""
from __future__ import annotations

from typing import Iterator, Protocol

from pydantic import BaseModel, field_validator


class ChatParams(BaseModel):
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None          # Gemini only; ignored by the OpenAI provider
    max_output_tokens: int = 1024
    seed: int | None = None
    stop_sequence: str | None = None  # comma-separated for multiple stop strings

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        return v

    @field_validator("top_p")
    @classmethod
    def _check_top_p(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("top_p must be between 0 and 1")
        return v

    @field_validator("top_k")
    @classmethod
    def _check_top_k(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("top_k must be >= 1")
        return v

    @field_validator("max_output_tokens")
    @classmethod
    def _check_max_output_tokens(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_output_tokens must be >= 1")
        return v


class ProviderError(Exception):
    """Raised when a provider is misconfigured or the upstream API call fails."""


class ChatProvider(Protocol):
    def stream(
        self,
        messages: list[dict],
        system_prompt: str,
        model: str,
        params: ChatParams,
    ) -> Iterator[str]:
        ...
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_providers_base.py -v
```
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add app/providers/__init__.py app/providers/base.py tests/test_providers_base.py
git commit -m "Add ChatProvider interface and ChatParams model"
```

---

### Task 3: OpenAI provider

**Files:**
- Create: `app/providers/openai_provider.py`
- Test: `tests/test_openai_provider.py`

**Interfaces:**
- Consumes: `app.providers.base.ChatParams`, `ProviderError` (Task 2); `app.config.get_openai_key`, `app.config.MissingAPIKeyError` (Task 1).
- Produces: `app.providers.openai_provider.OpenAIProvider`, constructor `OpenAIProvider(api_key: str | None = None, client=None)`, method `.stream(messages, system_prompt, model, params) -> Iterator[str]`, static method `._build_kwargs(messages, system_prompt, model, params) -> dict`. Task 8 imports `OpenAIProvider`.

- [ ] **Step 1: Write the failing test**

`tests/test_openai_provider.py`:
```python
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
    assert kwargs["max_tokens"] == 256
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_openai_provider.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.providers.openai_provider'`.

- [ ] **Step 3: Implement `app/providers/openai_provider.py`**

```python
"""OpenAI chat completion adapter implementing the ChatProvider interface."""
from __future__ import annotations

from typing import Iterator

from openai import OpenAI

from app.config import MissingAPIKeyError, get_openai_key
from app.providers.base import ChatParams, ProviderError


class OpenAIProvider:
    def __init__(self, api_key: str | None = None, client=None):
        if client is not None:
            self._client = client
            return
        try:
            key = api_key or get_openai_key()
        except MissingAPIKeyError as exc:
            raise ProviderError(str(exc)) from exc
        self._client = OpenAI(api_key=key)

    def stream(
        self,
        messages: list[dict],
        system_prompt: str,
        model: str,
        params: ChatParams,
    ) -> Iterator[str]:
        kwargs = self._build_kwargs(messages, system_prompt, model, params)
        try:
            response_stream = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ProviderError(str(exc)) from exc
        for chunk in response_stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    @staticmethod
    def _build_kwargs(
        messages: list[dict], system_prompt: str, model: str, params: ChatParams
    ) -> dict:
        kwargs: dict = {
            "model": model,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_output_tokens,
            "stream": True,
        }
        if params.seed is not None:
            kwargs["seed"] = params.seed
        if params.stop_sequence:
            kwargs["stop"] = [s.strip() for s in params.stop_sequence.split(",") if s.strip()]
        return kwargs
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_openai_provider.py -v
```
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add app/providers/openai_provider.py tests/test_openai_provider.py
git commit -m "Add OpenAI chat provider"
```

---

### Task 4: Gemini provider

**Files:**
- Create: `app/providers/gemini_provider.py`
- Test: `tests/test_gemini_provider.py`

**Interfaces:**
- Consumes: `app.providers.base.ChatParams`, `ProviderError` (Task 2); `app.config.get_gemini_key`, `app.config.MissingAPIKeyError` (Task 1).
- Produces: `app.providers.gemini_provider.GeminiProvider`, constructor `GeminiProvider(api_key: str | None = None, client=None)`, method `.stream(messages, system_prompt, model, params) -> Iterator[str]`, static methods `._build_config(system_prompt, params)` and `._to_contents(messages) -> list[dict]`. Task 8 imports `GeminiProvider`.

- [ ] **Step 1: Write the failing test**

`tests/test_gemini_provider.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_gemini_provider.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.providers.gemini_provider'`.

- [ ] **Step 3: Implement `app/providers/gemini_provider.py`**

```python
"""Gemini chat adapter implementing the ChatProvider interface."""
from __future__ import annotations

from typing import Iterator

from google import genai
from google.genai import types

from app.config import MissingAPIKeyError, get_gemini_key
from app.providers.base import ChatParams, ProviderError

_ROLE_MAP = {"user": "user", "assistant": "model"}


class GeminiProvider:
    def __init__(self, api_key: str | None = None, client=None):
        if client is not None:
            self._client = client
            return
        try:
            key = api_key or get_gemini_key()
        except MissingAPIKeyError as exc:
            raise ProviderError(str(exc)) from exc
        self._client = genai.Client(api_key=key)

    def stream(
        self,
        messages: list[dict],
        system_prompt: str,
        model: str,
        params: ChatParams,
    ) -> Iterator[str]:
        config = self._build_config(system_prompt, params)
        contents = self._to_contents(messages)
        try:
            response_stream = self._client.models.generate_content_stream(
                model=model, contents=contents, config=config,
            )
        except Exception as exc:
            raise ProviderError(str(exc)) from exc
        for chunk in response_stream:
            text = getattr(chunk, "text", None)
            if text:
                yield text

    @staticmethod
    def _build_config(system_prompt: str, params: ChatParams) -> types.GenerateContentConfig:
        stop_sequences = None
        if params.stop_sequence:
            stop_sequences = [s.strip() for s in params.stop_sequence.split(",") if s.strip()]
        return types.GenerateContentConfig(
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            max_output_tokens=params.max_output_tokens,
            seed=params.seed,
            stop_sequences=stop_sequences,
            system_instruction=system_prompt,
        )

    @staticmethod
    def _to_contents(messages: list[dict]) -> list[dict]:
        return [
            {"role": _ROLE_MAP.get(m["role"], "user"), "parts": [{"text": m["content"]}]}
            for m in messages
        ]
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_gemini_provider.py -v
```
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add app/providers/gemini_provider.py tests/test_gemini_provider.py
git commit -m "Add Gemini chat provider"
```

---

### Task 5: SQLite persistence layer

**Files:**
- Create: `app/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: nothing beyond the stdlib.
- Produces: `app.db.get_connection(db_path: str) -> sqlite3.Connection`, `app.db.init_db(conn)`, `app.db.create_conversation(conn, title, system_prompt, provider, model, params_json) -> dict`, `app.db.list_conversations(conn) -> list[dict]`, `app.db.get_conversation(conn, conv_id) -> dict | None`, `app.db.update_conversation(conn, conv_id, **fields) -> dict | None`, `app.db.delete_conversation(conn, conv_id) -> None`, `app.db.add_message(conn, conversation_id, role, content) -> dict`, `app.db.list_messages(conn, conversation_id) -> list[dict]`. Each dict uses the exact column names from the spec's schema (`id`, `title`, `system_prompt`, `provider`, `model`, `params_json`, `created_at`, `updated_at` for conversations; `id`, `conversation_id`, `role`, `content`, `created_at` for messages). Tasks 7 and 8 call these directly.

- [ ] **Step 1: Write the failing test**

`tests/test_db.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_db.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.db'`.

- [ ] **Step 3: Implement `app/db.py`**

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_db.py -v
```
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add app/db.py tests/test_db.py
git commit -m "Add SQLite persistence layer for conversations and messages"
```

---

### Task 6: API request/response models

**Files:**
- Create: `app/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `app.providers.base.ChatParams` (Task 2).
- Produces: `app.models.ConversationCreate` (fields `provider: str = "openai"`, `model: str = "gpt-5.1"`, `system_prompt: str = ""`, `params: ChatParams = ChatParams()`), `app.models.ConversationUpdate` (all fields optional: `system_prompt`, `provider`, `model`, `params`), `app.models.ConversationSummary` (`id`, `title`, `provider`, `model`, `updated_at`, all `str`), `app.models.MessageOut` (`id`, `role`, `content`, `created_at`, all `str`), `app.models.ConversationDetail` (`id`, `title`, `system_prompt`, `provider`, `model`, `params: ChatParams`, `created_at`, `updated_at`, `messages: list[MessageOut]`), `app.models.MessageCreate` (`content: str`). Tasks 7 and 8 use all of these as FastAPI request/response types.

- [ ] **Step 1: Write the failing test**

`tests/test_models.py`:
```python
from app.models import ConversationCreate, ConversationDetail, ConversationUpdate, MessageOut
from app.providers.base import ChatParams


def test_conversation_create_defaults():
    created = ConversationCreate()
    assert created.provider == "openai"
    assert created.model == "gpt-5.1"
    assert created.system_prompt == ""
    assert created.params == ChatParams()


def test_conversation_create_accepts_custom_params():
    created = ConversationCreate(provider="gemini", model="gemini-3-pro", params=ChatParams(temperature=0.2))
    assert created.provider == "gemini"
    assert created.params.temperature == 0.2


def test_conversation_update_fields_are_optional():
    update = ConversationUpdate()
    assert update.system_prompt is None
    assert update.provider is None
    assert update.model is None
    assert update.params is None


def test_conversation_detail_round_trip():
    detail = ConversationDetail(
        id="abc", title="Title", system_prompt="sys", provider="openai", model="gpt-5.1",
        params=ChatParams(), created_at="t1", updated_at="t2",
        messages=[MessageOut(id="m1", role="user", content="hi", created_at="t1")],
    )
    dumped = detail.model_dump()
    assert dumped["messages"][0]["content"] == "hi"
    assert dumped["params"]["temperature"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_models.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.models'`.

- [ ] **Step 3: Implement `app/models.py`**

```python
"""Pydantic request/response schemas for the HTTP API."""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.providers.base import ChatParams


class ConversationCreate(BaseModel):
    provider: str = "openai"
    model: str = "gpt-5.1"
    system_prompt: str = ""
    params: ChatParams = Field(default_factory=ChatParams)


class ConversationUpdate(BaseModel):
    system_prompt: str | None = None
    provider: str | None = None
    model: str | None = None
    params: ChatParams | None = None


class ConversationSummary(BaseModel):
    id: str
    title: str
    provider: str
    model: str
    updated_at: str


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: str


class ConversationDetail(BaseModel):
    id: str
    title: str
    system_prompt: str
    provider: str
    model: str
    params: ChatParams
    created_at: str
    updated_at: str
    messages: list[MessageOut]


class MessageCreate(BaseModel):
    content: str
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_models.py -v
```
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add app/models.py tests/test_models.py
git commit -m "Add API request/response models"
```

---

### Task 7: Conversation CRUD API

**Files:**
- Create: `app/main.py`
- Test: `tests/test_api_conversations.py`

**Interfaces:**
- Consumes: `app.db.*` (Task 5), `app.models.*` (Task 6), `static/index.html` (Task 1 placeholder, replaced in Task 9).
- Produces: `app.main.app` (the FastAPI instance), `app.main.DB_PATH` (module-level string, monkeypatchable by tests), `app.main.get_db_connection() -> sqlite3.Connection` (opens+inits a connection against `DB_PATH`), routes `GET/POST /api/conversations`, `GET/PATCH/DELETE /api/conversations/{conv_id}`. Task 8 adds the `/messages` route and the `PROVIDER_CLASSES` dict to this same file.

- [ ] **Step 1: Write the failing test**

`tests/test_api_conversations.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_api_conversations.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.main'`.

- [ ] **Step 3: Implement `app/main.py`**

```python
"""FastAPI app: conversation CRUD routes and static frontend serving."""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app import db
from app.models import (
    ConversationCreate,
    ConversationDetail,
    ConversationSummary,
    ConversationUpdate,
    MessageOut,
)
from app.providers.base import ChatParams

DB_PATH = os.environ.get("ASSISTANT_DB_PATH", "assistant.db")

app = FastAPI(title="My Personal Assistant")


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


# Mounted last so it never shadows the /api/* routes above.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_api_conversations.py -v
```
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_api_conversations.py
git commit -m "Add conversation CRUD API routes"
```

---

### Task 8: Chat streaming API

**Files:**
- Modify: `app/main.py` (append to the file created in Task 7, insert before the `app.mount(...)` line at the bottom)
- Test: `tests/test_api_messages.py`

**Interfaces:**
- Consumes: `app.main.app`, `app.main.get_db_connection`, `app.main.DB_PATH` (Task 7); `app.db.*` (Task 5); `app.providers.openai_provider.OpenAIProvider`, `app.providers.gemini_provider.GeminiProvider`, `app.providers.base.ProviderError` (Tasks 3, 4).
- Produces: `app.main.PROVIDER_CLASSES` (dict `{"openai": OpenAIProvider, "gemini": GeminiProvider}`, monkeypatchable by tests), `app.main.get_provider_instance(name: str)`, route `POST /api/conversations/{conv_id}/messages` returning a `text/event-stream` response with `event: token`, `event: done`, and `event: error` frames.

- [ ] **Step 1: Write the failing test**

`tests/test_api_messages.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_api_messages.py -v
```
Expected: FAIL with `AttributeError: module 'app.main' has no attribute 'PROVIDER_CLASSES'`.

- [ ] **Step 3: Add the streaming route to `app/main.py`**

Add these imports near the top of `app/main.py` (alongside the existing ones from Task 7):
```python
import json

from fastapi.responses import StreamingResponse

from app.models import MessageCreate
from app.providers.base import ProviderError
from app.providers.gemini_provider import GeminiProvider
from app.providers.openai_provider import OpenAIProvider
```

Add this constant and helper right after the `app = FastAPI(...)` line:
```python
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
```

Insert the route below `api_delete_conversation` and above the final `app.mount(...)` line:
```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_api_messages.py -v
```
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full backend test suite to confirm nothing broke**

```bash
pytest -v
```
Expected: PASS (all tests across all `tests/*.py` files so far).

- [ ] **Step 6: Commit**

```bash
git add app/main.py tests/test_api_messages.py
git commit -m "Add chat streaming API route with SSE"
```

---

### Task 9: Frontend HTML/CSS skeleton

**Files:**
- Modify: `static/index.html` (replaces the Task 1 placeholder)
- Create: `static/style.css`

**Interfaces:**
- Consumes: nothing (pure markup/styling; `app.js` is added in Task 10 and referenced here via `<script>` but doesn't need to exist yet for this task's verification).
- Produces: DOM element IDs that Task 10's JavaScript binds to: `#banner`, `#sidebar-list`, `#new-chat-btn`, `#system-prompt`, `#message-thread`, `#message-input`, `#send-btn`, `#provider-select`, `#model-select`, `#temperature`, `#top-p`, `#top-k-row`, `#top-k`, `#max-output-tokens`, `#seed`, `#stop-sequence`, `#stream-toggle`.

- [ ] **Step 1: Write `static/index.html`**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>My Personal Assistant</title>
    <link rel="stylesheet" href="style.css" />
  </head>
  <body>
    <div id="banner" class="banner" hidden></div>
    <div class="layout">
      <aside class="sidebar">
        <button id="new-chat-btn">+ New Chat</button>
        <ul id="sidebar-list"></ul>
      </aside>

      <main class="chat-panel">
        <details class="system-prompt-box" open>
          <summary>SYSTEM</summary>
          <textarea id="system-prompt" placeholder="You are my personal assistant"></textarea>
        </details>

        <div id="message-thread" class="message-thread"></div>

        <div class="input-row">
          <textarea id="message-input" placeholder="Type a message..."></textarea>
          <button id="send-btn">Submit</button>
        </div>
      </main>

      <aside class="params-panel">
        <h2>Parameters</h2>

        <label for="provider-select">Provider</label>
        <select id="provider-select">
          <option value="openai">OpenAI</option>
          <option value="gemini">Gemini</option>
        </select>

        <label for="model-select">Model</label>
        <select id="model-select"></select>

        <label for="temperature">Temperature</label>
        <input type="range" id="temperature" min="0" max="2" step="0.01" value="1" />

        <label for="top-p">Top P</label>
        <input type="range" id="top-p" min="0" max="1" step="0.01" value="1" />

        <div id="top-k-row">
          <label for="top-k">Top K</label>
          <input type="number" id="top-k" min="1" step="1" placeholder="(optional)" />
        </div>

        <label for="max-output-tokens">Max Output Tokens</label>
        <input type="number" id="max-output-tokens" min="1" step="1" value="1024" />

        <label for="seed">Seed</label>
        <input type="number" id="seed" step="1" placeholder="(optional)" />

        <label for="stop-sequence">Stop Sequence</label>
        <input type="text" id="stop-sequence" placeholder="comma-separated (optional)" />

        <label class="toggle-label">
          <input type="checkbox" id="stream-toggle" checked />
          Stream
        </label>
      </aside>
    </div>
    <script src="app.js" defer></script>
  </body>
</html>
```

- [ ] **Step 2: Write `static/style.css`**

```css
:root {
  color-scheme: light dark;
  --border: #d8d8d8;
  --bg: #ffffff;
  --bg-muted: #f5f5f5;
  --text: #1a1a1a;
  --accent: #e35c3f;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--text);
  background: var(--bg);
}

.layout {
  display: grid;
  grid-template-columns: 220px 1fr 280px;
  height: 100vh;
}

.sidebar, .params-panel {
  background: var(--bg-muted);
  padding: 16px;
  overflow-y: auto;
}

.sidebar { border-right: 1px solid var(--border); }
.params-panel { border-left: 1px solid var(--border); }

#new-chat-btn {
  width: 100%;
  padding: 8px;
  margin-bottom: 12px;
  cursor: pointer;
}

#sidebar-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

#sidebar-list li {
  padding: 8px;
  border-radius: 6px;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

#sidebar-list li:hover, #sidebar-list li.active {
  background: #e8e8e8;
}

.chat-panel {
  display: flex;
  flex-direction: column;
  padding: 16px;
  min-width: 0;
}

.system-prompt-box {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 12px;
  margin-bottom: 12px;
}

.system-prompt-box textarea {
  width: 100%;
  min-height: 60px;
  margin-top: 8px;
  border: none;
  resize: vertical;
  font-family: inherit;
}

.message-thread {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 8px 0;
}

.message {
  max-width: 70%;
  padding: 10px 14px;
  border-radius: 12px;
  white-space: pre-wrap;
}

.message.user { align-self: flex-end; background: var(--accent); color: white; }
.message.assistant { align-self: flex-start; background: var(--bg-muted); }
.message.error { align-self: flex-start; background: #fde2e2; color: #7a1f1f; }

.input-row {
  display: flex;
  gap: 8px;
  margin-top: 12px;
}

.input-row textarea {
  flex: 1;
  min-height: 48px;
  font-family: inherit;
}

.input-row button, #send-btn {
  padding: 0 20px;
  cursor: pointer;
}

.params-panel label {
  display: block;
  margin-top: 12px;
  font-size: 0.85em;
  font-weight: 600;
}

.params-panel input, .params-panel select {
  width: 100%;
  margin-top: 4px;
}

.toggle-label {
  display: flex !important;
  align-items: center;
  gap: 8px;
}

.toggle-label input { width: auto; margin-top: 0; }

.banner {
  background: #fde2e2;
  color: #7a1f1f;
  padding: 10px 16px;
  text-align: center;
  font-size: 0.9em;
}

.banner[hidden] { display: none; }
```

- [ ] **Step 3: Verify with curl (server must be runnable — it will be after Task 7/8, but the static files can be checked standalone)**

```bash
python3 -c "
import http.server, functools, threading, time, urllib.request
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory='static')
server = http.server.HTTPServer(('127.0.0.1', 8123), handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
time.sleep(0.3)
body = urllib.request.urlopen('http://127.0.0.1:8123/').read().decode()
assert 'My Personal Assistant' in body
assert 'id=\"message-thread\"' in body
assert 'id=\"top-k-row\"' in body
assert 'id=\"banner\"' in body
print('OK: index.html served with expected elements')
server.shutdown()
"
```
Expected output: `OK: index.html served with expected elements`

- [ ] **Step 4: Commit**

```bash
git add static/index.html static/style.css
git commit -m "Add frontend HTML/CSS skeleton"
```

---

### Task 10: Frontend JavaScript (chat, sidebar, parameters)

**Files:**
- Create: `static/app.js`

**Interfaces:**
- Consumes: DOM IDs from Task 9; the HTTP API from Tasks 7-8 (`GET/POST /api/conversations`, `GET/PATCH/DELETE /api/conversations/{id}`, `POST /api/conversations/{id}/messages`).
- Produces: a working browser app; no other module imports this file.

- [ ] **Step 1: Write `static/app.js`**

```javascript
const MODELS_BY_PROVIDER = {
  openai: ["gpt-5.1", "gpt-5.1-mini"],
  gemini: ["gemini-3-pro", "gemini-3-flash"],
};

let currentConversationId = null;

const el = (id) => document.getElementById(id);

function showBanner(message) {
  const banner = el("banner");
  banner.textContent = message;
  banner.hidden = false;
}

function hideBanner() {
  el("banner").hidden = true;
}

// Wraps fetch for the plain (non-streaming) JSON endpoints: shows the
// "can't reach the server" banner on a network failure, and surfaces any
// non-2xx response body's `detail` (e.g. a 422 validation error) as a
// banner instead of failing silently.
async function apiFetch(url, options) {
  let res;
  try {
    res = await fetch(url, options);
  } catch (err) {
    showBanner("Can't reach the assistant server");
    throw err;
  }
  if (res.ok) {
    hideBanner();
    return res;
  }
  const body = await res.json().catch(() => ({}));
  showBanner(body.detail ? String(body.detail) : `Request failed (${res.status})`);
  return res;
}

async function apiListConversations() {
  const res = await apiFetch("/api/conversations");
  return res.json();
}

async function apiCreateConversation() {
  const res = await apiFetch("/api/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  return res.json();
}

async function apiGetConversation(id) {
  const res = await apiFetch(`/api/conversations/${id}`);
  return res.json();
}

async function apiUpdateConversation(id, patch) {
  const res = await apiFetch(`/api/conversations/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  return res.json();
}

async function apiDeleteConversation(id) {
  await apiFetch(`/api/conversations/${id}`, { method: "DELETE" });
}

async function apiSendMessage(id, content, { onToken, onDone, onError }) {
  let res;
  try {
    res = await fetch(`/api/conversations/${id}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
    });
  } catch (err) {
    showBanner("Can't reach the assistant server");
    onError("Can't reach the assistant server");
    return;
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const message = body.detail ? String(body.detail) : `Request failed (${res.status})`;
    onError(message);
    return;
  }
  hideBanner();
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sepIndex;
    while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, sepIndex);
      buffer = buffer.slice(sepIndex + 2);
      const eventLine = frame.split("\n").find((l) => l.startsWith("event: "));
      const dataLine = frame.split("\n").find((l) => l.startsWith("data: "));
      if (!eventLine || !dataLine) continue;
      const event = eventLine.slice("event: ".length);
      const data = JSON.parse(dataLine.slice("data: ".length));
      if (event === "token") onToken(data.text);
      else if (event === "done") onDone(data.text);
      else if (event === "error") onError(data.message);
    }
  }
}

function renderSidebar(conversations) {
  const list = el("sidebar-list");
  list.innerHTML = "";
  for (const conv of conversations) {
    const li = document.createElement("li");
    li.textContent = conv.title || "New chat";
    li.className = conv.id === currentConversationId ? "active" : "";
    li.addEventListener("click", () => loadConversation(conv.id));

    const del = document.createElement("span");
    del.textContent = "✕";
    del.style.cursor = "pointer";
    del.addEventListener("click", async (e) => {
      e.stopPropagation();
      await apiDeleteConversation(conv.id);
      if (conv.id === currentConversationId) currentConversationId = null;
      await refreshSidebar();
    });

    li.appendChild(del);
    list.appendChild(li);
  }
}

async function refreshSidebar() {
  const conversations = await apiListConversations();
  renderSidebar(conversations);
  return conversations;
}

function appendMessage(role, content) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.textContent = content;
  el("message-thread").appendChild(div);
  el("message-thread").scrollTop = el("message-thread").scrollHeight;
  return div;
}

function populateModelOptions(provider, selectedModel) {
  const select = el("model-select");
  select.innerHTML = "";
  for (const model of MODELS_BY_PROVIDER[provider]) {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    select.appendChild(option);
  }
  select.value = selectedModel || MODELS_BY_PROVIDER[provider][0];
}

function updateTopKVisibility(provider) {
  el("top-k-row").style.display = provider === "gemini" ? "block" : "none";
}

function fillParamsPanel(detail) {
  el("provider-select").value = detail.provider;
  populateModelOptions(detail.provider, detail.model);
  updateTopKVisibility(detail.provider);
  el("system-prompt").value = detail.system_prompt || "";
  el("temperature").value = detail.params.temperature;
  el("top-p").value = detail.params.top_p;
  el("top-k").value = detail.params.top_k ?? "";
  el("max-output-tokens").value = detail.params.max_output_tokens;
  el("seed").value = detail.params.seed ?? "";
  el("stop-sequence").value = detail.params.stop_sequence ?? "";
}

function readParamsFromPanel() {
  const provider = el("provider-select").value;
  const topK = el("top-k").value;
  const seed = el("seed").value;
  const stop = el("stop-sequence").value;
  return {
    provider,
    model: el("model-select").value,
    system_prompt: el("system-prompt").value,
    params: {
      temperature: parseFloat(el("temperature").value),
      top_p: parseFloat(el("top-p").value),
      top_k: provider === "gemini" && topK !== "" ? parseInt(topK, 10) : null,
      max_output_tokens: parseInt(el("max-output-tokens").value, 10),
      seed: seed !== "" ? parseInt(seed, 10) : null,
      stop_sequence: stop !== "" ? stop : null,
    },
  };
}

async function loadConversation(id) {
  currentConversationId = id;
  const detail = await apiGetConversation(id);
  fillParamsPanel(detail);
  el("message-thread").innerHTML = "";
  for (const message of detail.messages) {
    appendMessage(message.role, message.content);
  }
  const conversations = await apiListConversations();
  renderSidebar(conversations);
}

async function handleNewChat() {
  const created = await apiCreateConversation();
  await refreshSidebar();
  await loadConversation(created.id);
}

async function handleSend() {
  const input = el("message-input");
  const content = input.value.trim();
  if (!content || !currentConversationId) return;
  input.value = "";
  appendMessage("user", content);
  const assistantDiv = appendMessage("assistant", "");

  await apiSendMessage(currentConversationId, content, {
    onToken: (text) => {
      assistantDiv.textContent += text;
      el("message-thread").scrollTop = el("message-thread").scrollHeight;
    },
    onDone: () => {
      refreshSidebar();
    },
    onError: (message) => {
      assistantDiv.className = "message error";
      assistantDiv.textContent = message;
    },
  });
}

async function handleParamsChanged() {
  if (!currentConversationId) return;
  updateTopKVisibility(el("provider-select").value);
  const patch = readParamsFromPanel();
  await apiUpdateConversation(currentConversationId, patch);
}

function init() {
  el("new-chat-btn").addEventListener("click", handleNewChat);
  el("send-btn").addEventListener("click", handleSend);
  el("message-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  el("provider-select").addEventListener("change", () => {
    populateModelOptions(el("provider-select").value);
    handleParamsChanged();
  });
  for (const id of [
    "model-select", "temperature", "top-p", "top-k",
    "max-output-tokens", "seed", "stop-sequence", "system-prompt",
  ]) {
    el(id).addEventListener("change", handleParamsChanged);
  }

  refreshSidebar().then(async (conversations) => {
    if (conversations.length > 0) {
      await loadConversation(conversations[0].id);
    } else {
      await handleNewChat();
    }
  });
}

init();
```

- [ ] **Step 2: Manual verification (requires real or dummy API keys per Task 1's `.env`)**

```bash
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY and/or GEMINI_API_KEY
uvicorn app.main:app --reload
```
Open `http://127.0.0.1:8000/` in a browser and confirm, in order:
1. A new conversation is created automatically on first load (sidebar shows "New chat").
2. Selecting "Gemini" in the Provider dropdown shows the Top K row; selecting "OpenAI" hides it.
3. Typing a message and pressing Submit (or Enter) appends a user bubble, then streams an assistant reply token-by-token into a new bubble.
4. Clicking "+ New Chat" creates a second conversation, listed above the first, and clicking between them loads the right history.
5. Clicking the ✕ next to a conversation removes it from the sidebar.
6. Reloading the page keeps the conversation list and messages (persistence).
7. Stop the server (`Ctrl+C`) and click "+ New Chat" again — confirm the red "Can't reach the assistant server" banner appears instead of the page silently doing nothing. Restart the server and confirm the banner does not reappear on the next successful action.
8. Open the browser's developer console and confirm there are no JavaScript errors during the above steps.

- [ ] **Step 3: Commit**

```bash
git add static/app.js
git commit -m "Add frontend chat, sidebar, and parameters JavaScript"
```

---

### Task 11: README and end-to-end verification

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: the whole app (Tasks 1-10).
- Produces: setup documentation for a human running the app.

- [ ] **Step 1: Write `README.md`**

```markdown
# My Personal Assistant

A local personal AI assistant web app for chatting with your own OpenAI and
Gemini models, using your own API keys.

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```
   - Get an OpenAI key at https://platform.openai.com/api-keys
   - Get a Gemini key at https://aistudio.google.com/apikey

   You only need to fill in the key(s) for the provider(s) you plan to use;
   the other provider will show a friendly error if selected without a key.
3. Run the server:
   ```bash
   uvicorn app.main:app --reload
   ```
4. Open http://127.0.0.1:8000/ in your browser.

Your API keys stay in `.env` on your machine and are never sent anywhere
except directly to OpenAI/Google from the backend process.

## Running tests

```bash
pytest -v
```

## Manual smoke test

With both API keys set in `.env`:
1. Start a new chat, leave the system prompt as-is or edit it, select the
   OpenAI provider and send a message — confirm a streamed reply appears.
2. Switch the provider to Gemini on a new chat, adjust Top K, and send a
   message — confirm a streamed reply appears and Top K is visible only
   for Gemini.
3. Restart the server (`Ctrl+C`, then `uvicorn app.main:app --reload` again)
   and reload the browser — confirm both conversations and their full
   message history are still there.

## Project layout

- `app/` - FastAPI backend, providers, persistence.
- `static/` - frontend (HTML/CSS/vanilla JS, no build step).
- `tests/` - pytest test suite.
```

- [ ] **Step 2: Run the full test suite**

```bash
pytest -v
```
Expected: PASS (every test file created in Tasks 1-8).

- [ ] **Step 3: Structural server check without requiring real API keys**

```bash
ASSISTANT_DB_PATH=/tmp/assistant_smoke_test.db uvicorn app.main:app --port 8124 &
SERVER_PID=$!
sleep 1
curl -s http://127.0.0.1:8124/ | grep -q "My Personal Assistant" && echo "OK: root page served"
CONV_ID=$(curl -s -X POST http://127.0.0.1:8124/api/conversations -H "Content-Type: application/json" -d '{}' | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
curl -s -X POST "http://127.0.0.1:8124/api/conversations/$CONV_ID/messages" -H "Content-Type: application/json" -d '{"content":"hi"}' | grep -q "event: error" && echo "OK: graceful missing-key error surfaced"
kill $SERVER_PID
rm -f /tmp/assistant_smoke_test.db
```
Expected output includes both `OK: root page served` and `OK: graceful missing-key error surfaced` (the second confirms the app fails gracefully, not with a crash, when `OPENAI_API_KEY` isn't set in this shell).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Add README with setup instructions and manual smoke test"
```

- [ ] **Step 5: Hand off to the user for the real-key smoke test**

Tell the user: the app is built and all automated tests pass; ask them to
follow the "Manual smoke test" section of `README.md` with their real API
keys in `.env` to confirm end-to-end behavior with actual OpenAI and Gemini
responses.
