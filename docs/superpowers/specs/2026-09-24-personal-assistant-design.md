# My Personal Assistant — Design Spec

Date: 2026-09-24

## 1. Purpose

A local web app, titled "My Personal Assistant", that lets the user chat with
either OpenAI or Google Gemini models using their own API keys, with a
Playground-style parameter panel (temperature, top_p, top_k, max output
tokens, seed, stop sequence) inspired by the attached Groq Console
screenshots. Runs entirely on the user's machine; API keys never leave the
backend process.

## 2. Scope (v1)

In scope:
- Local web app: Python FastAPI backend + static HTML/CSS/vanilla-JS frontend.
- Chat with OpenAI (`gpt-5.1`, `gpt-5.1-mini`) and Gemini (`gemini-3-pro`,
  `gemini-3-flash`) models, selectable per conversation.
- Editable per-conversation system prompt.
- Streaming responses (Server-Sent Events) with a Stream toggle in the UI
  (on by default) for visual parity with the screenshots.
- Parameter controls: temperature, top_p, top_k (Gemini only — hidden/disabled
  for OpenAI), max output tokens, seed, stop sequence.
- Conversation persistence to local SQLite: list, create, rename (via first
  message), delete, resume past conversations.
- API keys read from a local `.env` file (`OPENAI_API_KEY`, `GEMINI_API_KEY`),
  never sent to or requested from the browser, never logged, never committed
  (`.env` is gitignored).

Out of scope (explicitly deferred):
- Function/tool calling, JSON mode, MCP servers, file/image attachments,
  multi-user auth, deployment beyond localhost.
- Dynamic model list fetched from provider APIs (v1 uses a curated static
  list; can be revisited later).

## 3. Architecture

```
Browser (static/index.html, app.js, style.css)
   |  fetch / EventSource
   v
FastAPI app (app/main.py)
   |-- app/db.py        SQLite: conversations, messages
   |-- app/models.py    Pydantic request/response schemas
   |-- app/providers/
         base.py               ChatProvider interface
         openai_provider.py    OpenAI SDK adapter
         gemini_provider.py    google-genai SDK adapter
   |
   v
OpenAI API / Gemini API (outbound only, using keys from .env)
```

The backend is the only component that imports the OpenAI/Gemini SDKs or
reads the API keys. The frontend never sees a key.

### 3.1 Provider interface

```python
# app/providers/base.py
class ChatParams(BaseModel):
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None          # Gemini only; ignored by OpenAI provider
    max_output_tokens: int = 1024
    seed: int | None = None
    stop_sequence: str | None = None  # single string; split on comma for multiple

class ChatProvider(Protocol):
    def stream(
        self,
        messages: list[dict],       # [{"role": "user"|"assistant", "content": str}, ...]
        system_prompt: str,
        model: str,
        params: ChatParams,
    ) -> Iterator[str]: ...
```

`OpenAIProvider.stream` maps `ChatParams` to the OpenAI `chat.completions`
(or `responses`) API's `temperature`, `top_p`, `max_tokens`/`max_output_tokens`,
`seed`, `stop`, and silently drops `top_k` (not supported by OpenAI).

`GeminiProvider.stream` maps `ChatParams` to `google-genai`'s
`GenerationConfig` (`temperature`, `top_p`, `top_k`, `max_output_tokens`,
`seed`, `stop_sequences`).

Both raise a common `ProviderError(message: str)` on missing API key or
upstream API failure, which `main.py` catches and turns into an SSE `error`
event / JSON error response — never a raw traceback to the client.

### 3.2 Data model (SQLite)

```sql
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,           -- uuid4
    title TEXT NOT NULL,           -- derived from first user message, editable later
    system_prompt TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL,        -- 'openai' | 'gemini'
    model TEXT NOT NULL,
    params_json TEXT NOT NULL,     -- serialized ChatParams
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE messages (
    id TEXT PRIMARY KEY,           -- uuid4
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,            -- 'user' | 'assistant' | 'error'
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

`app/db.py` exposes plain functions (`create_conversation`, `list_conversations`,
`get_conversation`, `delete_conversation`, `add_message`, `list_messages`) —
no ORM, just `sqlite3` with parameterized queries. Kept dependency-free and
easy to unit test with an in-memory (`:memory:`) database.

### 3.3 API endpoints

- `GET /api/conversations` — list conversations (id, title, provider, model, updated_at), newest first.
- `POST /api/conversations` — create a new conversation (defaults: empty system prompt, `openai`/`gpt-5.1`, default params). Returns the new conversation.
- `GET /api/conversations/{id}` — full conversation incl. messages.
- `PATCH /api/conversations/{id}` — update system_prompt, provider, model, params.
- `DELETE /api/conversations/{id}` — delete a conversation and its messages.
- `POST /api/conversations/{id}/messages` — body: `{content: str}`. Persists the user message, then streams the assistant reply via SSE (`event: token` chunks, `event: done` at the end, `event: error` on failure), and persists the final assistant message (or an `error`-role message) when the stream ends.

### 3.4 Frontend

Single-page vanilla JS app, three-column layout matching the screenshots:
- **Left sidebar:** "+ New Chat" button, list of conversations (click to load, small delete icon per row).
- **Center:** collapsible SYSTEM prompt textarea at top, scrollable message thread below (user messages right-aligned, assistant left-aligned, errors styled distinctly), input textarea + Submit button at the bottom.
- **Right panel ("Parameters"):** Provider toggle (OpenAI/Gemini) → Model dropdown filtered to that provider; sliders+numeric inputs for Temperature, Top P, Top K (hidden when provider = OpenAI), Max Output Tokens, Seed, Stop Sequence text input; a Stream toggle (default on).

Changing provider/model/params calls `PATCH /api/conversations/{id}` and
takes effect on the next message in that conversation. Since the streaming
endpoint is a `POST` (it needs a body), `app.js` reads the SSE response via
`fetch` + a `ReadableStream` reader (not `EventSource`, which only supports
GET), parsing `event:`/`data:` lines itself and rendering tokens
incrementally into the DOM as they arrive.

## 4. Configuration & secrets

- `.env.example` checked in with `OPENAI_API_KEY=` and `GEMINI_API_KEY=`
  placeholders and a comment on where to get each key.
- `.env` is gitignored; the app loads it via `python-dotenv` at startup.
- If a key is missing when a provider is used, the API returns a clear error
  ("OPENAI_API_KEY not set — add it to your .env file and restart") rendered
  as an error bubble in that conversation; the app does not crash and other
  provider still works if its key is present.
- Keys are read once at process start into memory; never logged, never
  included in any response body, never written to SQLite.

## 5. Error handling

- Missing/invalid API key → friendly inline error (see above), HTTP 400.
- Upstream provider error (rate limit, bad request, network) → caught in the
  provider adapter, wrapped in `ProviderError`, surfaced as an `error`-role
  message in the thread; conversation remains usable afterward.
- Invalid request params (e.g. temperature out of range) → Pydantic
  validation error, HTTP 422, shown as a toast/inline notice in the UI
  before any provider call is made.
- Frontend network failure (server not running) → simple "Can't reach the
  assistant server" banner.

## 6. Testing

- `tests/test_db.py` — CRUD functions against an in-memory SQLite DB.
- `tests/test_providers.py` — given a `ChatParams` instance, assert each
  provider adapter builds the correct SDK call kwargs; provider SDK clients
  are mocked, no real network/API calls in tests.
- `tests/test_api.py` — FastAPI `TestClient` hitting `/api/conversations`
  CRUD endpoints against a temp SQLite file; chat-streaming endpoint tested
  with a mocked provider.
- Manual smoke test (documented in README): run the server, create a
  conversation, send a message on each provider, confirm streaming and
  persistence after restart.

## 7. Out-of-scope follow-ups (not v1)

- Dynamic model listing from provider APIs.
- Tool/function calling, JSON mode, file attachments.
- Multi-user support / auth.
- Packaging as a desktop app.

## 8. Parallelization notes for implementation

The following units have no interdependencies and can be built concurrently
by separate subagents once scaffolding (project structure, requirements.txt,
`.env.example`, `app/models.py`) exists:
- `app/db.py` + `tests/test_db.py`
- `app/providers/openai_provider.py` + `app/providers/gemini_provider.py` + `tests/test_providers.py`
- `static/` frontend (can be built against the documented API contract before the backend routes are wired up)

`app/main.py` (wiring routes to db + providers) and `tests/test_api.py`
depend on the above and come last.
