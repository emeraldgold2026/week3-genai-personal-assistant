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
