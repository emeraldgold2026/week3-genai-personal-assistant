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
