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
            "max_completion_tokens": params.max_output_tokens,
            "stream": True,
        }
        if params.seed is not None:
            kwargs["seed"] = params.seed
        if params.stop_sequence:
            kwargs["stop"] = [s.strip() for s in params.stop_sequence.split(",") if s.strip()]
        return kwargs
