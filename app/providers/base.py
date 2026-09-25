"""Shared provider interface and parameters for chat model calls."""
from typing import Iterator, Optional, Protocol

from pydantic import BaseModel, field_validator


class ChatParams(BaseModel):
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: Optional[int] = None          # Gemini only; ignored by the OpenAI provider
    max_output_tokens: int = 1024
    seed: Optional[int] = None
    stop_sequence: Optional[str] = None  # comma-separated for multiple stop strings

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
    def _check_top_k(cls, v: Optional[int]) -> Optional[int]:
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
