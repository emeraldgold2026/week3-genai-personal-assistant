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
