"""Ollama implementation of the grounded-answer chat port."""

from __future__ import annotations

from typing import Protocol

import ollama

from config import CHAT_MODEL, OLLAMA_HOST
from rag.schema import GroundedModelOutput


class ChatClient(Protocol):
    def chat(self, model: str, messages: list, **kwargs):
        """Send a chat completion and return the model response."""
        ...


class OllamaChatAdapter:
    """Complete one prompt with an Ollama chat model and return the text."""

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        client: ChatClient | None = None,
    ) -> None:
        """Configure the chat model, host, and client."""
        self.model = model or CHAT_MODEL
        self.client = client or ollama.Client(host=host or OLLAMA_HOST)

    def complete(self, prompt: str) -> str:
        """Send one prompt and return the model's text."""
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format=GroundedModelOutput.model_json_schema(),
            think=False,
            options={"temperature": 0},
        )
        content = getattr(response, "message", None)
        text = getattr(content, "content", None) if content is not None else None
        if not text and isinstance(response, dict):
            text = response.get("message", {}).get("content")
        if not text:
            raise ValueError("generation model returned an empty response")
        return text
