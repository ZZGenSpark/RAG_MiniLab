from __future__ import annotations

import json
import re
from typing import Protocol

import ollama

from rag.config import DEFAULT_CHAT_MODEL, DEFAULT_OLLAMA_HOST
from rag.schema import (
    REFUSAL_ANSWER,
    Citation,
    GroundedModelOutput,
    RetrievedChunk,
)

INSTRUCTION = """Does the policy excerpt below, by itself, answer the question?
Answer using only this excerpt. Do not use outside knowledge and do not
assume any other policy sections exist.

Return JSON with keys:
- answerable: true if this excerpt alone answers the question, false otherwise
- answer: the grounded answer when answerable is true, otherwise an empty string"""

REFUSAL_MARKERS = (
    "does not answer this question",
    "do not contain the answer",
    "does not contain the answer",
    "policy does not answer",
    "not answered by the provided policy",
)


class ChatClient(Protocol):
    def chat(self, model: str, messages: list, **kwargs): ...


class Generator:
    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        client: ChatClient | None = None,
    ) -> None:
        self.model = model or DEFAULT_CHAT_MODEL
        self.client = client or ollama.Client(host=host or DEFAULT_OLLAMA_HOST)

    def complete(self, prompt: str) -> str:
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


def build_prompt(question: str, chunk: RetrievedChunk) -> str:
    excerpt = f"[Section {chunk.citation_section}]\n{chunk.text}"
    return f"{INSTRUCTION}\n\nQuestion: {question}\n\nPolicy excerpt:\n{excerpt}"


def is_refusal(answer: str) -> bool:
    normalized = " ".join(answer.lower().split())
    return any(marker in normalized for marker in REFUSAL_MARKERS)


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    generator: Generator | None = None,
) -> tuple[str, Citation | None]:
    if not chunks:
        return REFUSAL_ANSWER, None

    generator = generator or Generator()

    # Chunks are already sorted by cosine distance ascending (closest first).
    # Try the closest chunk first; only fall back to a farther chunk if the
    # closer one truly cannot answer the question on its own.
    for chunk in chunks:
        raw = generator.complete(build_prompt(question, chunk))
        parsed = _parse_model_output(raw)

        if parsed.answerable and parsed.answer.strip():
            return parsed.answer.strip(), Citation(
                document=chunk.document,
                version=chunk.version,
                section=chunk.citation_section,
            )

    return REFUSAL_ANSWER, None


def _parse_model_output(raw: str) -> GroundedModelOutput:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        return GroundedModelOutput.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValueError):
        if is_refusal(raw):
            return GroundedModelOutput(answerable=False, answer="")
        return GroundedModelOutput(answerable=True, answer=raw.strip())
