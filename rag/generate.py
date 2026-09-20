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

INSTRUCTION = """Answer the question using only the policy excerpts below.
Include the section that supports your answer.
If the excerpts do not contain the answer, respond:
"The provided policy does not answer this question."

Return JSON with keys:
- answer: the grounded answer, or the refusal sentence
- section: the supporting section label such as "1. Meals", or null when refusing
Do not add information that is not in the excerpts."""

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
        )
        content = getattr(response, "message", None)
        text = getattr(content, "content", None) if content is not None else None
        if not text and isinstance(response, dict):
            text = response.get("message", {}).get("content")
        if not text:
            raise ValueError("generation model returned an empty response")
        return text


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    excerpts = []
    for chunk in chunks:
        excerpts.append(f"[Section {chunk.citation_section}]\n{chunk.text}")
    excerpt_block = "\n\n".join(excerpts) if excerpts else "[No policy excerpts retrieved]"
    return f"{INSTRUCTION}\n\nQuestion: {question}\n\nPolicy excerpts:\n{excerpt_block}"


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
    raw = generator.complete(build_prompt(question, chunks))
    parsed = _parse_model_output(raw)

    if is_refusal(parsed.answer):
        return REFUSAL_ANSWER, None

    supporting = _matching_chunk(parsed.section, chunks)
    if supporting is None:
        return REFUSAL_ANSWER, None

    return parsed.answer.strip(), Citation(
        document=supporting.document,
        version=supporting.version,
        section=supporting.citation_section,
    )


def _parse_model_output(raw: str) -> GroundedModelOutput:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        return GroundedModelOutput.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValueError):
        if is_refusal(raw):
            return GroundedModelOutput(answer=REFUSAL_ANSWER, section=None)
        return GroundedModelOutput(answer=raw.strip(), section=_section_in_text(raw))


def _section_in_text(text: str) -> str | None:
    match = re.search(r"\b(\d+)\.\s+([A-Z][A-Za-z ]+)", text)
    if match is None:
        return None
    return f"{match.group(1)}. {match.group(2).strip()}"


def _matching_chunk(section: str | None, chunks: list[RetrievedChunk]) -> RetrievedChunk | None:
    if not section:
        return None
    normalized = " ".join(section.lower().split())
    for chunk in chunks:
        labels = {
            chunk.citation_section.lower(),
            chunk.section,
            chunk.section_title.lower(),
            f"{chunk.section}. {chunk.section_title}".lower(),
        }
        if normalized in labels or normalized.rstrip(".") in labels:
            return chunk
    return None
