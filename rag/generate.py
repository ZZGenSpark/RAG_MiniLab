"""Ground an answer in one retrieved policy excerpt.

Asks the chat model whether a single chunk answers the question, then cites that chunk or refuses.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

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


class Generator(Protocol):
    """Chat provider that turns one grounded prompt into model text.

    A second provider implements complete and returns the raw model string.
    """

    def complete(self, prompt: str) -> str:
        """Send one prompt and return the model's text."""
        ...


def build_prompt(question: str, chunk: RetrievedChunk) -> str:
    """Build the single-excerpt prompt for one question and chunk."""
    excerpt = f"[Section {chunk.citation_section}]\n{chunk.text}"
    return f"{INSTRUCTION}\n\nQuestion: {question}\n\nPolicy excerpt:\n{excerpt}"


def is_refusal(answer: str) -> bool:
    """Return whether the text matches a known refusal phrase."""
    normalized = " ".join(answer.lower().split())
    return any(marker in normalized for marker in REFUSAL_MARKERS)


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    generator: Generator | None = None,
) -> tuple[str, Citation | None]:
    """Answer from the closest usable chunk, or refuse when none can."""
    if not chunks:
        return REFUSAL_ANSWER, None

    if generator is None:
        from adapter.ollama_chat import OllamaChatAdapter

        generator = OllamaChatAdapter()

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
    """Parse model JSON, treating known refusal text as unanswerable."""
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
