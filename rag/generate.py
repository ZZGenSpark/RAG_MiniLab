"""Ground an answer in one retrieved policy excerpt.

Asks the chat model whether a single chunk answers the question, then cites that chunk or refuses.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from prompt.grounded_excerpt import build_grounded_excerpt_prompt
from rag.schema import (
    REFUSAL_ANSWER,
    Citation,
    GroundedModelOutput,
    RetrievedChunk,
)

_AMOUNT_RE = re.compile(r"\$\d+(?:,\d{3})*(?:\.\d+)?")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


class Generator(Protocol):
    """Chat provider that turns one grounded prompt into model text.

    A second provider implements complete and returns the raw model string.
    """

    def complete(self, prompt: str) -> str:
        """Send one prompt and return the model's text."""
        ...


def build_prompt(question: str, chunk: RetrievedChunk) -> str:
    """Build the single-excerpt prompt for one question and chunk."""
    return build_grounded_excerpt_prompt(question, chunk.citation_section, chunk.text)


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    generator: Generator,
) -> tuple[str, Citation | None]:
    """Answer from the closest usable chunk, or refuse when none can."""
    if not chunks:
        return REFUSAL_ANSWER, None

    # Chunks are already sorted by cosine distance ascending (closest first).
    # Try the closest chunk first; only fall back to a farther chunk if the
    # closer one truly cannot answer the question on its own.
    for chunk in chunks:
        raw = generator.complete(build_prompt(question, chunk))
        parsed = _parse_model_output(raw)
        answer = parsed.answer.strip()
        if parsed.answerable and answer and _answer_claims_are_supported(answer, chunk.text, question):
            return answer, Citation(
                document=chunk.document,
                version=chunk.version,
                section=chunk.citation_section,
            )

    return REFUSAL_ANSWER, None


def _parse_model_output(raw: str) -> GroundedModelOutput:
    """Parse model JSON. Unparseable text does not answer the question."""
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        return GroundedModelOutput.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValueError):
        return GroundedModelOutput(answerable=False, answer="")


def _answer_claims_are_supported(answer: str, excerpt: str, question: str) -> bool:
    """Return whether every amount and number in the answer is in the excerpt or question."""
    haystack = f"{excerpt}\n{question}".lower()
    for amount in _AMOUNT_RE.findall(answer):
        if amount.lower() not in haystack:
            return False
    remainder = _AMOUNT_RE.sub(" ", answer)
    for number in _NUMBER_RE.findall(remainder):
        if number not in haystack:
            return False
    return True
