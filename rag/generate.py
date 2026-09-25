"""Ground an answer in the final retrieved policy excerpts.

One prompt shows every chunk. The answer cites each excerpt the model used.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
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


def build_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    """Build one prompt that includes every final chunk."""
    excerpts = [
        "\n".join(
            [
                f"Excerpt {index}",
                f"Document: {chunk.document}",
                f"Version: {chunk.version}",
                f"Section: {chunk.citation_section}",
                chunk.text,
            ]
        )
        for index, chunk in enumerate(chunks, start=1)
    ]
    return build_grounded_excerpt_prompt(question, excerpts)


def generate_answer(
    question: str,
    chunks: Sequence[RetrievedChunk],
    *,
    generator: Generator,
) -> tuple[str, list[Citation]]:
    """Answer from the final chunks and cite every source the model used."""
    if not chunks:
        return REFUSAL_ANSWER, []

    parsed = _parse_model_output(generator.complete(build_prompt(question, chunks)))
    answer = parsed.answer.strip()
    used = _chunks_for_sources(parsed.sources, chunks)
    if not parsed.answerable or not answer or not used:
        return REFUSAL_ANSWER, []
    if not _answer_claims_are_supported(answer, used, question):
        return REFUSAL_ANSWER, []
    return answer, [_citation(chunk) for chunk in used]


def _chunks_for_sources(sources: Sequence[int], chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep each named excerpt once, in the order the model listed it."""
    used: list[RetrievedChunk] = []
    seen: set[int] = set()
    for source in sources:
        index = source - 1
        if index in seen or index < 0 or index >= len(chunks):
            continue
        seen.add(index)
        used.append(chunks[index])
    return used


def _citation(chunk: RetrievedChunk) -> Citation:
    """Cite one chunk the answer used."""
    return Citation(document=chunk.document, version=chunk.version, section=chunk.citation_section)


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


def _answer_claims_are_supported(answer: str, chunks: Sequence[RetrievedChunk], question: str) -> bool:
    """Return whether every amount and number in the answer is in a cited excerpt or the question."""
    haystack = "\n".join([question, *(chunk.text for chunk in chunks)]).lower()
    for amount in _AMOUNT_RE.findall(answer):
        if amount.lower() not in haystack:
            return False
    remainder = _AMOUNT_RE.sub(" ", answer)
    for number in _NUMBER_RE.findall(remainder):
        if number not in haystack:
            return False
    return True
