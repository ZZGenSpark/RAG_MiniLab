"""Ground an answer in the final retrieved policy excerpts.

One prompt shows every chunk. The answer cites each excerpt the model used.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from itertools import combinations
from typing import Protocol

from prompt.grounded_excerpt import build_grounded_excerpt_prompt
from rag.schema import (
    REFUSAL_ANSWER,
    Citation,
    GroundedModelOutput,
    PolicyChunk,
    RetrievedChunk,
)

_AMOUNT_RE = re.compile(r"\$\d+(?:,\d{3})*(?:\.\d+)?")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_HEADING_SUFFIX_RE = re.compile(r"\s+[—–-]\s+.*$")
_HEADING_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?\.?\s+")
_MEASURE_NUMBER = (
    r"\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?"
    r"|forty-five|fifteen|thirty|twelve|eight|seven|three|four|five|nine|one|two|six|ten|forty"
)
_MEASURE_RE = re.compile(
    rf"(?P<number>{_MEASURE_NUMBER})(?:\s+\([^)]*\))?(?:-|\s+)"
    r"(?P<unit>minutes?|hours?|days?|weeks?|tokens?|sessions?|mg|g)\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-z0-9]+")
_WORD_AMOUNTS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "twelve": 12,
    "fifteen": 15,
    "thirty": 30,
    "forty": 40,
    "forty-five": 45,
}
_UNIT_SCALE = {
    "minute": ("duration", 1),
    "hour": ("duration", 60),
    "day": ("duration", 1440),
    "week": ("duration", 10080),
    "token": ("tokens", 1),
    "session": ("sessions", 1),
    "mg": ("mg", 1),
    "g": ("g", 1),
}
_COPY_OVERLAP = 0.75
_STOPWORDS = frozenset(
    """
    a an the of to and or for in on at by with from that this it is are be as not no
    any their its per into over than then they them these those
    """.split()
)


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
    """Answer from the final chunks and cite every source the model used.

    The model is asked to refuse when copies of one document disagree. A
    one-sided answer is still a refusal when those copies state different
    amounts. Agreeing copies are answered from the excerpts.
    """
    if not chunks:
        return REFUSAL_ANSWER, []

    parsed = _parse_model_output(generator.complete(build_prompt(question, chunks)))
    if excerpts_conflict(chunks):
        return REFUSAL_ANSWER, []
    answer = parsed.answer.strip()
    used = _chunks_for_sources(parsed.sources, chunks)
    if not parsed.answerable or not answer or not used:
        return REFUSAL_ANSWER, []
    if not _answer_claims_are_supported(answer, used, question):
        return REFUSAL_ANSWER, []
    return answer, [_citation(chunk) for chunk in used]


def excerpts_conflict(chunks: Sequence[PolicyChunk]) -> bool:
    """Return whether two versions of one document disagree in these excerpts.

    The same rule at 20 minutes in both versions is not a conflict. A different
    amount, or a section that was rewritten, is.
    """
    grouped: dict[str, list[PolicyChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.document, []).append(chunk)
    return any(_document_conflicts(group) for group in grouped.values())


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


def _document_conflicts(chunks: Sequence[PolicyChunk]) -> bool:
    """Return whether any cross-version pair in one document disagrees."""
    if len({chunk.version for chunk in chunks}) < 2:
        return False
    return any(
        _claims_conflict(left.text, right.text)
        for left, right in combinations(chunks, 2)
        if left.version != right.version and _comparable(left, right)
    )


def _comparable(left: PolicyChunk, right: PolicyChunk) -> bool:
    """Return whether these chunks are copies of one rule or one parent section."""
    if left.section == right.section:
        return True
    same_title = _plain_heading(left.section_title) == _plain_heading(right.section_title)
    same_topic = _topic(left) == _topic(right)
    if same_title and same_topic:
        return True
    if same_topic and _measures(left.text) and _measures(right.text):
        return True
    return _topic(left) == _heading_key(right) or _topic(right) == _heading_key(left)


def _claims_conflict(left: str, right: str) -> bool:
    """Return whether two copies state different amounts, or were rewritten."""
    left_measures = _measures(left)
    right_measures = _measures(right)
    if left_measures and right_measures:
        return left_measures != right_measures
    if left_measures or right_measures:
        return False
    return not _near_copy(left, right)


def _measures(text: str) -> frozenset[tuple[str, int]]:
    """Return normalized amounts such as duration in minutes or token counts."""
    found: set[tuple[str, int]] = set()
    for match in _MEASURE_RE.finditer(text):
        family, scale = _UNIT_SCALE[match.group("unit").lower().rstrip("s")]
        found.add((family, round(_amount(match.group("number")) * scale)))
    return frozenset(found)


def _amount(raw: str) -> float:
    """Read 500,000, 20, or two as a number."""
    text = raw.lower().replace(",", "")
    if text in _WORD_AMOUNTS:
        return float(_WORD_AMOUNTS[text])
    return float(text)


def _near_copy(left: str, right: str) -> bool:
    """Return whether the shorter excerpt mostly repeats the longer one."""
    left_words = _content_words(left)
    right_words = _content_words(right)
    if not left_words or not right_words:
        return False
    shared = len(left_words & right_words)
    return shared / min(len(left_words), len(right_words)) >= _COPY_OVERLAP


def _content_words(text: str) -> set[str]:
    """Return the content words used to compare a rewritten section."""
    return {word for word in _WORD_RE.findall(text.lower()) if len(word) > 2 and word not in _STOPWORDS}


def _topic(chunk: PolicyChunk) -> str:
    """Return the parent section name, or this chunk's own heading when it is the parent."""
    if chunk.parent_heading:
        return _plain_heading(chunk.parent_heading)
    return _heading_key(chunk)


def _heading_key(chunk: PolicyChunk) -> str:
    """Return this chunk's heading without its number or version note."""
    return _plain_heading(f"{chunk.section}. {chunk.section_title}")


def _plain_heading(heading: str) -> str:
    """Drop a version note and a leading section number."""
    without_note = _HEADING_SUFFIX_RE.sub("", heading).strip().lower()
    return _HEADING_NUMBER_RE.sub("", without_note)
