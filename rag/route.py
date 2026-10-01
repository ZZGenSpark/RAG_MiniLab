"""Choose vector or hybrid retrieval.

Jev makes that choice when ``TYPESAFE_API_KEY`` is set. Otherwise a section
code selects hybrid and every other question selects vector. The decision
has no score and no blend weight.
"""

from __future__ import annotations

import re
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from config import typesafe_api_key

Strategy = Literal["vector", "hybrid"]

# A section code is "Section 7", "Section 7.3", or a dotted rule such as "7.3".
# A version ("v2.0", "version 2.0") is not a section code. A bare integer is not either.
_SECTION_CODE = re.compile(r"(?i)\bsection\s+\d+(?:\.\d+)?\b|(?<!version )(?<!v)\b\d+\.\d+\b")


class RetrievalDecision(BaseModel):
    """The only routing choice: vector search, or vector search plus keywords."""

    model_config = ConfigDict(extra="forbid")

    strategy: Strategy = Field(description="vector searches by similarity. hybrid also searches by keywords.")


class Router(Protocol):
    """Something that chooses vector or hybrid for one question."""

    def choose(self, question: str) -> RetrievalDecision:
        """Return the retrieval strategy for this question."""
        ...


def question_names_section(question: str) -> bool:
    """Return whether the question names a policy section code."""
    return _SECTION_CODE.search(question) is not None


def fallback_decision(question: str) -> RetrievalDecision:
    """Select hybrid for a section code and vector for every other question."""
    if not question.strip():
        raise ValueError("question must not be empty")
    strategy: Strategy = "hybrid" if question_names_section(question) else "vector"
    return RetrievalDecision(strategy=strategy)


class FallbackRouter:
    """Use the section-code rule. This router does not call Jev."""

    def choose(self, question: str) -> RetrievalDecision:
        """Return the fallback strategy for this question."""
        return fallback_decision(question)


def default_router() -> Router:
    """Return Jev when the API key is set, and the section-code fallback otherwise."""
    if typesafe_api_key():
        from adapter.jev_router import JevRouter

        return JevRouter()
    return FallbackRouter()


def route(question: str, *, router: Router | None = None) -> RetrievalDecision:
    """Choose vector or hybrid. An injected router replaces Jev and the fallback."""
    if not question.strip():
        raise ValueError("question must not be empty")
    chosen = router if router is not None else default_router()
    return chosen.choose(question)
