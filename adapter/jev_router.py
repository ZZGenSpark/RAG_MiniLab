"""TypeSafe Jev implementation of the retrieval router.

Jev answers one choice: vector or hybrid. It does not score the question
and it does not assign a keyword weight.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, cast

from pydantic import ValidationError
from typesafe_sdk import Choice, TypeSafeClient

from config import typesafe_api_key
from rag.route import RetrievalDecision

_STRATEGY = Choice(
    instructions=(
        "Choose hybrid when the question names a policy section code such as Section 7.3. "
        "Choose vector for every other question."
    ),
    criteria={
        "vector": "Search stored policy text by semantic similarity.",
        "hybrid": "Search by semantic similarity and by keywords, including a section code.",
    },
)


class ChoiceAnswer(Protocol):
    """The selected label from one TypeSafe choice."""

    choice: str


class SystemOneResponse(Protocol):
    """The choices returned by one System One call."""

    choices: Mapping[str, ChoiceAnswer]


class SystemOneClient(Protocol):
    """The TypeSafe call this router uses."""

    def system_one(
        self,
        state: str | Mapping[str, str],
        questions: Mapping[str, object],
    ) -> SystemOneResponse:
        """Answer the named questions about the given state."""
        ...


class JevRouter:
    """Ask Jev to choose vector or hybrid retrieval."""

    def __init__(self, client: SystemOneClient | None = None) -> None:
        """Store an optional client. The live client is opened on the first choice."""
        self._client = client

    def choose(self, question: str) -> RetrievalDecision:
        """Fill a retrieval decision from Jev's strategy choice."""
        if not question.strip():
            raise ValueError("question must not be empty")
        response = self._client_or_live().system_one(
            {"question": question},
            {"strategy": _STRATEGY},
        )
        try:
            return RetrievalDecision.model_validate({"strategy": response.choices["strategy"].choice})
        except ValidationError as exc:
            raise ValueError("retrieval strategy must be vector or hybrid") from exc

    def _client_or_live(self) -> SystemOneClient:
        """Return the injected client, or open a TypeSafe client with the configured key."""
        if self._client is None:
            self._client = cast(SystemOneClient, TypeSafeClient(api_key=typesafe_api_key()))
        return self._client
