"""Shared paths and doubles for tests that should not load a hosted or local model."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rag.schema import RetrievedChunk

EXPENSE_POLICY_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "expense-policy.md"


class KeepingReranker:
    """Keep the shortlist order so a test does not load the cross-encoder."""

    def __init__(self) -> None:
        """Start with no scored shortlists."""
        self.shortlists: list[list[RetrievedChunk]] = []

    def score(self, question: str, chunks: Sequence[RetrievedChunk]) -> list[float]:
        """Give the current first chunk the highest score."""
        self.shortlists.append(list(chunks))
        return [float(len(chunks) - index) for index in range(len(chunks))]
