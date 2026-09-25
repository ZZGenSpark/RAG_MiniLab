"""Rerank a shortlist of retrieved chunks. One scorer, then the top hits."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from config import TOP_K
from rag.schema import RetrievedChunk


class Reranker(Protocol):
    """Scores question-chunk pairs. A higher score is a better match."""

    def score(self, question: str, chunks: Sequence[RetrievedChunk]) -> Sequence[float]:
        """Return one relevance score for each chunk, in the same order."""
        ...


def rerank(
    question: str,
    chunks: Sequence[RetrievedChunk],
    *,
    reranker: Reranker,
    limit: int = TOP_K,
) -> list[RetrievedChunk]:
    """Order the shortlist by the reranker and keep the highest scores."""
    if not question.strip():
        raise ValueError("question must not be empty")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if not chunks:
        return []

    scores = list(reranker.score(question, chunks))
    if len(scores) != len(chunks):
        raise ValueError("reranker must return one score per chunk")
    order = sorted(range(len(chunks)), key=lambda index: (-scores[index], chunks[index].chunk_id))
    return [chunks[index] for index in order[:limit]]
