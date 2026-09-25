"""Rerank a shortlist of retrieved chunks. One scorer, then the top hits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from config import TOP_K
from rag.schema import RetrievedChunk


class Reranker(Protocol):
    """Scores question-chunk pairs. A higher score is a better match."""

    def score(self, question: str, chunks: Sequence[RetrievedChunk]) -> Sequence[float]:
        """Return one relevance score for each chunk, in the same order."""
        ...


@dataclass(frozen=True)
class RerankResult:
    """The kept chunks, plus one score for every chunk in the shortlist."""

    chunks: list[RetrievedChunk]
    scores: list[tuple[str, float]]


def rerank(
    question: str,
    chunks: Sequence[RetrievedChunk],
    *,
    reranker: Reranker,
    limit: int = TOP_K,
) -> RerankResult:
    """Order the shortlist by the reranker and keep the highest scores."""
    if not question.strip():
        raise ValueError("question must not be empty")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if not chunks:
        return RerankResult(chunks=[], scores=[])

    scores = list(reranker.score(question, chunks))
    if len(scores) != len(chunks):
        raise ValueError("reranker must return one score per chunk")
    order = sorted(range(len(chunks)), key=lambda index: (-float(scores[index]), chunks[index].chunk_id))
    return RerankResult(
        chunks=[chunks[index] for index in order[:limit]],
        scores=[(chunks[index].chunk_id, float(scores[index])) for index in order],
    )
