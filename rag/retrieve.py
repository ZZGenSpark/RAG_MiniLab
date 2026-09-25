"""Retrieve policy chunks by vector search, or by vector search fused with BM25."""

from __future__ import annotations

from collections.abc import Sequence

from config import FUSED_CANDIDATES, HYBRID_CANDIDATES, RRF_K, VECTOR_CANDIDATES
from rag.embeddings import Embedder
from rag.lexical import bm25_top
from rag.route import RetrievalDecision
from rag.schema import PolicyChunk, RetrievedChunk
from rag.store import PolicyStore

_BM25_ONLY_DISTANCE = 1.0


def fuse_reciprocal_ranks(
    rankings: Sequence[Sequence[RetrievedChunk]],
    *,
    limit: int = FUSED_CANDIDATES,
) -> list[RetrievedChunk]:
    """Merge rankings with equal-weight reciprocal rank fusion.

    Each leg adds ``1 / (60 + rank)``. Rank starts at 1. Neither leg is weighted
    above the other.
    """
    scores: dict[str, float] = {}
    chosen: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            chosen.setdefault(chunk.chunk_id, chunk)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [chosen[chunk_id] for chunk_id in ordered[:limit]]


def retrieve(
    question: str,
    *,
    store: PolicyStore,
    embedder: Embedder,
    decision: RetrievalDecision | None = None,
    n_results: int = FUSED_CANDIDATES,
) -> list[RetrievedChunk]:
    """Return the vector shortlist, or the hybrid shortlist of 5.

    Vector search is one cosine query for 5 chunks. Hybrid search asks for 10
    vector chunks and 10 BM25 chunks, then keeps 5 by equal-weight RRF.
    Omitting the decision uses vector search. The caller passes a decision from
    the router when the question should be allowed to select hybrid.
    """
    if not question.strip():
        raise ValueError("question must not be empty")
    if n_results < 1:
        raise ValueError("n_results must be at least 1")

    chosen = decision or RetrievalDecision(strategy="vector")
    limit = min(n_results, FUSED_CANDIDATES)
    if chosen.strategy == "vector":
        return _vector_hits(question, store=store, embedder=embedder, n_results=VECTOR_CANDIDATES)[:limit]
    return _hybrid_hits(question, store=store, embedder=embedder)[:limit]


def _vector_hits(
    question: str,
    *,
    store: PolicyStore,
    embedder: Embedder,
    n_results: int,
) -> list[RetrievedChunk]:
    """Embed the question and return the nearest stored chunks."""
    return store.query_similar(embedder.embed_query(question), n_results=n_results)


def _hybrid_hits(
    question: str,
    *,
    store: PolicyStore,
    embedder: Embedder,
) -> list[RetrievedChunk]:
    """Fuse the vector top 10 with the BM25 top 10 and keep 5."""
    vector_hits = _vector_hits(question, store=store, embedder=embedder, n_results=HYBRID_CANDIDATES)
    by_id = {hit.chunk_id: hit for hit in vector_hits}
    keyword_hits = [
        _retrieved(chunk, by_id[chunk.chunk_id].distance if chunk.chunk_id in by_id else _BM25_ONLY_DISTANCE)
        for chunk in bm25_top(question, store.get_all(), HYBRID_CANDIDATES)
    ]
    return fuse_reciprocal_ranks([vector_hits, keyword_hits])


def _retrieved(chunk: PolicyChunk, distance: float) -> RetrievedChunk:
    """Copy a stored chunk into a retrieval hit."""
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        document=chunk.document,
        version=chunk.version,
        section=chunk.section,
        section_title=chunk.section_title,
        parent_heading=chunk.parent_heading,
        text=chunk.text,
        distance=distance,
    )
