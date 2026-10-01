"""Retrieve policy chunks by vector search, or by vector search fused with BM25."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from config import FUSED_CANDIDATES, HYBRID_CANDIDATES, RRF_K, TOP_K, VECTOR_CANDIDATES
from rag.audit import ChunkScore
from rag.embeddings import Embedder
from rag.lexical import bm25_top
from rag.rerank import Reranker, rerank
from rag.route import RetrievalDecision, Strategy
from rag.schema import PolicyChunk, RetrievedChunk
from rag.store import PolicyStore

_BM25_ONLY_DISTANCE = 1.0


@dataclass(frozen=True)
class RetrievalOutcome:
    """The chunks returned to the caller and the candidate lists behind them."""

    chunks: list[RetrievedChunk]
    strategy: Strategy
    vector_ids: list[str]
    bm25_ids: list[str]
    rrf_ids: list[str]
    cross_encoder_scores: list[ChunkScore]


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
    reranker: Reranker | None = None,
    n_results: int = TOP_K,
) -> list[RetrievedChunk]:
    """Rerank the shortlist of 5 and return the top chunks."""
    return retrieve_outcome(
        question,
        store=store,
        embedder=embedder,
        decision=decision,
        reranker=reranker,
        n_results=n_results,
    ).chunks


def retrieve_outcome(
    question: str,
    *,
    store: PolicyStore,
    embedder: Embedder,
    decision: RetrievalDecision | None = None,
    reranker: Reranker | None = None,
    n_results: int = TOP_K,
) -> RetrievalOutcome:
    """Rerank the shortlist and keep the candidate ids that produced it.

    Vector search is one cosine query for 5 chunks. Hybrid search asks for 10
    vector chunks and 10 BM25 chunks, then keeps 5 by equal-weight RRF. Both
    paths score that shortlist once. A vector decision leaves the keyword and
    RRF lists empty.
    """
    if not question.strip():
        raise ValueError("question must not be empty")
    if n_results < 1:
        raise ValueError("n_results must be at least 1")

    chosen = decision or RetrievalDecision(strategy="vector")
    if chosen.strategy == "vector":
        vector_hits = _vector_hits(question, store=store, embedder=embedder, n_results=VECTOR_CANDIDATES)
        shortlist = vector_hits
        bm25_ids: list[str] = []
        rrf_ids: list[str] = []
    else:
        vector_hits, keyword_hits, shortlist = _hybrid_lists(question, store=store, embedder=embedder)
        bm25_ids = [hit.chunk_id for hit in keyword_hits]
        rrf_ids = [hit.chunk_id for hit in shortlist]
    ranked = rerank(question, shortlist, reranker=reranker or _cross_encoder(), limit=TOP_K)
    return RetrievalOutcome(
        chunks=ranked.chunks[:n_results],
        strategy=chosen.strategy,
        vector_ids=[hit.chunk_id for hit in vector_hits],
        bm25_ids=bm25_ids,
        rrf_ids=rrf_ids,
        cross_encoder_scores=[ChunkScore(chunk_id=chunk_id, score=score) for chunk_id, score in ranked.scores],
    )


def _cross_encoder() -> Reranker:
    """Return the local cross-encoder. It is the only reranker."""
    from adapter.cross_encoder import CrossEncoderReranker

    return CrossEncoderReranker()


def _vector_hits(
    question: str,
    *,
    store: PolicyStore,
    embedder: Embedder,
    n_results: int,
) -> list[RetrievedChunk]:
    """Embed the question and return the nearest stored chunks."""
    return store.query_similar(embedder.embed_query(question), n_results=n_results)


def _hybrid_lists(
    question: str,
    *,
    store: PolicyStore,
    embedder: Embedder,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk], list[RetrievedChunk]]:
    """Return the vector top 10, the BM25 top 10, and the RRF shortlist of 5."""
    vector_hits = _vector_hits(question, store=store, embedder=embedder, n_results=HYBRID_CANDIDATES)
    by_id = {hit.chunk_id: hit for hit in vector_hits}
    keyword_hits = [
        _retrieved(chunk, by_id[chunk.chunk_id].distance if chunk.chunk_id in by_id else _BM25_ONLY_DISTANCE)
        for chunk in bm25_top(question, store.get_all(), HYBRID_CANDIDATES)
    ]
    return vector_hits, keyword_hits, fuse_reciprocal_ranks([vector_hits, keyword_hits])


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
