"""Embed a question and fetch the closest policy chunks."""

from __future__ import annotations

from config import TOP_K
from rag.embeddings import Embedder
from rag.schema import RetrievedChunk
from rag.store import PolicyStore


def retrieve(
    question: str,
    *,
    store: PolicyStore,
    embedder: Embedder,
    n_results: int = TOP_K,
) -> list[RetrievedChunk]:
    """Embed the question and return the nearest stored policy chunks."""
    if not question.strip():
        raise ValueError("question must not be empty")

    query_embedding = embedder.embed_query(question)
    return store.query_similar(query_embedding, n_results=min(n_results, TOP_K))
