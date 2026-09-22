from __future__ import annotations

from rag.config import DEFAULT_CHROMA_PATH
from rag.embeddings import Embedder
from rag.schema import RetrievedChunk
from rag.store import PolicyStore

TOP_K = 3


def retrieve(
    question: str,
    *,
    store: PolicyStore | None = None,
    embedder: Embedder | None = None,
    n_results: int = TOP_K,
) -> list[RetrievedChunk]:
    if not question.strip():
        raise ValueError("question must not be empty")

    embedder = embedder or Embedder()
    store = store or PolicyStore(DEFAULT_CHROMA_PATH)
    query_embedding = embedder.embed_query(question)
    return store.query_similar(query_embedding, n_results=min(n_results, TOP_K))
