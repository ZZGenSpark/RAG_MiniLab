from __future__ import annotations

from pathlib import Path

from rag.chunking import chunk_policy_file
from rag.config import DEFAULT_CHROMA_PATH, DEFAULT_POLICY_PATH
from rag.embeddings import Embedder
from rag.store import PolicyStore


def ingest_policy(
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    *,
    chroma_path: str | Path | None = None,
    store: PolicyStore | None = None,
    embedder: Embedder | None = None,
) -> list[str]:
    chunks = chunk_policy_file(policy_path)
    embedder = embedder or Embedder()
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
    store = store or PolicyStore(chroma_path or DEFAULT_CHROMA_PATH)
    return store.upsert_chunks(chunks, embeddings)
