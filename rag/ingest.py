"""Chunk a policy file, embed each section, and store it in Chroma."""

from __future__ import annotations

from pathlib import Path

from adapter.chroma_store import ChromaPolicyStore
from adapter.ollama_embeddings import OllamaEmbeddingAdapter
from config import CHROMA_PATH, POLICY_PATH
from rag.chunking import chunk_policy_file
from rag.embeddings import Embedder
from rag.store import PolicyStore


def ingest_policy(
    policy_path: str | Path = POLICY_PATH,
    *,
    chroma_path: str | Path | None = None,
    store: PolicyStore | None = None,
    embedder: Embedder | None = None,
) -> list[str]:
    """Chunk, embed, and store a policy file, returning the written chunk ids."""
    chunks = chunk_policy_file(policy_path)
    embedder = embedder or OllamaEmbeddingAdapter()
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
    store = store or ChromaPolicyStore(chroma_path or CHROMA_PATH)
    return store.upsert_chunks(chunks, embeddings)
