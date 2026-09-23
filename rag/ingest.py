"""Chunk a policy file, embed each section, and store it in Chroma."""

from __future__ import annotations

from pathlib import Path

from config import POLICY_PATH
from rag.chunking import chunk_policy_file
from rag.embeddings import Embedder
from rag.store import PolicyStore


def ingest_policy(
    policy_path: str | Path = POLICY_PATH,
    *,
    store: PolicyStore,
    embedder: Embedder,
) -> list[str]:
    """Chunk, embed, and store a policy file, returning the written chunk ids."""
    chunks = chunk_policy_file(policy_path)
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
    return store.upsert_chunks(chunks, embeddings)
