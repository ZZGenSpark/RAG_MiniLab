"""Chunk policy markdown, embed each section, and store it in Chroma."""

from __future__ import annotations

from pathlib import Path

from rag.chunking import chunk_policy_file
from rag.embeddings import Embedder
from rag.schema import PolicyChunk
from rag.store import PolicyStore


def ingest_policy(
    policy_path: str | Path,
    *,
    store: PolicyStore,
    embedder: Embedder,
) -> list[str]:
    """Chunk, embed, and store policy markdown, returning the written chunk ids.

    A directory ingests every markdown file in one batch. A file ingests that file.
    """
    chunks = _chunks_from(Path(policy_path))
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
    return store.upsert_chunks(chunks, embeddings)


def _chunks_from(path: Path) -> list[PolicyChunk]:
    """Load chunks from one markdown file or from every markdown file in a directory."""
    files = _markdown_files(path)
    chunks = [chunk for policy_file in files for chunk in chunk_policy_file(policy_file)]
    ids = [chunk.chunk_id for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("chunk ids must be unique across the ingested policies")
    return chunks


def _markdown_files(path: Path) -> list[Path]:
    """Return the markdown files a path contributes to ingest."""
    if path.is_dir():
        files = sorted(path.glob("*.md"))
        if not files:
            raise ValueError(f"{path} contains no markdown policies")
        return files
    return [path]
