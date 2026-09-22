"""Port for storing policy chunks and querying them by distance."""

from __future__ import annotations

from typing import Protocol, Sequence

from rag.schema import EmbeddedChunk, PolicyChunk, RetrievedChunk


class PolicyStore(Protocol):
    """Vector store used by ingest and retrieval.

    A second database implements these methods. Chunk text, vectors, and citation
    metadata stay in the same shapes either way.
    """

    def upsert_chunks(
        self,
        chunks: Sequence[PolicyChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> list[str]:
        """Store chunk text, embeddings, and metadata, keyed by chunk id."""
        ...

    def count(self) -> int:
        """Return how many chunks are currently stored."""
        ...

    def query_similar(
        self,
        embedding: Sequence[float],
        n_results: int = 3,
    ) -> list[RetrievedChunk]:
        """Return the chunks nearest to the query embedding."""
        ...

    def get_all(self) -> list[EmbeddedChunk]:
        """Return every stored chunk, sorted by section number."""
        ...
