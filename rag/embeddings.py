"""Port for turning policy text and questions into embedding vectors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    """Embedding provider used by ingest and retrieval.

    A second provider implements these two methods and returns plain float vectors.
    """

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one complete embedding vector for each input text."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Return the embedding vector for a single question."""
        ...
