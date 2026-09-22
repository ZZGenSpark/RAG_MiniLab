"""Chroma implementation of the policy vector store."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import chromadb
from chromadb.api.models.Collection import Collection

from config import CHROMA_PATH
from rag.schema import (
    COLLECTION_METADATA,
    COLLECTION_NAME,
    DISTANCE_SPACE,
    EmbeddedChunk,
    PolicyChunk,
    RetrievedChunk,
    to_chroma_records,
)


class ChromaPolicyStore:
    """Persist policy chunks and query them by cosine distance in Chroma."""

    def __init__(self, path: str | Path | None = None) -> None:
        """Open or create the persistent expense-policy collection."""
        self.path = Path(path or CHROMA_PATH)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection: Collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata=COLLECTION_METADATA,
            embedding_function=None,
        )

    @property
    def distance_space(self) -> str:
        """Return the collection distance metric, defaulting to cosine."""
        metadata = self.collection.metadata or {}
        if metadata.get("hnsw:space"):
            return str(metadata["hnsw:space"])

        configuration = self.collection.configuration_json or {}
        hnsw = configuration.get("hnsw") or {}
        return str(hnsw.get("space") or DISTANCE_SPACE)

    def upsert_chunks(
        self,
        chunks: Sequence[PolicyChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> list[str]:
        """Store chunk text, embeddings, and metadata, keyed by chunk id."""
        records = to_chroma_records(chunks, embeddings)
        self.collection.upsert(**records.as_upsert())
        return records.ids

    def count(self) -> int:
        """Return how many chunks are currently stored."""
        return self.collection.count()

    def query_similar(
        self,
        embedding: Sequence[float],
        n_results: int = 3,
    ) -> list[RetrievedChunk]:
        """Return the chunks nearest to the query embedding."""
        available = self.count()
        if available == 0:
            return []

        result = self.collection.query(
            query_embeddings=[list(embedding)],
            n_results=min(n_results, 3, available),
            include=["documents", "metadatas", "distances"],
        )
        ids = _first_query_row(result["ids"])
        documents = _first_query_row(result["documents"])
        metadatas = _first_query_row(result["metadatas"])
        distances = _first_query_row(result["distances"])

        hits: list[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            if text is None or metadata is None or distance is None:
                raise ValueError(f"retrieved record {chunk_id} is missing text, metadata, or distance")
            hits.append(
                RetrievedChunk.model_validate(
                    {
                        "chunk_id": chunk_id,
                        "text": text,
                        **dict(metadata),
                        "distance": float(distance),
                    }
                )
            )
        hits.sort(key=lambda hit: hit.distance)
        return hits

    def get_all(self) -> list[EmbeddedChunk]:
        """Return every stored chunk, sorted by section number."""
        result = self.collection.get(include=["documents", "metadatas", "embeddings"])
        records: list[EmbeddedChunk] = []
        ids = list(result["ids"])
        documents = list(result["documents"] or [])
        metadatas = list(result["metadatas"] or [])
        embeddings = list(result["embeddings"] if result["embeddings"] is not None else [])

        for chunk_id, text, metadata, embedding in zip(
            ids, documents, metadatas, embeddings, strict=True
        ):
            if text is None or metadata is None or embedding is None:
                raise ValueError(f"stored record {chunk_id} is missing text, metadata, or embedding")
            records.append(
                EmbeddedChunk.from_stored(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=dict(metadata),
                    embedding=embedding,
                )
            )
        records.sort(key=lambda record: int(record.section))
        return records


def _first_query_row(value) -> list:
    """Return the first result row from a Chroma query field."""
    if value is None or len(value) == 0:
        return []
    row = value[0]
    return list(row) if row is not None else []
