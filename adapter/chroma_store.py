"""Chroma implementation of the policy vector store."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

from config import CHROMA_PATH, COLLECTION_NAME, TOP_K
from rag.schema import (
    COLLECTION_METADATA,
    DISTANCE_SPACE,
    EmbeddedChunk,
    PolicyChunk,
    RetrievedChunk,
    require_uniform_embedding_width,
    to_chroma_records,
)


class ChromaPolicyStore:
    """Persist policy chunks and query them by cosine distance in Chroma."""

    def __init__(self, path: str | Path | None = None) -> None:
        """Open or create the persistent company-policies collection."""
        self.path = Path(path or CHROMA_PATH)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection: Collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata=COLLECTION_METADATA,
            embedding_function=None,
        )
        if self.distance_space != DISTANCE_SPACE:
            raise ValueError(
                f"collection {COLLECTION_NAME} uses {self.distance_space} distance; expected {DISTANCE_SPACE}"
            )

    @property
    def distance_space(self) -> str:
        """Return the collection distance metric."""
        metadata = self.collection.metadata or {}
        if metadata.get("hnsw:space"):
            return str(metadata["hnsw:space"])

        configuration = self.collection.configuration_json or {}
        hnsw = configuration.get("hnsw") or {}
        space = hnsw.get("space")
        if not space:
            raise ValueError(f"collection {COLLECTION_NAME} has no hnsw distance space")
        return str(space)

    def upsert_chunks(
        self,
        chunks: Sequence[PolicyChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> list[str]:
        """Store chunk text, embeddings, and metadata, and drop ids not in this batch."""
        records = to_chroma_records(chunks, embeddings)
        width = require_uniform_embedding_width(records.embeddings)
        stored_width = self._stored_embedding_width()
        if stored_width is not None and width is not None and stored_width != width:
            raise ValueError(f"embedding width {width} does not match stored width {stored_width}")
        self.collection.upsert(**records.as_upsert())
        existing_ids = list(self.collection.get()["ids"])
        stale_ids = [chunk_id for chunk_id in existing_ids if chunk_id not in set(records.ids)]
        if stale_ids:
            self.collection.delete(ids=stale_ids)
        return records.ids

    def count(self) -> int:
        """Return how many chunks are currently stored."""
        return self.collection.count()

    def query_similar(
        self,
        embedding: Sequence[float],
        n_results: int = TOP_K,
    ) -> list[RetrievedChunk]:
        """Return the chunks nearest to the query embedding."""
        if n_results < 1:
            raise ValueError("n_results must be at least 1")

        available = self.count()
        if available == 0:
            return []

        query = [float(value) for value in embedding]
        if not query:
            raise ValueError("query embedding must be a complete vector")
        stored_width = self._stored_embedding_width()
        if stored_width is not None and len(query) != stored_width:
            raise ValueError(f"query embedding width {len(query)} does not match stored width {stored_width}")

        query_vectors: list[Sequence[float]] = [query]
        result = self.collection.query(
            query_embeddings=query_vectors,
            n_results=min(n_results, available),
            include=["documents", "metadatas", "distances"],
        )
        ids = _first_query_row(result["ids"])
        documents = _first_query_row(result["documents"])
        metadatas = _first_query_row(result["metadatas"])
        distances = _first_query_row(result["distances"])

        hits: list[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances, strict=True):
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

        for chunk_id, text, metadata, embedding in zip(ids, documents, metadatas, embeddings, strict=True):
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
        records.sort(key=lambda record: _section_order(record.section))
        return records

    def _stored_embedding_width(self) -> int | None:
        """Return the width of one stored vector, or None when the collection is empty."""
        if self.count() == 0:
            return None
        result = self.collection.get(limit=1, include=["embeddings"])
        embeddings = result.get("embeddings")
        if embeddings is None or len(embeddings) == 0 or embeddings[0] is None:
            return None
        return len(list(embeddings[0]))


def _section_order(section: str) -> tuple[int, ...]:
    """Order `7` before `7.3` and `7.3` before `8`."""
    return tuple(int(part) for part in section.split("."))


def _first_query_row(value) -> list:
    """Return the first result row from a Chroma query field."""
    if value is None or len(value) == 0:
        return []
    row = value[0]
    return list(row) if row is not None else []
