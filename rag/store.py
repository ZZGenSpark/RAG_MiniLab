from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import chromadb
from chromadb.api.models.Collection import Collection

from rag.schema import (
    COLLECTION_METADATA,
    COLLECTION_NAME,
    DISTANCE_SPACE,
    EmbeddedChunk,
    PolicyChunk,
    to_chroma_records,
)

DEFAULT_CHROMA_PATH = "chroma_db"


class PolicyStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or os.environ.get("CHROMA_PATH", DEFAULT_CHROMA_PATH))
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection: Collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata=COLLECTION_METADATA,
            embedding_function=None,
        )

    @property
    def distance_space(self) -> str:
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
        records = to_chroma_records(chunks, embeddings)
        self.collection.upsert(**records.as_upsert())
        return records.ids

    def count(self) -> int:
        return self.collection.count()

    def get_all(self) -> list[EmbeddedChunk]:
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
