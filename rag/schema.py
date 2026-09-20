from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

COLLECTION_NAME = "expense_policy"
DISTANCE_SPACE = "cosine"
COLLECTION_METADATA = {"hnsw:space": DISTANCE_SPACE}
CHUNK_ID_PREFIX = "expense-policy"


class ChunkMetadata(BaseModel):
    """Citation fields stored beside each Chroma vector."""

    model_config = ConfigDict(extra="forbid")

    document: str = Field(min_length=1)
    version: str = Field(pattern=r"^\d+\.\d+$")
    section: str = Field(pattern=r"^\d+$")
    section_title: str = Field(min_length=1)

    @property
    def citation_section(self) -> str:
        return f"{self.section}. {self.section_title}"


REQUIRED_METADATA_KEYS = tuple(ChunkMetadata.model_fields)


class PolicyChunk(ChunkMetadata):
    """One numbered policy section before an embedding is attached."""

    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def chunk_id_matches_version_and_section(self) -> PolicyChunk:
        expected = f"{CHUNK_ID_PREFIX}:v{self.version}:section-{self.section}"
        if self.chunk_id != expected:
            raise ValueError(f"chunk_id must be {expected}")
        return self

    def to_metadata(self) -> ChunkMetadata:
        return ChunkMetadata(
            document=self.document,
            version=self.version,
            section=self.section,
            section_title=self.section_title,
        )


class EmbeddedChunk(PolicyChunk):
    """Assignment record: original text, complete vector, and metadata."""

    embedding: list[float] = Field(min_length=1)

    @classmethod
    def from_stored(
        cls,
        chunk_id: str,
        text: str,
        metadata: dict[str, object],
        embedding: Sequence[float],
    ) -> EmbeddedChunk:
        return cls.model_validate(
            {
                "chunk_id": chunk_id,
                "text": text,
                **metadata,
                "embedding": [float(value) for value in embedding],
            }
        )


class ChromaRecords(BaseModel):
    """Field mapping for one Chroma upsert."""

    model_config = ConfigDict(extra="forbid")

    ids: list[str]
    documents: list[str]
    embeddings: list[list[float]]
    metadatas: list[ChunkMetadata]

    @model_validator(mode="after")
    def aligned_record_fields(self) -> ChromaRecords:
        lengths = {
            len(self.ids),
            len(self.documents),
            len(self.embeddings),
            len(self.metadatas),
        }
        if len(lengths) != 1:
            raise ValueError("ids, documents, embeddings, and metadatas must be the same length")
        return self

    def as_upsert(self) -> dict[str, list]:
        return {
            "ids": self.ids,
            "documents": self.documents,
            "embeddings": self.embeddings,
            "metadatas": [metadata.model_dump() for metadata in self.metadatas],
        }


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document: str = Field(min_length=1)
    version: str = Field(pattern=r"^\d+\.\d+$")
    section: str = Field(min_length=1, pattern=r"^\d+\.\s+.+$")


class RetrievedChunkRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section: str = Field(min_length=1)
    distance: float


class RetrievedChunk(PolicyChunk):
    """A ranked policy excerpt returned by cosine retrieval."""

    distance: float

    def to_ref(self) -> RetrievedChunkRef:
        return RetrievedChunkRef(section=self.citation_section, distance=self.distance)


REFUSAL_ANSWER = "The provided policy does not answer this question."


class GroundedModelOutput(BaseModel):
    """JSON the generation model is asked to return."""

    model_config = ConfigDict(extra="ignore")

    answer: str = Field(min_length=1)
    section: str | None = None


class AskResponse(BaseModel):
    """Structured output required by the assignment."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    citation: Citation | None
    retrieved_chunks: list[RetrievedChunkRef] = Field(max_length=3)

    @field_validator("retrieved_chunks")
    @classmethod
    def distances_are_sorted(cls, chunks: list[RetrievedChunkRef]) -> list[RetrievedChunkRef]:
        distances = [chunk.distance for chunk in chunks]
        if distances != sorted(distances):
            raise ValueError("retrieved_chunks must be sorted by cosine distance ascending")
        return chunks


def to_chroma_records(
    chunks: Sequence[PolicyChunk],
    embeddings: Sequence[Sequence[float]],
) -> ChromaRecords:
    if len(chunks) != len(embeddings):
        raise ValueError("each chunk must have exactly one embedding vector")

    embedded = [
        EmbeddedChunk.model_validate({**chunk.model_dump(), "embedding": list(embedding)})
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]
    return ChromaRecords(
        ids=[chunk.chunk_id for chunk in embedded],
        documents=[chunk.text for chunk in embedded],
        embeddings=[chunk.embedding for chunk in embedded],
        metadatas=[chunk.to_metadata() for chunk in embedded],
    )
