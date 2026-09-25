"""Append one JSON object per ask.

The row records the retrieval decision, the candidate lists, and any document
title that reached the model at more than one version.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from config import AUDIT_PATH
from rag.schema import PolicyChunk, SourceConflict


class ChunkScore(BaseModel):
    """One cross-encoder score for a shortlisted chunk."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    score: float


class AuditRecord(BaseModel):
    """One ask, written as a single JSONL row."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    strategy: str = Field(pattern=r"^(vector|hybrid)$")
    vector_ids: list[str]
    bm25_ids: list[str]
    rrf_ids: list[str]
    cross_encoder_scores: list[ChunkScore]
    source_conflicts: list[SourceConflict]


def source_conflicts(chunks: Sequence[PolicyChunk]) -> list[SourceConflict]:
    """List titles that appear at more than one version in these chunks.

    One version of a title is not a conflict. The amounts in the text are not
    consulted.
    """
    versions_by_title: dict[str, set[str]] = {}
    for chunk in chunks:
        versions_by_title.setdefault(chunk.document, set()).add(chunk.version)
    conflicts = [
        SourceConflict(document=title, versions=sorted(found, key=_version_key))
        for title, found in versions_by_title.items()
        if len(found) > 1
    ]
    conflicts.sort(key=lambda conflict: conflict.document)
    return conflicts


def append_audit(record: AuditRecord, path: Path | None = None) -> Path:
    """Append one JSON object as a line. Return the file that received it."""
    destination = path or AUDIT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")
    return destination


def _version_key(version: str) -> tuple[int, ...]:
    """Order 1.0 before 2.0."""
    return tuple(int(part) for part in version.split("."))
