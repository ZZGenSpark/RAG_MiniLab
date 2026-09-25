"""Record one audit row per ask, including source conflicts and hybrid candidates."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from adapter.chroma_store import ChromaPolicyStore
from rag.ask import ask
from rag.audit import source_conflicts
from rag.chunking import chunk_policy_file
from rag.route import RetrievalDecision, Strategy
from rag.schema import PolicyChunk
from rag.slug import slugify
from tests.support import EXPENSE_POLICY_FIXTURE, KeepingReranker

TIME = "Time & Usage Policy"


class FixedEmbedder:
    """Return one query vector."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return the query vector once per text."""
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return the query vector."""
        return [1.0, 0.0, 0.0]


class FixedRouter:
    """Choose one scripted strategy."""

    def __init__(self, strategy: Strategy) -> None:
        """Store the strategy this router returns."""
        self.strategy = strategy

    def choose(self, question: str) -> RetrievalDecision:
        """Return the scripted decision."""
        return RetrievalDecision(strategy=self.strategy)


class NotAnswerable:
    """Tell generation that the excerpt does not answer."""

    def complete(self, prompt: str) -> str:
        """Return one unanswerable model payload."""
        return '{"answerable": false, "answer": ""}'


def test_two_versions_of_one_title_are_a_source_conflict() -> None:
    """Flag a title when the chunks sent to the model include two versions of it."""
    conflicts = source_conflicts(
        [
            _chunk(TIME, "1.0", "1", "Token Allotment", "Employees receive a token allotment."),
            _chunk(TIME, "2.0", "1", "Token Allotment", "Employees receive a revised token allotment."),
            _chunk("HR Policy", "2.0", "1", "Purpose", "This policy explains workplace rules."),
        ]
    )
    assert [(conflict.document, conflict.versions) for conflict in conflicts] == [(TIME, ["1.0", "2.0"])]


def test_one_version_leaves_source_conflicts_empty() -> None:
    """Leave the flag empty when each title appears at only one version."""
    one_version = [
        _chunk(TIME, "2.0", "1", "Token Allotment", "Employees receive a token allotment."),
        _chunk(TIME, "2.0", "2", "Foosball", "Foosball is capped at a short session."),
        _chunk("HR Policy", "2.0", "1", "Purpose", "This policy explains workplace rules."),
    ]
    assert source_conflicts(one_version) == []
    assert source_conflicts(one_version[:1]) == []


def test_ask_records_a_conflict_from_the_chunks_sent_to_the_model(tmp_path: Path) -> None:
    """Write one JSONL row whose conflict names the title sent at two versions."""
    path = tmp_path / "audit.jsonl"
    store = _store(
        tmp_path,
        [
            (_chunk(TIME, "1.0", "1", "Token Allotment", "Employees receive a token allotment."), [1.0, 0.0, 0.0]),
            (
                _chunk(TIME, "2.0", "1", "Token Allotment", "Employees receive a revised token allotment."),
                [0.8, 0.2, 0.0],
            ),
            (_chunk("HR Policy", "2.0", "1", "Purpose", "This policy explains workplace rules."), [0.0, 0.0, 1.0]),
        ],
    )

    response = ask(
        "What allotment do employees receive?",
        store=store,
        embedder=FixedEmbedder(),
        generator=NotAnswerable(),
        router=FixedRouter("vector"),
        reranker=KeepingReranker(),
        audit_path=path,
    )

    rows = _rows(path)
    assert len(rows) == 1
    assert rows[0]["strategy"] == "vector"
    assert rows[0]["bm25_ids"] == []
    assert rows[0]["rrf_ids"] == []
    assert response.source_conflicts == source_conflicts(
        [
            _chunk(TIME, "1.0", "1", "Token Allotment", "Employees receive a token allotment."),
            _chunk(TIME, "2.0", "1", "Token Allotment", "Employees receive a revised token allotment."),
        ]
    )
    assert rows[0]["source_conflicts"] == [{"document": TIME, "versions": ["1.0", "2.0"]}]


def test_ask_leaves_source_conflicts_empty_for_one_version(tmp_path: Path) -> None:
    """Write an empty conflict list when the sent chunks share one version."""
    path = tmp_path / "audit.jsonl"
    store = _store(
        tmp_path,
        [
            (_chunk(TIME, "2.0", "1", "Token Allotment", "Employees receive a token allotment."), [1.0, 0.0, 0.0]),
            (_chunk("HR Policy", "2.0", "1", "Purpose", "This policy explains workplace rules."), [0.0, 0.0, 1.0]),
        ],
    )

    response = ask(
        "What allotment do employees receive?",
        store=store,
        embedder=FixedEmbedder(),
        generator=NotAnswerable(),
        router=FixedRouter("vector"),
        reranker=KeepingReranker(),
        audit_path=path,
    )

    assert response.source_conflicts == []
    assert _rows(path)[0]["source_conflicts"] == []


def test_hybrid_audit_row_contains_both_legs_rrf_ids_and_scores(tmp_path: Path) -> None:
    """A hybrid ask records the vector ids, keyword ids, fused ids, and cross-encoder scores."""
    path = tmp_path / "audit.jsonl"
    chunks = chunk_policy_file(EXPENSE_POLICY_FIXTURE)
    store = ChromaPolicyStore(tmp_path / "chroma")
    store.upsert_chunks(
        chunks,
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
    )

    ask(
        "How much can I spend on food each day?",
        store=store,
        embedder=_ExpenseEmbedder(),
        generator=NotAnswerable(),
        router=FixedRouter("hybrid"),
        reranker=KeepingReranker(),
        audit_path=path,
    )

    row = _rows(path)[0]
    assert row["strategy"] == "hybrid"
    assert row["vector_ids"]
    assert row["bm25_ids"]
    assert len(row["rrf_ids"]) == 5
    assert [item["chunk_id"] for item in row["cross_encoder_scores"]] == row["rrf_ids"]
    assert len(row["cross_encoder_scores"]) == 5
    assert row["source_conflicts"] == []


class _ExpenseEmbedder:
    """Return the meals-shaped query used by the expense ranking tests."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one query vector per text."""
        return [[0.95, 0.2, 0.1, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return the query vector."""
        return [0.95, 0.2, 0.1, 0.0, 0.0, 0.0]


def _rows(path: Path) -> list[dict[str, Any]]:
    """Read the JSONL audit file."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _store(tmp_path: Path, rows: list[tuple[PolicyChunk, list[float]]]) -> ChromaPolicyStore:
    """Store the given chunks with the paired vectors."""
    store = ChromaPolicyStore(tmp_path / "chroma")
    store.upsert_chunks([chunk for chunk, _ in rows], [vector for _, vector in rows])
    return store


def _chunk(document: str, version: str, section: str, title: str, text: str) -> PolicyChunk:
    """Build a chunk whose id encodes the document, version, and section."""
    return PolicyChunk(
        chunk_id=f"{slugify(document)}:v{version}:section-{section}",
        document=document,
        version=version,
        section=section,
        section_title=title,
        text=text,
    )
