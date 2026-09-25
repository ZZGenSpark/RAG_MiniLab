"""Check the six-chunk ingest path."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from adapter.chroma_store import ChromaPolicyStore
from adapter.sentence_transformer_embeddings import SentenceTransformerEmbeddingAdapter
from config import COLLECTION_NAME, POLICIES_DIR
from rag.chunking import chunk_policy_file
from rag.ingest import _minilm, ingest_corpus, ingest_policy
from rag.schema import PolicyChunk
from tests.support import EXPENSE_POLICY_FIXTURE

EMBEDDING_DIM = 8


class FakeEmbedder:
    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        """Store the embedding width returned for each input."""
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Record the inputs and return one fixed-width vector per text."""
        self.calls.append(list(texts))
        return [
            [float(index + 1 if position == index else 0) for position in range(self.dim)]
            for index, _ in enumerate(texts)
        ]

    def embed_query(self, text: str) -> list[float]:
        """Return the embedding vector for a single question."""
        return self.embed_texts([text])[0]


def test_ingest_embeds_chunks_and_upserts_six_chroma_records(tmp_path: Path) -> None:
    """Embed the policy and store six records with stable chunk ids."""
    embedder = FakeEmbedder()
    store = ChromaPolicyStore(tmp_path / "chroma")

    written_ids = ingest_policy(
        EXPENSE_POLICY_FIXTURE,
        store=store,
        embedder=embedder,
    )

    assert written_ids == [
        "employee-expense-policy:v2.0:section-1",
        "employee-expense-policy:v2.0:section-2",
        "employee-expense-policy:v2.0:section-3",
        "employee-expense-policy:v2.0:section-4",
        "employee-expense-policy:v2.0:section-5",
        "employee-expense-policy:v2.0:section-6",
    ]
    stored = store.get_all()
    assert store.count() == 6
    assert [record.chunk_id for record in stored] == written_ids
    assert all(len(record.embedding) == EMBEDDING_DIM for record in stored)
    assert stored[0].text.startswith("Employees may claim up to $65 per day")
    assert embedder.calls == [[chunk.text for chunk in chunk_policy_file(EXPENSE_POLICY_FIXTURE)]]


def test_ingest_reads_every_markdown_file_in_a_directory(tmp_path: Path) -> None:
    """Treat a directory as the policy input and store its markdown sections."""
    policies = tmp_path / "policies"
    policies.mkdir()
    (policies / "expense-policy.md").write_text(
        EXPENSE_POLICY_FIXTURE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    embedder = FakeEmbedder()
    store = ChromaPolicyStore(tmp_path / "chroma")

    written_ids = ingest_policy(policies, store=store, embedder=embedder)

    assert written_ids == [chunk.chunk_id for chunk in chunk_policy_file(EXPENSE_POLICY_FIXTURE)]
    assert store.count() == 6


def test_ingest_rejects_a_directory_with_no_markdown(tmp_path: Path) -> None:
    """Reject a policy directory that has nothing to chunk."""
    policies = tmp_path / "policies"
    policies.mkdir()
    with pytest.raises(ValueError, match="no markdown policies"):
        ingest_policy(policies, store=ChromaPolicyStore(tmp_path / "chroma"), embedder=FakeEmbedder())


def test_ingest_rejects_duplicate_chunk_ids_before_embedding(tmp_path: Path) -> None:
    """Refuse a directory whose files would overwrite one another in the store."""
    policies = tmp_path / "policies"
    policies.mkdir()
    text = EXPENSE_POLICY_FIXTURE.read_text(encoding="utf-8")
    (policies / "a.md").write_text(text, encoding="utf-8")
    (policies / "b.md").write_text(text, encoding="utf-8")

    class ExplodingEmbedder:
        def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
            """Fail if colliding policies are embedded."""
            raise AssertionError("duplicate chunk ids are rejected before embedding")

        def embed_query(self, text: str) -> list[float]:
            """Fail if colliding policies are embedded."""
            raise AssertionError("duplicate chunk ids are rejected before embedding")

    with pytest.raises(ValueError, match="unique"):
        ingest_policy(
            policies,
            store=ChromaPolicyStore(tmp_path / "chroma"),
            embedder=ExplodingEmbedder(),
        )


def test_ingest_corpus_defaults_to_minilm_without_loading_weights() -> None:
    """Use the local MiniLM adapter, and do not download weights until encode."""
    embedder = _minilm()
    assert isinstance(embedder, SentenceTransformerEmbeddingAdapter)
    assert embedder._encoder is None


def test_ingest_corpus_indexes_policy_markdown_and_drops_stale_ids(tmp_path: Path) -> None:
    """Embed every markdown policy, keep rule 7.3, and delete ids absent from the batch."""
    embedder = FakeEmbedder()
    store = ChromaPolicyStore(tmp_path / "chroma")
    stale = PolicyChunk(
        chunk_id="old-policy:v1.0:section-1",
        document="Old Policy",
        version="1.0",
        section="1",
        section_title="Retired",
        text="This section is no longer in the corpus.",
    )
    store.upsert_chunks([stale], _vector())

    written_ids = ingest_corpus(POLICIES_DIR, store=store, embedder=embedder)

    expected_ids = [chunk.chunk_id for path in sorted(POLICIES_DIR.glob("*.md")) for chunk in chunk_policy_file(path)]
    assert store.collection.name == COLLECTION_NAME == "company_policies"
    assert written_ids == expected_ids
    assert "hr-policy:v2.0:section-7.3" in written_ids
    assert store.count() == len(written_ids)
    assert stale.chunk_id not in {record.chunk_id for record in store.get_all()}
    expected_texts = [chunk.text for path in sorted(POLICIES_DIR.glob("*.md")) for chunk in chunk_policy_file(path)]
    assert embedder.calls == [expected_texts]


def test_ingest_corpus_ignores_binaries_in_the_policies_directory(tmp_path: Path) -> None:
    """Read markdown only, even when a PDF sits beside it."""
    policies = tmp_path / "policies"
    policies.mkdir()
    (policies / "expense-policy.md").write_text(EXPENSE_POLICY_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    (policies / "notes.pdf").write_bytes(b"%PDF-1.4")
    store = ChromaPolicyStore(tmp_path / "chroma")

    written_ids = ingest_corpus(policies, store=store, embedder=FakeEmbedder())

    assert written_ids == [chunk.chunk_id for chunk in chunk_policy_file(EXPENSE_POLICY_FIXTURE)]
    assert store.count() == 6


def _vector() -> list[list[float]]:
    """Return one embedding wide enough for the fake embedder."""
    return [[float(position) for position in range(EMBEDDING_DIM)]]
