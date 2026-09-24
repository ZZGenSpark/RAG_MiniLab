"""Check cosine retrieval ranking and the three-chunk cap."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from adapter.chroma_store import ChromaPolicyStore
from rag.chunking import chunk_policy_file
from rag.retrieve import retrieve
from tests.support import EXPENSE_POLICY_FIXTURE


class FakeEmbedder:
    def __init__(self, query_vector: list[float]) -> None:
        """Store the vector returned for every embed call."""
        self.query_vector = query_vector
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Record the inputs and return the fixed query vector for each."""
        self.calls.append(list(texts))
        return [list(self.query_vector) for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return the embedding vector for a single question."""
        return self.embed_texts([text])[0]


def _section_embeddings() -> list[list[float]]:
    """Return one orthogonal vector per policy section."""
    return [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]


@pytest.fixture
def ranked_store(tmp_path: Path) -> ChromaPolicyStore:
    """Store the six policy chunks with orthogonal section embeddings."""
    chunks = chunk_policy_file(EXPENSE_POLICY_FIXTURE)
    store = ChromaPolicyStore(tmp_path / "chroma")
    store.upsert_chunks(chunks, _section_embeddings())
    return store


def test_retrieve_returns_at_most_three_chunks_sorted_by_cosine_distance(ranked_store: ChromaPolicyStore) -> None:
    """Return at most three chunks, closest cosine distance first."""
    embedder = FakeEmbedder(query_vector=[0.95, 0.2, 0.1, 0.0, 0.0, 0.0])
    hits = retrieve(
        "How much can I spend on food each day?",
        store=ranked_store,
        embedder=embedder,
    )

    assert len(hits) <= 3
    assert [hit.citation_section for hit in hits] == ["1. Meals", "2. Hotels", "3. Airfare"]
    distances = [hit.distance for hit in hits]
    assert distances == sorted(distances)
    assert all(isinstance(distance, float) for distance in distances)
    assert all(isinstance(hit.to_ref().distance, float) for hit in hits)
    assert embedder.calls == [["How much can I spend on food each day?"]]


def test_retrieve_rejects_a_blank_question(ranked_store: ChromaPolicyStore) -> None:
    """Reject a question that has no text before any embedding call."""

    class ExplodingEmbedder:
        def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
            """Fail if a blank question is embedded."""
            raise AssertionError("blank questions are not embedded")

        def embed_query(self, text: str) -> list[float]:
            """Fail if a blank question is embedded."""
            raise AssertionError("blank questions are not embedded")

    with pytest.raises(ValueError, match="empty"):
        retrieve("   ", store=ranked_store, embedder=ExplodingEmbedder())


def test_retrieve_caps_the_caller_limit_at_top_k(ranked_store: ChromaPolicyStore) -> None:
    """Return at most the configured top-k even when the caller asks for more."""
    hits = retrieve(
        "How much can I spend on food each day?",
        store=ranked_store,
        embedder=FakeEmbedder([0.95, 0.2, 0.1, 0.0, 0.0, 0.0]),
        n_results=10,
    )
    assert len(hits) == 3


def test_retrieve_honors_a_smaller_result_limit(ranked_store: ChromaPolicyStore) -> None:
    """Return fewer than top-k when the caller asks for a smaller limit."""
    hits = retrieve(
        "How much can I spend on food each day?",
        store=ranked_store,
        embedder=FakeEmbedder([0.95, 0.2, 0.1, 0.0, 0.0, 0.0]),
        n_results=1,
    )
    assert [hit.citation_section for hit in hits] == ["1. Meals"]


def test_retrieve_does_not_depend_on_exact_keywords(ranked_store: ChromaPolicyStore) -> None:
    """Rank the airfare section first from the query embedding alone."""
    hits = retrieve(
        "Can I book first-class airfare?",
        store=ranked_store,
        embedder=FakeEmbedder([0.05, 0.1, 0.98, 0.0, 0.0, 0.0]),
    )
    assert hits[0].citation_section == "3. Airfare"
    assert hits[0].text.startswith("Employees must purchase economy airfare.")
