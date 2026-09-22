"""Check cosine retrieval ranking, the three-chunk cap, and live questions."""

from pathlib import Path

import pytest

from adapter.chroma_store import ChromaPolicyStore
from config import POLICY_PATH
from rag.chunking import chunk_policy_file
from rag.ingest import ingest_policy
from rag.retrieve import retrieve

REQUIRED_QUESTIONS = [
    ("How much can I spend on food each day?", "1. Meals"),
    ("Can I book first-class airfare?", "3. Airfare"),
    ("My hotel costs $250. What do I need?", "2. Hotels"),
    ("Do I need a receipt for a $20 taxi?", "5. Receipts"),
    ("Can I claim a limousine upgrade?", "4. Ground Transportation"),
    ("Does the company reimburse gym memberships?", None),
]


class FakeEmbedder:
    def __init__(self, query_vector: list[float]) -> None:
        """Store the vector returned for every embed call."""
        self.query_vector = query_vector
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
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
    chunks = chunk_policy_file(POLICY_PATH)
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


def test_retrieve_does_not_depend_on_exact_keywords(ranked_store: ChromaPolicyStore) -> None:
    """Rank the airfare section first from the query embedding alone."""
    hits = retrieve(
        "Can I book first-class airfare?",
        store=ranked_store,
        embedder=FakeEmbedder([0.05, 0.1, 0.98, 0.0, 0.0, 0.0]),
    )
    assert hits[0].citation_section == "3. Airfare"
    assert hits[0].text.startswith("Employees must purchase economy airfare.")


@pytest.fixture(scope="module")
def live_store(tmp_path_factory: pytest.TempPathFactory) -> ChromaPolicyStore:
    """Ingest the policy into a temporary store, or skip if Ollama is down."""
    chroma_path = tmp_path_factory.mktemp("live-chroma")
    try:
        ingest_policy(POLICY_PATH, chroma_path=chroma_path)
    except Exception as exc:
        pytest.skip(f"live Ollama ingest unavailable: {exc}")
    return ChromaPolicyStore(chroma_path)


@pytest.mark.parametrize("question, expected_section", REQUIRED_QUESTIONS[:5])
def test_live_retrieve_finds_expected_section(
    live_store: ChromaPolicyStore,
    question: str,
    expected_section: str,
) -> None:
    """Confirm live retrieval includes the expected section within three chunks."""
    hits = retrieve(question, store=live_store)
    assert len(hits) <= 3
    distances = [hit.distance for hit in hits]
    assert distances == sorted(distances)
    assert all(isinstance(distance, float) for distance in distances)
    assert expected_section in [hit.citation_section for hit in hits]


def test_live_retrieve_caps_unsupported_question(live_store: ChromaPolicyStore) -> None:
    """Cap an unsupported live question at three sorted chunks."""
    hits = retrieve("Does the company reimburse gym memberships?", store=live_store)
    assert 0 < len(hits) <= 3
    distances = [hit.distance for hit in hits]
    assert distances == sorted(distances)
    assert all(isinstance(distance, float) for distance in distances)
