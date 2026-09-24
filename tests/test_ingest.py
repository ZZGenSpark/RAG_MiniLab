"""Check embedding completeness and the six-chunk ingest path."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from adapter.chroma_store import ChromaPolicyStore
from adapter.ollama_embeddings import DOCUMENT_PREFIX, QUERY_PREFIX, OllamaEmbeddingAdapter
from rag.chunking import chunk_policy_file
from rag.ingest import ingest_policy
from tests.support import EXPENSE_POLICY_FIXTURE

EMBEDDING_DIM = 8


class FakeOllama:
    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        """Store the embedding width returned for each input."""
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, model: str, input: str | list[str]):
        """Record the inputs and return one fixed-width vector per text."""
        texts = input if isinstance(input, list) else [input]
        self.calls.append(list(texts))
        return SimpleNamespace(
            embeddings=[
                [float(index + 1 if position == index else 0) for position in range(self.dim)]
                for index, _ in enumerate(texts)
            ]
        )


class FakeEmbedder:
    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        """Store the embedding width returned for each input."""
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Record the inputs and return one fixed-width vector per text."""
        self.calls.append(list(texts))
        return [
            [float(index + 1 if position == index else 0) for position in range(self.dim)]
            for index, _ in enumerate(texts)
        ]

    def embed_query(self, text: str) -> list[float]:
        """Return the embedding vector for a single question."""
        return self.embed_texts([text])[0]


def test_embedder_returns_one_complete_vector_per_chunk() -> None:
    """Return one full vector for each of the six policy chunks."""
    chunks = chunk_policy_file(EXPENSE_POLICY_FIXTURE)
    client = FakeOllama()
    embedder = OllamaEmbeddingAdapter(model="nomic-embed-text", client=client)

    vectors = embedder.embed_texts([chunk.text for chunk in chunks])

    assert len(vectors) == 6
    assert client.calls[0] == [f"{DOCUMENT_PREFIX}{chunk.text}" for chunk in chunks]
    assert all(len(vector) == EMBEDDING_DIM for vector in vectors)


def test_query_embedding_uses_the_search_prefix() -> None:
    """Prefix questions for nomic without changing how documents are prefixed."""
    client = FakeOllama()
    embedder = OllamaEmbeddingAdapter(model="nomic-embed-text", client=client)

    vector = embedder.embed_query("How much can I spend on food each day?")

    assert client.calls == [[f"{QUERY_PREFIX}How much can I spend on food each day?"]]
    assert len(vector) == EMBEDDING_DIM


def test_embedder_returns_nothing_for_an_empty_batch() -> None:
    """Skip the embedding client when there is no text to embed."""

    class ExplodingClient:
        def embed(self, model: str, input: str | list[str]):
            """Fail if an empty batch reaches the client."""
            raise AssertionError("empty batches are not sent to the embedding client")

    assert OllamaEmbeddingAdapter(client=ExplodingClient()).embed_texts([]) == []


def test_embedder_rejects_a_short_vector_batch() -> None:
    """Reject a client that returns fewer vectors than input texts."""

    class ShortClient:
        def embed(self, model: str, input: str | list[str]):
            """Return a single vector regardless of how many texts were sent."""
            return SimpleNamespace(embeddings=[[0.1, 0.2]])

    with pytest.raises(ValueError, match="one vector per chunk"):
        OllamaEmbeddingAdapter(client=ShortClient()).embed_texts(
            ["Employees may claim up to $65 per day.", "Hotels are reimbursable."]
        )


def test_embedder_rejects_empty_vectors() -> None:
    """Reject an embedding client that returns empty vectors."""

    class EmptyClient:
        def embed(self, model: str, input: str | list[str]):
            """Return an empty vector for each input text."""
            texts = input if isinstance(input, list) else [input]
            return SimpleNamespace(embeddings=[[] for _ in texts])

    with pytest.raises(ValueError, match="complete vector"):
        OllamaEmbeddingAdapter(client=EmptyClient()).embed_texts(["Employees may claim up to $65 per day."])


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
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
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
