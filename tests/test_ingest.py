"""Check embedding completeness and the six-chunk ingest path."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from adapter.chroma_store import ChromaPolicyStore
from adapter.ollama_embeddings import DOCUMENT_PREFIX, QUERY_PREFIX, OllamaEmbeddingAdapter
from config import POLICY_PATH
from rag.chunking import chunk_policy_file
from rag.ingest import ingest_policy

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
    chunks = chunk_policy_file(POLICY_PATH)
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
        POLICY_PATH,
        store=store,
        embedder=embedder,
    )

    assert written_ids == [
        "expense-policy:v2.0:section-1",
        "expense-policy:v2.0:section-2",
        "expense-policy:v2.0:section-3",
        "expense-policy:v2.0:section-4",
        "expense-policy:v2.0:section-5",
        "expense-policy:v2.0:section-6",
    ]
    stored = store.get_all()
    assert store.count() == 6
    assert [record.chunk_id for record in stored] == written_ids
    assert all(len(record.embedding) == EMBEDDING_DIM for record in stored)
    assert stored[0].text.startswith("Employees may claim up to $65 per day")
