from pathlib import Path
from types import SimpleNamespace

import pytest

from rag.chunking import chunk_policy_file
from rag.embeddings import Embedder
from rag.ingest import ingest_policy
from rag.store import PolicyStore

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "policy.md"
EMBEDDING_DIM = 8


class FakeOllama:
    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, model: str, input: str | list[str]):
        texts = input if isinstance(input, list) else [input]
        self.calls.append(list(texts))
        return SimpleNamespace(
            embeddings=[
                [float(index + 1 if position == index else 0) for position in range(self.dim)]
                for index, _ in enumerate(texts)
            ]
        )


def test_embedder_returns_one_complete_vector_per_chunk() -> None:
    chunks = chunk_policy_file(POLICY_PATH)
    client = FakeOllama()
    embedder = Embedder(model="nomic-embed-text", client=client)

    vectors = embedder.embed_texts([chunk.text for chunk in chunks])

    assert len(vectors) == 6
    assert client.calls[0] == [chunk.text for chunk in chunks]
    assert all(len(vector) == EMBEDDING_DIM for vector in vectors)


def test_embedder_rejects_empty_vectors() -> None:
    class EmptyClient:
        def embed(self, model: str, input: str | list[str]):
            texts = input if isinstance(input, list) else [input]
            return SimpleNamespace(embeddings=[[] for _ in texts])

    with pytest.raises(ValueError, match="complete vector"):
        Embedder(client=EmptyClient()).embed_texts(["Employees may claim up to $65 per day."])


def test_ingest_embeds_chunks_and_upserts_six_chroma_records(tmp_path: Path) -> None:
    client = FakeOllama()
    store = PolicyStore(tmp_path / "chroma")

    written_ids = ingest_policy(
        POLICY_PATH,
        store=store,
        embedder=Embedder(model="nomic-embed-text", client=client),
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
