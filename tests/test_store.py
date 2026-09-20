from pathlib import Path

from rag.chunking import chunk_policy_file
from rag.schema import (
    COLLECTION_NAME,
    DISTANCE_SPACE,
    REQUIRED_METADATA_KEYS,
    to_chroma_records,
)
from rag.store import PolicyStore

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "policy.md"
EMBEDDING_DIM = 8


def _chunks():
    return chunk_policy_file(POLICY_PATH)


def _embeddings(count: int, dim: int = EMBEDDING_DIM):
    return [
        [float(index + 1 if position == index else 0) for position in range(dim)]
        for index in range(count)
    ]


def test_to_chroma_records_maps_assignment_fields() -> None:
    chunks = _chunks()
    embeddings = _embeddings(len(chunks))
    records = to_chroma_records(chunks, embeddings)

    assert records.ids == [chunk.chunk_id for chunk in chunks]
    assert records.documents == [chunk.text for chunk in chunks]
    assert records.embeddings == embeddings
    for metadata, chunk in zip(records.metadatas, chunks, strict=True):
        assert set(metadata.model_dump()) == set(REQUIRED_METADATA_KEYS)
        assert metadata.document == chunk.document
        assert metadata.version == chunk.version
        assert metadata.section == chunk.section
        assert metadata.section_title == chunk.section_title


def test_store_creates_cosine_collection(tmp_path: Path) -> None:
    store = PolicyStore(tmp_path / "chroma")
    assert store.collection.name == COLLECTION_NAME
    assert store.distance_space == DISTANCE_SPACE


def test_store_persists_six_chunks_with_text_embeddings_and_metadata(tmp_path: Path) -> None:
    chunks = _chunks()
    embeddings = _embeddings(len(chunks))
    store = PolicyStore(tmp_path / "chroma")

    written_ids = store.upsert_chunks(chunks, embeddings)

    assert len(written_ids) == 6
    assert store.count() == 6

    stored = store.get_all()
    assert [record.chunk_id for record in stored] == written_ids
    for record, chunk, embedding in zip(stored, chunks, embeddings, strict=True):
        assert record.text == chunk.text
        assert record.embedding == embedding
        assert len(record.embedding) == EMBEDDING_DIM
        for key in REQUIRED_METADATA_KEYS:
            assert getattr(record, key) == getattr(chunk, key)


def test_upsert_is_idempotent_by_stable_chunk_id(tmp_path: Path) -> None:
    chunks = _chunks()
    store = PolicyStore(tmp_path / "chroma")
    store.upsert_chunks(chunks, _embeddings(len(chunks)))
    store.upsert_chunks(chunks, _embeddings(len(chunks)))
    assert store.count() == 6


def test_persistent_client_reloads_records_from_disk(tmp_path: Path) -> None:
    chroma_path = tmp_path / "chroma"
    chunks = _chunks()
    embeddings = _embeddings(len(chunks))

    first = PolicyStore(chroma_path)
    first.upsert_chunks(chunks, embeddings)
    assert first.count() == 6

    reopened = PolicyStore(chroma_path)
    stored = reopened.get_all()
    assert reopened.distance_space == DISTANCE_SPACE
    assert [record.chunk_id for record in stored] == [chunk.chunk_id for chunk in chunks]
    assert [record.embedding for record in stored] == embeddings
