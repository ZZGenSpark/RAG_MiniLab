"""Check Chroma upserts, cosine metadata, and reload from disk."""

from pathlib import Path

import pytest

from adapter.chroma_store import ChromaPolicyStore
from config import POLICY_PATH
from rag.chunking import chunk_policy_file
from rag.schema import (
    COLLECTION_NAME,
    DISTANCE_SPACE,
    REQUIRED_METADATA_KEYS,
    to_chroma_records,
)

EMBEDDING_DIM = 8


def _chunks():
    """Load the six chunks from the policy file."""
    return chunk_policy_file(POLICY_PATH)


def _embeddings(count: int, dim: int = EMBEDDING_DIM):
    """Build one distinct fixed-width vector per chunk."""
    return [
        [float(index + 1 if position == index else 0) for position in range(dim)]
        for index in range(count)
    ]


def test_to_chroma_records_maps_assignment_fields() -> None:
    """Map chunk text, ids, embeddings, and citation metadata into Chroma fields."""
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
    """Create the expense-policy collection in cosine space."""
    store = ChromaPolicyStore(tmp_path / "chroma")
    assert store.collection.name == COLLECTION_NAME
    assert store.distance_space == DISTANCE_SPACE


def test_store_persists_six_chunks_with_text_embeddings_and_metadata(tmp_path: Path) -> None:
    """Store six chunks with their original text, vectors, and metadata."""
    chunks = _chunks()
    embeddings = _embeddings(len(chunks))
    store = ChromaPolicyStore(tmp_path / "chroma")

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
    """Keep six records when the same chunk ids are upserted twice."""
    chunks = _chunks()
    store = ChromaPolicyStore(tmp_path / "chroma")
    store.upsert_chunks(chunks, _embeddings(len(chunks)))
    store.upsert_chunks(chunks, _embeddings(len(chunks)))
    assert store.count() == 6


def test_upsert_drops_ids_missing_from_the_batch(tmp_path: Path) -> None:
    """Delete a stored section that the next ingest no longer includes."""
    chunks = _chunks()
    store = ChromaPolicyStore(tmp_path / "chroma")
    store.upsert_chunks(chunks, _embeddings(len(chunks)))

    store.upsert_chunks(chunks[:5], _embeddings(5))

    assert store.count() == 5
    assert [record.section for record in store.get_all()] == ["1", "2", "3", "4", "5"]


def test_query_on_an_empty_collection_returns_no_hits(tmp_path: Path) -> None:
    """Return no hits when nothing has been stored."""
    store = ChromaPolicyStore(tmp_path / "chroma")
    assert store.query_similar([1.0, 0.0], n_results=3) == []
    assert store.get_all() == []


def test_query_rejects_a_non_positive_result_limit(tmp_path: Path) -> None:
    """Reject a request for zero nearest chunks."""
    store = ChromaPolicyStore(tmp_path / "chroma")
    with pytest.raises(ValueError, match="at least 1"):
        store.query_similar([1.0, 0.0], n_results=0)


def test_query_rejects_an_empty_vector(tmp_path: Path) -> None:
    """Reject a query that has no embedding components."""
    chunks = _chunks()
    store = ChromaPolicyStore(tmp_path / "chroma")
    store.upsert_chunks(chunks, _embeddings(len(chunks)))
    with pytest.raises(ValueError, match="complete vector"):
        store.query_similar([], n_results=1)


def test_get_all_sorts_by_section_number(tmp_path: Path) -> None:
    """Return stored chunks in section order even when the batch is reversed."""
    chunks = _chunks()
    reversed_chunks = list(reversed(chunks))
    store = ChromaPolicyStore(tmp_path / "chroma")
    store.upsert_chunks(reversed_chunks, _embeddings(len(reversed_chunks)))
    assert [record.section for record in store.get_all()] == ["1", "2", "3", "4", "5", "6"]


def test_query_can_return_every_stored_chunk(tmp_path: Path) -> None:
    """Honor n_results above the ask cap of three."""
    chunks = _chunks()
    store = ChromaPolicyStore(tmp_path / "chroma")
    store.upsert_chunks(chunks, _embeddings(len(chunks)))

    hits = store.query_similar([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], n_results=6)

    assert len(hits) == 6


def test_upsert_rejects_a_different_embedding_width(tmp_path: Path) -> None:
    """Keep the stored vectors when a later batch uses another width."""
    chunks = _chunks()
    store = ChromaPolicyStore(tmp_path / "chroma")
    store.upsert_chunks(chunks, _embeddings(len(chunks)))

    with pytest.raises(ValueError, match="width"):
        store.upsert_chunks(chunks, _embeddings(len(chunks), dim=4))

    assert store.count() == 6


def test_query_rejects_a_different_embedding_width(tmp_path: Path) -> None:
    """Reject a query vector that does not match the stored width."""
    chunks = _chunks()
    store = ChromaPolicyStore(tmp_path / "chroma")
    store.upsert_chunks(chunks, _embeddings(len(chunks)))

    with pytest.raises(ValueError, match="width"):
        store.query_similar([1.0, 0.0], n_results=1)


def test_store_rejects_a_collection_that_is_not_cosine(tmp_path: Path) -> None:
    """Fail when an existing collection was created in another distance space."""
    import chromadb

    path = tmp_path / "chroma"
    client = chromadb.PersistentClient(path=str(path))
    client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "l2"})

    with pytest.raises(ValueError, match="cosine"):
        ChromaPolicyStore(path)


def test_to_chroma_records_rejects_mixed_widths() -> None:
    """Reject a batch whose vectors do not share one width."""
    chunks = _chunks()[:2]
    with pytest.raises(ValueError, match="width"):
        to_chroma_records(chunks, [[1.0, 0.0], [1.0, 0.0, 0.0]])


def test_persistent_client_reloads_records_from_disk(tmp_path: Path) -> None:
    """Reload the same chunks and cosine space from a new store client."""
    chroma_path = tmp_path / "chroma"
    chunks = _chunks()
    embeddings = _embeddings(len(chunks))

    first = ChromaPolicyStore(chroma_path)
    first.upsert_chunks(chunks, embeddings)
    assert first.count() == 6

    reopened = ChromaPolicyStore(chroma_path)
    stored = reopened.get_all()
    assert reopened.distance_space == DISTANCE_SPACE
    assert [record.chunk_id for record in stored] == [chunk.chunk_id for chunk in chunks]
    assert [record.embedding for record in stored] == embeddings
