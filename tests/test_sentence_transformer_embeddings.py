"""Check the local MiniLM embedder without downloading the model."""

from collections.abc import Sequence

import pytest

from adapter.sentence_transformer_embeddings import SentenceTransformerEmbeddingAdapter
from config import SENTENCE_TRANSFORMER_MODEL
from rag.ingest import ingest_policy
from tests.support import EXPENSE_POLICY_FIXTURE


class FakeEncoder:
    """Return one short vector per text and record the encode call."""

    def __init__(self, rows: list[list[float]] | None = None) -> None:
        """Store scripted rows. Each input gets one row when rows is omitted."""
        self.rows = rows
        self.calls: list[list[str]] = []
        self.kwargs: list[dict[str, object]] = []

    def encode(self, sentences: str | Sequence[str], **kwargs: object) -> list[list[float]]:
        """Record the sentences and return one vector each."""
        texts = [sentences] if isinstance(sentences, str) else list(sentences)
        self.calls.append(texts)
        self.kwargs.append(kwargs)
        if self.rows is not None:
            return self.rows
        return [[float(index + 1), 0.0] for index, _ in enumerate(texts)]


def test_embedder_returns_one_normalized_vector_per_text() -> None:
    """Encode each chunk once, with cosine normalization, and no query prefix."""
    encoder = FakeEncoder()
    embedder = SentenceTransformerEmbeddingAdapter(encoder=encoder)
    texts = ["Employees may claim up to $65 per day.", "Hotels are reimbursable up to $225 per night."]

    vectors = embedder.embed_texts(texts)

    assert embedder.model_name == SENTENCE_TRANSFORMER_MODEL
    assert vectors == [[1.0, 0.0], [2.0, 0.0]]
    assert encoder.calls == [texts]
    assert encoder.kwargs == [{"normalize_embeddings": True}]


def test_query_uses_the_same_encoder_without_a_prefix() -> None:
    """Embed a question as plain text. MiniLM does not use a search prefix."""
    encoder = FakeEncoder()
    embedder = SentenceTransformerEmbeddingAdapter(encoder=encoder)

    vector = embedder.embed_query("How many tokens does an employee receive?")

    assert vector == [1.0, 0.0]
    assert encoder.calls == [["How many tokens does an employee receive?"]]


def test_empty_batch_does_not_load_the_encoder() -> None:
    """Skip the model when there is no text to embed."""

    class ExplodingEncoder:
        def encode(self, sentences: str | Sequence[str], **kwargs: object) -> list[list[float]]:
            """Fail if an empty batch reaches the encoder."""
            raise AssertionError("empty batches are not encoded")

    assert SentenceTransformerEmbeddingAdapter(encoder=ExplodingEncoder()).embed_texts([]) == []


def test_embedder_rejects_a_short_vector_batch() -> None:
    """Reject an encoder that returns fewer vectors than input texts."""
    encoder = FakeEncoder(rows=[[0.1, 0.2]])
    with pytest.raises(ValueError, match="one vector per chunk"):
        SentenceTransformerEmbeddingAdapter(encoder=encoder).embed_texts(
            ["Employees may claim up to $65 per day.", "Hotels are reimbursable up to $225 per night."]
        )


def test_embedder_rejects_an_empty_vector() -> None:
    """Reject an encoder that returns a vector with no components."""
    encoder = FakeEncoder(rows=[[]])
    with pytest.raises(ValueError, match="complete vector"):
        SentenceTransformerEmbeddingAdapter(encoder=encoder).embed_texts(["Employees may claim up to $65 per day."])


def test_numpy_rows_are_converted_to_plain_floats() -> None:
    """Accept the ndarray shape SentenceTransformer.encode returns."""

    class NumpyLike:
        def __init__(self, rows: list[list[int]]) -> None:
            """Store the integer rows that `tolist` returns."""
            self.rows = rows

        def tolist(self) -> list[list[int]]:
            """Return the rows the way a numpy array does."""
            return self.rows

    class NumpyEncoder:
        def encode(self, sentences: str | Sequence[str], **kwargs: object) -> NumpyLike:
            """Return one integer row per sentence."""
            texts = [sentences] if isinstance(sentences, str) else list(sentences)
            return NumpyLike([[index + 1, 0] for index, _ in enumerate(texts)])

    vectors = SentenceTransformerEmbeddingAdapter(encoder=NumpyEncoder()).embed_texts(["meals"])
    assert vectors == [[1.0, 0.0]]


def test_adapter_satisfies_the_embedder_port(tmp_path) -> None:
    """Ingest through the Embedder port using the local adapter and a fake encoder."""
    from adapter.chroma_store import ChromaPolicyStore

    encoder = FakeEncoder()
    store = ChromaPolicyStore(tmp_path / "chroma")
    written = ingest_policy(
        EXPENSE_POLICY_FIXTURE,
        store=store,
        embedder=SentenceTransformerEmbeddingAdapter(encoder=encoder),
    )
    assert len(written) == store.count() == 6
    assert encoder.calls[0][0].startswith("Employees may claim up to $65 per day")
