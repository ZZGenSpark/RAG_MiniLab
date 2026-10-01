"""Local sentence-transformers implementation of the embedding port."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, cast

from config import SENTENCE_TRANSFORMER_MODEL


def _rows(encoded: object) -> list[Sequence[Any]]:
    """Turn an encode result into rows. A numpy array uses tolist."""
    tolist = getattr(encoded, "tolist", None)
    if callable(tolist):
        rows = tolist()
    else:
        rows = encoded
    if not isinstance(rows, list):
        raise ValueError("embedding model did not return one vector per chunk")
    return cast(list[Sequence[Any]], rows)


class TextEncoder(Protocol):
    """The encode method SentenceTransformer provides."""

    def encode(self, sentences: str | Sequence[str], **kwargs: object) -> object:
        """Return one vector per input sentence."""
        ...


class SentenceTransformerEmbeddingAdapter:
    """Embed text locally with all-MiniLM-L6-v2 and return plain float vectors.

    Documents and questions use the same encoder. MiniLM has no nomic-style
    prefix. Vectors are L2-normalized so Chroma cosine distance matches the
    model's similarity.
    """

    def __init__(
        self,
        model_name: str | None = None,
        encoder: TextEncoder | None = None,
    ) -> None:
        """Configure the model name and an optional already-built encoder."""
        self.model_name = model_name or SENTENCE_TRANSFORMER_MODEL
        self._encoder = encoder

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one complete embedding vector for each input text."""
        if not texts:
            return []
        return self._embed(list(texts))

    def embed_query(self, text: str) -> list[float]:
        """Return the embedding vector for a single question."""
        return self._embed([text])[0]

    def _encoder_model(self) -> TextEncoder:
        """Load the sentence-transformers model the first time it is needed."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = cast(TextEncoder, SentenceTransformer(self.model_name))
        return self._encoder

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Encode texts and check that each one produced a full vector."""
        encoded = self._encoder_model().encode(texts, normalize_embeddings=True)
        rows = _rows(encoded)
        embeddings = [[float(value) for value in row] for row in rows]
        if len(embeddings) != len(texts):
            raise ValueError("embedding model did not return one vector per chunk")
        if any(len(vector) == 0 for vector in embeddings):
            raise ValueError("embedding must be the complete vector returned by the model")
        return embeddings
