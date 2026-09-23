"""Ollama implementation of the embedding port."""

from __future__ import annotations

from typing import Protocol, Sequence

import ollama

from config import EMBED_MODEL, OLLAMA_HOST

DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class EmbeddingClient(Protocol):
    def embed(self, model: str, input: str | Sequence[str]):
        """Embed one string or a batch and return the client response."""
        ...


class OllamaEmbeddingAdapter:
    """Embed text with an Ollama model and return plain float vectors.

    nomic-embed-text expects a document prefix on stored chunks and a query
    prefix on questions. The original section text stays in the vector store.
    """

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        client: EmbeddingClient | None = None,
    ) -> None:
        """Configure the embedding model, host, and client."""
        self.model = model or EMBED_MODEL
        self.client = client or ollama.Client(host=host or OLLAMA_HOST)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one complete embedding vector for each input text."""
        if not texts:
            return []
        prefixed = [f"{DOCUMENT_PREFIX}{text}" for text in texts]
        return self._embed(prefixed)

    def embed_query(self, text: str) -> list[float]:
        """Return the embedding vector for a single question."""
        vectors = self._embed([f"{QUERY_PREFIX}{text}"])
        return vectors[0]

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed already prefixed strings and check the returned vectors."""
        response = self.client.embed(model=self.model, input=list(texts))
        embeddings = [[float(value) for value in vector] for vector in response.embeddings]
        if len(embeddings) != len(texts):
            raise ValueError("embedding model did not return one vector per chunk")
        if any(len(vector) == 0 for vector in embeddings):
            raise ValueError("embedding must be the complete vector returned by the model")
        return embeddings
