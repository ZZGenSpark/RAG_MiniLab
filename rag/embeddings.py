from __future__ import annotations

from typing import Protocol, Sequence

import ollama

from rag.config import DEFAULT_EMBED_MODEL, DEFAULT_OLLAMA_HOST


class EmbeddingClient(Protocol):
    def embed(self, model: str, input: str | Sequence[str]): ...


class Embedder:
    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        client: EmbeddingClient | None = None,
    ) -> None:
        self.model = model or DEFAULT_EMBED_MODEL
        self.client = client or ollama.Client(host=host or DEFAULT_OLLAMA_HOST)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embed(model=self.model, input=list(texts))
        embeddings = [[float(value) for value in vector] for vector in response.embeddings]
        if len(embeddings) != len(texts):
            raise ValueError("embedding model did not return one vector per chunk")
        if any(len(vector) == 0 for vector in embeddings):
            raise ValueError("embedding must be the complete vector returned by the model")
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_texts([text])
        return vectors[0]
