from __future__ import annotations

from rag.config import DEFAULT_CHROMA_PATH
from rag.embeddings import Embedder
from rag.generate import Generator, generate_answer
from rag.retrieve import retrieve
from rag.schema import AskResponse
from rag.store import PolicyStore


def ask(
    question: str,
    *,
    store: PolicyStore | None = None,
    embedder: Embedder | None = None,
    generator: Generator | None = None,
) -> AskResponse:
    store = store or PolicyStore(DEFAULT_CHROMA_PATH)
    hits = retrieve(question, store=store, embedder=embedder)
    answer, citation = generate_answer(question, hits, generator=generator)
    return AskResponse(
        answer=answer,
        citation=citation,
        retrieved_chunks=[hit.to_ref() for hit in hits],
    )
