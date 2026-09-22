"""Answer a question from the ingested expense policy.

Retrieves the closest chunks and returns a grounded answer with a citation.
"""

from __future__ import annotations

from adapter.chroma_store import ChromaPolicyStore
from config import CHROMA_PATH
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
    """Retrieve policy excerpts and return a cited answer or a refusal."""
    store = store or ChromaPolicyStore(CHROMA_PATH)
    hits = retrieve(question, store=store, embedder=embedder)
    answer, citation = generate_answer(question, hits, generator=generator)
    return AskResponse(
        answer=answer,
        citation=citation,
        retrieved_chunks=[hit.to_ref() for hit in hits],
    )
