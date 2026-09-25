"""Answer a question from the ingested policies.

Routes the question, retrieves excerpts, and returns a grounded answer with a citation.
"""

from __future__ import annotations

from config import TOP_K
from rag.embeddings import Embedder
from rag.generate import Generator, generate_answer
from rag.retrieve import retrieve
from rag.route import Router, route
from rag.schema import AskResponse
from rag.store import PolicyStore


def ask(
    question: str,
    *,
    store: PolicyStore,
    embedder: Embedder,
    generator: Generator,
    router: Router | None = None,
) -> AskResponse:
    """Choose vector or hybrid, retrieve excerpts, and return a cited answer or a refusal."""
    decision = route(question, router=router)
    hits = retrieve(question, store=store, embedder=embedder, decision=decision)
    shown = hits[:TOP_K]
    answer, citation = generate_answer(question, shown, generator=generator)
    cited = sorted(shown, key=lambda hit: (hit.distance, hit.chunk_id))
    return AskResponse(
        answer=answer,
        citation=citation,
        retrieved_chunks=[hit.to_ref() for hit in cited],
    )
