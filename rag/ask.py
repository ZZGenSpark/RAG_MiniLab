"""Answer a question from the ingested policies.

Routes the question, retrieves excerpts, and returns a grounded answer with a citation.
"""

from __future__ import annotations

from pathlib import Path

from config import TOP_K
from rag.audit import AuditRecord, append_audit, source_conflicts
from rag.embeddings import Embedder
from rag.generate import Generator, generate_answer
from rag.rerank import Reranker
from rag.retrieve import retrieve_outcome
from rag.route import Router, route
from rag.schema import AskResponse, RetrievalInfo
from rag.store import PolicyStore


def ask(
    question: str,
    *,
    store: PolicyStore,
    embedder: Embedder,
    generator: Generator,
    router: Router | None = None,
    reranker: Reranker | None = None,
    audit_path: Path | None = None,
) -> AskResponse:
    """Choose vector or hybrid, rerank the shortlist, and return a cited answer or a refusal."""
    decision = route(question, router=router)
    outcome = retrieve_outcome(question, store=store, embedder=embedder, decision=decision, reranker=reranker)
    shown = outcome.chunks[:TOP_K]
    answer, citations = generate_answer(question, shown, generator=generator)
    conflicts = source_conflicts(shown)
    append_audit(
        AuditRecord(
            question=question,
            strategy=decision.strategy,
            vector_ids=outcome.vector_ids,
            bm25_ids=outcome.bm25_ids,
            rrf_ids=outcome.rrf_ids,
            cross_encoder_scores=outcome.cross_encoder_scores,
            source_conflicts=conflicts,
        ),
        audit_path,
    )
    cited = sorted(shown, key=lambda hit: (hit.distance, hit.chunk_id))
    return AskResponse(
        answer=answer,
        citations=citations,
        retrieved_chunks=[hit.to_ref() for hit in cited],
        retrieval=RetrievalInfo(strategy=decision.strategy),
        source_conflicts=conflicts,
    )
