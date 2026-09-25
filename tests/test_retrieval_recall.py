"""Measure retrieval recall on the indexed policies."""

from collections.abc import Sequence

import pytest

from adapter.chroma_store import ChromaPolicyStore
from adapter.sentence_transformer_embeddings import SentenceTransformerEmbeddingAdapter
from rag.eval import RETRIEVAL_CASES, RetrievalCase, retrieval_recall
from rag.ingest import ingest_corpus
from rag.retrieve import retrieve
from rag.route import RetrievalDecision
from rag.schema import RetrievedChunk


def test_retrieval_cases_cover_the_required_shapes() -> None:
    """Keep eight cases, one refusal, Section 7.3, and both foosball versions."""
    assert len(RETRIEVAL_CASES) >= 8
    assert all(case.answer_markers for case in RETRIEVAL_CASES)
    refusals = [case for case in RETRIEVAL_CASES if not case.expected_labels]
    assert len(refusals) == 1
    assert any(
        label.section == "7.3 Weekend Abandonment Consequence"
        for case in RETRIEVAL_CASES
        for label in case.expected_labels
    )
    foosball = next(case for case in RETRIEVAL_CASES if "foosball" in case.question.lower())
    versions = {label.version for label in foosball.expected_labels}
    assert versions == {"1.0", "2.0"}
    assert {label.document for label in foosball.expected_labels} == {"Time & Usage Policy"}


def test_case_recalled_checks_document_version_and_section() -> None:
    """Count a case only when every expected label is in the retrieved chunks."""
    case = RetrievalCase(
        question="How long can an employee play foosball each day?",
        expected_labels=RETRIEVAL_CASES[1].expected_labels,
        answer_markers=["20 minutes"],
    )
    both = [
        _hit("Time & Usage Policy", "1.0", "4", "1", "Daily Allowance"),
        _hit("Time & Usage Policy", "2.0", "4", "1", "Daily Allowance"),
    ]
    only_current = [_hit("Time & Usage Policy", "2.0", "4", "1", "Daily Allowance")]
    assert retrieval_recall([(case, both), (RETRIEVAL_CASES[2], only_current)]) == 1.0
    assert retrieval_recall([(case, only_current)]) == 0.0


@pytest.fixture(scope="module")
def indexed_policies(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[ChromaPolicyStore, SentenceTransformerEmbeddingAdapter]:
    """Ingest the markdown policies with MiniLM. No chat model is involved."""
    store = ChromaPolicyStore(tmp_path_factory.mktemp("recall-chroma"))
    embedder = SentenceTransformerEmbeddingAdapter()
    ingest_corpus(store=store, embedder=embedder)
    return store, embedder


def test_indexed_policies_recall_every_expected_section(
    indexed_policies: tuple[ChromaPolicyStore, SentenceTransformerEmbeddingAdapter],
) -> None:
    """Retrieve each question and require every expected section label to come back."""
    store, embedder = indexed_policies
    pairs: list[tuple[RetrievalCase, Sequence[RetrievedChunk]]] = []
    misses: list[str] = []
    for case in RETRIEVAL_CASES:
        hits = retrieve(
            case.question,
            store=store,
            embedder=embedder,
            decision=RetrievalDecision(strategy="vector"),
        )
        pairs.append((case, hits))
        if not case.expected_labels:
            assert hits
            continue
        missing = [
            f"{label.document} v{label.version} {label.section}"
            for label in case.expected_labels
            if not any(
                hit.document == label.document
                and hit.version == label.version
                and hit.citation_section == label.section
                for hit in hits
            )
        ]
        if missing:
            found = [f"{hit.document} v{hit.version} {hit.citation_section}" for hit in hits]
            misses.append(f"{case.question}\n  missing: {missing}\n  retrieved: {found}")

    assert not misses, "\n".join(misses)
    assert retrieval_recall(pairs) == 1.0


def _hit(document: str, version: str, section: str, minor: str, title: str) -> RetrievedChunk:
    """Build a retrieved chunk whose citation label is `section.minor title` or `section. title`."""
    number = f"{section}.{minor}" if minor else section
    slug = document.lower().replace("&", "and").replace(" ", "-")
    return RetrievedChunk(
        chunk_id=f"{slug}:v{version}:section-{number}",
        document=document,
        version=version,
        section=number,
        section_title=title,
        text="Foosball is capped at 20 minutes per employee per day.",
        distance=0.1,
    )
