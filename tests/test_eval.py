"""Check scoring helpers for saved evaluation results."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from adapter.chroma_store import ChromaPolicyStore
from adapter.sentence_transformer_embeddings import SentenceTransformerEmbeddingAdapter
from rag.eval import (
    RETRIEVAL_CASES,
    EvalReport,
    EvalResult,
    ExpectedLabel,
    RetrievalCase,
    answer_accuracy,
    answer_matches,
    load_eval_report,
    score_eval_report,
    write_eval_report,
)
from rag.ingest import ingest_corpus
from rag.route import FallbackRouter
from rag.schema import REFUSAL_ANSWER, AskResponse, RetrievedChunkRef


def test_answer_matches_is_case_insensitive_and_requires_every_marker() -> None:
    """Accept an answer only when every marker appears, ignoring case."""
    case = RetrievalCase(
        question="How much can I spend on food each day?",
        expected_labels=[],
        answer_markers=["$65", "Alcohol"],
    )
    assert answer_matches(case, "Alcohol is not reimbursable. The limit is $65.")
    assert not answer_matches(case, "The limit is $65.")


def test_refusal_marker_matches_the_canonical_refusal() -> None:
    """Treat the canonical refusal sentence as the unsupported-question marker."""
    case = RetrievalCase(
        question="Does the company reimburse gym memberships?",
        expected_labels=[],
        answer_markers=[REFUSAL_ANSWER],
    )
    assert answer_matches(case, REFUSAL_ANSWER)
    assert not answer_matches(case, "Gym memberships are reimbursable.")


def test_answer_accuracy_requires_every_marker() -> None:
    """Count a case only when the answer contains every marker."""
    weekend = RETRIEVAL_CASES[0]
    refusal = next(case for case in RETRIEVAL_CASES if not case.expected_labels)
    assert answer_accuracy([(weekend, "The food is abandoned. Eat a spoonful.")]) == 1.0
    assert answer_accuracy([(weekend, "The food is abandoned.")]) == 0.0
    assert answer_accuracy([(weekend, "The food is abandoned."), (refusal, REFUSAL_ANSWER)]) == 0.5


def test_score_eval_report_counts_a_missing_version_and_a_missing_marker() -> None:
    """Score both metrics from a saved report without calling a generator."""
    foosball = next(case for case in RETRIEVAL_CASES if "foosball" in case.question.lower())
    joke = next(case for case in RETRIEVAL_CASES if "joke" in case.question.lower())
    report = EvalReport(
        results=[
            EvalResult(
                question=foosball.question,
                response=_saved("Foosball is capped at 20 minutes.", foosball.expected_labels[:1]),
            ),
            EvalResult(
                question=joke.question,
                response=_saved("Emails are required.", joke.expected_labels),
            ),
        ]
    )
    scores = score_eval_report(report)
    assert scores.recall == 0.5
    assert scores.accuracy == 0.5


@pytest.fixture(scope="module")
def indexed_policies(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[ChromaPolicyStore, SentenceTransformerEmbeddingAdapter]:
    """Ingest the markdown policies with MiniLM."""
    store = ChromaPolicyStore(tmp_path_factory.mktemp("eval-chroma"))
    embedder = SentenceTransformerEmbeddingAdapter()
    ingest_corpus(store=store, embedder=embedder)
    return store, embedder


def test_saved_report_scores_recall_and_accuracy(
    indexed_policies: tuple[ChromaPolicyStore, SentenceTransformerEmbeddingAdapter],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write every answer with a fake generator, then score the saved report."""

    def fail_if_jev_is_called(self: object, question: str) -> None:
        """Fail the test if the live router is constructed."""
        raise AssertionError("Jev")

    monkeypatch.setattr("adapter.jev_router.JevRouter.choose", fail_if_jev_is_called)
    store, embedder = indexed_policies
    generator = MarkerGenerator()
    path = write_eval_report(
        tmp_path / "eval_report.json",
        store=store,
        embedder=embedder,
        generator=generator,
        router=FallbackRouter(),
        audit_path=tmp_path / "audit.jsonl",
    )
    calls_before_scoring = generator.calls
    scores = score_eval_report(load_eval_report(path))
    assert generator.calls == calls_before_scoring
    assert [result.question for result in load_eval_report(path).results] == [case.question for case in RETRIEVAL_CASES]
    assert scores.recall == 1.0
    assert scores.accuracy == 1.0


class MarkerGenerator:
    """Return each case's markers and cite every excerpt in the prompt."""

    def __init__(self) -> None:
        """Start with no recorded prompts."""
        self.calls = 0

    def complete(self, prompt: str) -> str:
        """Answer with the markers for the question in this prompt."""
        self.calls += 1
        question = prompt.split("Question: ", 1)[1].split("\n", 1)[0]
        case = next(case for case in RETRIEVAL_CASES if case.question == question)
        if not case.expected_labels:
            return json.dumps({"answerable": False, "answer": "", "sources": []})
        excerpt_count = len([line for line in prompt.splitlines() if line.startswith("Excerpt ")])
        return json.dumps(
            {
                "answerable": True,
                "answer": " ".join(case.answer_markers),
                "sources": list(range(1, excerpt_count + 1)),
            }
        )


def _saved(answer: str, labels: Sequence[ExpectedLabel]) -> AskResponse:
    """Build a saved response whose retrieved chunks are the given labels."""
    return AskResponse(
        answer=answer,
        citations=[],
        retrieved_chunks=[
            RetrievedChunkRef(
                section=label.section,
                distance=0.1 + index,
                document=label.document,
                version=label.version,
            )
            for index, label in enumerate(labels)
        ],
    )
