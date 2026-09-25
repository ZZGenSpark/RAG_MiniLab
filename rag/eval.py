"""Score retrieval recall and answer accuracy for the policy questions.

Recall checks section labels. Accuracy checks that every answer marker is present.
`--retrieval-only` scores recall with MiniLM and the cross-encoder. The default CLI
asks Ollama and fails unless both scores are 1.0.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from adapter.chroma_store import ChromaPolicyStore
from config import CHROMA_PATH, EVAL_REPORT_PATH, RETRIEVAL_EVAL_PATH
from rag.ask import ask
from rag.embeddings import Embedder
from rag.generate import Generator
from rag.rerank import Reranker
from rag.retrieve import retrieve
from rag.route import FallbackRouter, Router
from rag.schema import REFUSAL_ANSWER, AskResponse, RetrievedChunk
from rag.store import PolicyStore


class ExpectedLabel(BaseModel):
    """One document version and section label that retrieval should return."""

    model_config = ConfigDict(extra="forbid")

    document: str
    version: str
    section: str


class RetrievalCase(BaseModel):
    """A question, the sections recall must find, and the answer markers saved for later."""

    model_config = ConfigDict(extra="forbid")

    question: str
    expected_labels: list[ExpectedLabel]
    answer_markers: list[str]


def answer_matches(case: RetrievalCase, answer: str) -> bool:
    """Return whether every expected marker appears in the answer."""
    text = answer.lower()
    return all(marker.lower() in text for marker in case.answer_markers)


RETRIEVAL_CASES = [
    RetrievalCase(
        question="What happens to food left in the shared refrigerator over the weekend?",
        expected_labels=[
            ExpectedLabel(
                document="HR Policy",
                version="2.0",
                section="7.3 Weekend Abandonment Consequence",
            )
        ],
        answer_markers=["abandoned", "spoonful"],
    ),
    RetrievalCase(
        question="How long can an employee play foosball each day?",
        expected_labels=[
            ExpectedLabel(document="Time & Usage Policy", version="1.0", section="4.1 Daily Allowance"),
            ExpectedLabel(document="Time & Usage Policy", version="2.0", section="4.1 Daily Allowance"),
        ],
        answer_markers=["20 minutes"],
    ),
    RetrievalCase(
        question="Does the company match retirement contributions?",
        expected_labels=[],
        answer_markers=[REFUSAL_ANSWER],
    ),
    RetrievalCase(
        question="Does every company email have to include a joke?",
        expected_labels=[
            ExpectedLabel(document="HR Policy", version="2.0", section="3.1 Requirement"),
        ],
        answer_markers=["joke"],
    ),
    RetrievalCase(
        question="How long must I wait before correcting a boss who is wrong?",
        expected_labels=[
            ExpectedLabel(document="HR Policy", version="2.0", section="6. Boss Error Grace Period"),
        ],
        answer_markers=["30 minutes"],
    ),
    RetrievalCase(
        question="How often are employees expected to work out?",
        expected_labels=[
            ExpectedLabel(document="Health & Wellness Policy", version="1.0", section="3.1 Minimum Requirement"),
        ],
        answer_markers=["three", "45 minutes"],
    ),
    RetrievalCase(
        question="How long do employees stay indoors after a nuclear event?",
        expected_labels=[
            ExpectedLabel(document="Preparedness Policy", version="2.0", section="4.3 Duration of Sheltering"),
        ],
        answer_markers=["two weeks"],
    ),
    RetrievalCase(
        question="How much paid time off do I get when I adopt a pet?",
        expected_labels=[
            ExpectedLabel(document="HR Policy", version="2.0", section="5.1 Leave Entitlement"),
        ],
        answer_markers=["5 days", "dog"],
    ),
    RetrievalCase(
        question="What does Section 7.3 say happens to food left in the refrigerator over the weekend?",
        expected_labels=[
            ExpectedLabel(
                document="HR Policy",
                version="2.0",
                section="7.3 Weekend Abandonment Consequence",
            )
        ],
        answer_markers=["abandoned", "spoonful"],
    ),
    RetrievalCase(
        question="What does Section 4.3 require employees to do after a nuclear event?",
        expected_labels=[
            ExpectedLabel(document="Preparedness Policy", version="2.0", section="4.3 Duration of Sheltering"),
        ],
        answer_markers=["two weeks"],
    ),
    RetrievalCase(
        question="Under Section 4.2, what does the foosball winner receive?",
        expected_labels=[
            ExpectedLabel(document="Time & Usage Policy", version="2.0", section="4.2 Winner-Takes-Tokens Rule"),
        ],
        answer_markers=["token"],
    ),
    RetrievalCase(
        question="What daily caffeine limit does Section 5.1 set?",
        expected_labels=[
            ExpectedLabel(document="Health & Wellness Policy", version="1.0", section="5.1 Daily Limit"),
        ],
        answer_markers=["400 mg"],
    ),
]


def label_was_retrieved(label: ExpectedLabel, chunks: Sequence[RetrievedChunk]) -> bool:
    """Return whether retrieval returned this document version and section label."""
    return any(
        chunk.document == label.document and chunk.version == label.version and chunk.citation_section == label.section
        for chunk in chunks
    )


def case_recalled(case: RetrievalCase, chunks: Sequence[RetrievedChunk]) -> bool:
    """Return whether every expected section label was retrieved.

    A refusal has no expected label, so it is not a recall miss.
    """
    return all(label_was_retrieved(label, chunks) for label in case.expected_labels)


def retrieval_recall(pairs: Sequence[tuple[RetrievalCase, Sequence[RetrievedChunk]]]) -> float:
    """Return the fraction of labeled cases whose expected sections were all retrieved."""
    labeled = [(case, chunks) for case, chunks in pairs if case.expected_labels]
    if not labeled:
        return 0.0
    hits = sum(1 for case, chunks in labeled if case_recalled(case, chunks))
    return hits / len(labeled)


def answer_accuracy(pairs: Sequence[tuple[RetrievalCase, str]]) -> float:
    """Return the fraction of cases whose answers contain every marker."""
    if not pairs:
        return 0.0
    hits = sum(1 for case, answer in pairs if answer_matches(case, answer))
    return hits / len(pairs)


class EvalResult(BaseModel):
    """One saved ask for an evaluation question."""

    model_config = ConfigDict(extra="forbid")

    question: str
    response: AskResponse


class EvalReport(BaseModel):
    """Saved answers that can be scored without calling a model again."""

    model_config = ConfigDict(extra="forbid")

    results: list[EvalResult]


class EvalScores(BaseModel):
    """Recall and accuracy computed from a saved report."""

    model_config = ConfigDict(extra="forbid")

    recall: float
    accuracy: float


def response_recalls(case: RetrievalCase, response: AskResponse) -> bool:
    """Return whether the saved chunks include every expected document version and section."""
    return all(
        any(
            chunk.document == label.document and chunk.version == label.version and chunk.section == label.section
            for chunk in response.retrieved_chunks
        )
        for label in case.expected_labels
    )


def live_eval_passed(scores: EvalScores) -> bool:
    """Return whether live recall and answer accuracy both meet the gate."""
    return scores.recall == 1.0 and scores.accuracy == 1.0


def score_eval_report(report: EvalReport) -> EvalScores:
    """Score recall and accuracy from saved answers. This does not call a model."""
    pairs = [(case_for_question(result.question), result.response) for result in report.results]
    labeled = [(case, response) for case, response in pairs if case.expected_labels]
    hits = sum(1 for case, response in labeled if response_recalls(case, response))
    recall = 0.0 if not labeled else hits / len(labeled)
    accuracy = answer_accuracy([(case, response.answer) for case, response in pairs])
    return EvalScores(recall=recall, accuracy=accuracy)


def case_for_question(question: str) -> RetrievalCase:
    """Return the evaluation case for one saved question."""
    for case in RETRIEVAL_CASES:
        if case.question == question:
            return case
    raise KeyError(question)


def run_eval(
    *,
    store: PolicyStore,
    embedder: Embedder,
    generator: Generator,
    router: Router | None = None,
    reranker: Reranker | None = None,
    audit_path: Path | None = None,
) -> EvalReport:
    """Ask each evaluation question and keep the responses for scoring."""
    results = []
    for case in RETRIEVAL_CASES:
        response = ask(
            case.question,
            store=store,
            embedder=embedder,
            generator=generator,
            router=router,
            reranker=reranker,
            audit_path=audit_path,
        )
        results.append(EvalResult(question=case.question, response=response))
    return EvalReport(results=results)


def write_eval_report(
    path: str | Path = EVAL_REPORT_PATH,
    *,
    store: PolicyStore,
    embedder: Embedder,
    generator: Generator,
    router: Router | None = None,
    reranker: Reranker | None = None,
    audit_path: Path | None = None,
) -> Path:
    """Run the evaluation questions and write the report JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = run_eval(
        store=store,
        embedder=embedder,
        generator=generator,
        router=router,
        reranker=reranker,
        audit_path=audit_path,
    )
    output_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return output_path


def load_eval_report(path: str | Path = EVAL_REPORT_PATH) -> EvalReport:
    """Load a saved evaluation report."""
    return EvalReport.model_validate_json(Path(path).read_text(encoding="utf-8"))


class RetrievalEvalResult(BaseModel):
    """One question after retrieval, with the section labels that were missing."""

    model_config = ConfigDict(extra="forbid")

    question: str
    recalled: bool
    missing: list[str]
    retrieved: list[str]


class RetrievalEvalReport(BaseModel):
    """Recall for the evaluation questions. This report has no generated answers."""

    model_config = ConfigDict(extra="forbid")

    recall: float
    results: list[RetrievalEvalResult]


def evaluate_retrieval(
    *,
    store: PolicyStore,
    embedder: Embedder,
    router: Router | None = None,
) -> RetrievalEvalReport:
    """Retrieve every evaluation question and score section recall.

    The router defaults to the section-code fallback, so this does not call Jev.
    Retrieval uses the local cross-encoder. No chat model is called.
    """
    chosen = router if router is not None else FallbackRouter()
    pairs: list[tuple[RetrievalCase, Sequence[RetrievedChunk]]] = []
    results: list[RetrievalEvalResult] = []
    for case in RETRIEVAL_CASES:
        hits = retrieve(
            case.question,
            store=store,
            embedder=embedder,
            decision=chosen.choose(case.question),
        )
        pairs.append((case, hits))
        missing = [
            f"{label.document} v{label.version} {label.section}"
            for label in case.expected_labels
            if not label_was_retrieved(label, hits)
        ]
        results.append(
            RetrievalEvalResult(
                question=case.question,
                recalled=case_recalled(case, hits),
                missing=missing,
                retrieved=[f"{hit.document} v{hit.version} {hit.citation_section}" for hit in hits],
            )
        )
    return RetrievalEvalReport(recall=retrieval_recall(pairs), results=results)


def retrieval_eval_passed(report: RetrievalEvalReport) -> bool:
    """Return whether every question retrieved chunks and every labeled section was found."""
    return report.recall == 1.0 and all(result.retrieved for result in report.results)


def write_retrieval_eval(path: str | Path, report: RetrievalEvalReport) -> Path:
    """Write a retrieval-only report. This file does not contain generated answers."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return output_path


def _print_retrieval_eval(report: RetrievalEvalReport) -> None:
    """Print recall and every question that missed a section or retrieved nothing."""
    print(f"retrieval recall: {report.recall:.3f}")
    for result in report.results:
        if result.recalled and result.retrieved:
            continue
        print(result.question)
        print(f"  missing: {result.missing}")
        print(f"  retrieved: {result.retrieved}")


def _run_retrieval_eval(output: Path, chroma_path: Path | None) -> None:
    """Ingest the policies, score recall, and stop when a required section is missing."""
    import tempfile

    from adapter.sentence_transformer_embeddings import SentenceTransformerEmbeddingAdapter
    from rag.ingest import ingest_corpus

    def _score(path: Path) -> None:
        store = ChromaPolicyStore(path)
        embedder = SentenceTransformerEmbeddingAdapter()
        ingest_corpus(store=store, embedder=embedder)
        report = evaluate_retrieval(store=store, embedder=embedder)
        written = write_retrieval_eval(output, report)
        _print_retrieval_eval(report)
        print(f"Wrote {written}")
        if not retrieval_eval_passed(report):
            raise SystemExit(1)

    if chroma_path is None:
        with tempfile.TemporaryDirectory(prefix="retrieval-eval-") as directory:
            _score(Path(directory))
        return
    _score(chroma_path)


def main() -> None:
    """Score retrieval, or ask the questions with Ollama and write outputs/eval_report.json."""
    parser = argparse.ArgumentParser(description="Run the evaluation questions and save the report.")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Score retrieval recall with MiniLM and the cross-encoder. Does not call Ollama.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write the report. Defaults to the retrieval or Ollama report path.",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=None,
        help="Optional Chroma persistence directory. Retrieval-only defaults to a temporary directory.",
    )
    args = parser.parse_args()
    if args.retrieval_only:
        _run_retrieval_eval(args.output or RETRIEVAL_EVAL_PATH, args.chroma_path)
        return

    from adapter.ollama_chat import OllamaChatAdapter
    from adapter.sentence_transformer_embeddings import SentenceTransformerEmbeddingAdapter

    store = ChromaPolicyStore(args.chroma_path or CHROMA_PATH)
    path = write_eval_report(
        args.output or EVAL_REPORT_PATH,
        store=store,
        embedder=SentenceTransformerEmbeddingAdapter(),
        generator=OllamaChatAdapter(),
    )
    scores = score_eval_report(load_eval_report(path))
    print(f"Wrote {path}")
    print(f"recall: {scores.recall:.3f}")
    print(f"accuracy: {scores.accuracy:.3f}")
    if not live_eval_passed(scores):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
