"""Score retrieval recall and answer accuracy for the policy questions.

Recall checks section labels. Accuracy checks that every answer marker is present.
CI runs the questions with a fake generator. The CLI writes a live Ollama report.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from adapter.chroma_store import ChromaPolicyStore
from adapter.ollama_chat import OllamaChatAdapter
from config import CHROMA_PATH, EVAL_REPORT_PATH
from rag.ask import ask
from rag.embeddings import Embedder
from rag.generate import Generator
from rag.rerank import Reranker
from rag.route import Router
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


def main() -> None:
    """Ask the evaluation questions with Ollama and write outputs/eval_report.json."""
    from adapter.sentence_transformer_embeddings import SentenceTransformerEmbeddingAdapter

    parser = argparse.ArgumentParser(description="Run the evaluation questions and save the report.")
    parser.add_argument(
        "--output",
        type=Path,
        default=EVAL_REPORT_PATH,
        help="Path to write the evaluation report.",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=None,
        help="Optional Chroma persistence directory. Defaults to CHROMA_PATH.",
    )
    args = parser.parse_args()
    store = ChromaPolicyStore(args.chroma_path or CHROMA_PATH)
    path = write_eval_report(
        args.output,
        store=store,
        embedder=SentenceTransformerEmbeddingAdapter(),
        generator=OllamaChatAdapter(),
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
