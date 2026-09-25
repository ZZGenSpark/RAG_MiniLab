"""Score retrieval recall and answer accuracy for the eight policy questions.

Recall checks section labels. Accuracy checks that every answer marker is present.
CI runs the questions with a fake generator. The CLI writes a live Ollama report.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from adapter.chroma_store import ChromaPolicyStore
from adapter.ollama_chat import OllamaChatAdapter
from adapter.ollama_embeddings import OllamaEmbeddingAdapter
from config import CHROMA_PATH, EVAL_OUTPUT_PATH, EVAL_REPORT_PATH
from rag.ask import ask
from rag.embeddings import Embedder
from rag.generate import Generator
from rag.rerank import Reranker
from rag.route import Router
from rag.schema import REFUSAL_ANSWER, AskResponse, RetrievedChunk
from rag.store import PolicyStore


def answer_matches(case: RequiredCase | RetrievalCase, answer: str) -> bool:
    """Return whether every expected marker appears in the answer."""
    text = answer.lower()
    return all(marker.lower() in text for marker in case.answer_markers)


class RequiredCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    expected_citation: str | None
    answer_markers: list[str]


REQUIRED_CASES = [
    RequiredCase(
        question="How much can I spend on food each day?",
        expected_citation="1. Meals",
        answer_markers=["$65"],
    ),
    RequiredCase(
        question="Can I book first-class airfare?",
        expected_citation="3. Airfare",
        answer_markers=["vice president"],
    ),
    RequiredCase(
        question="My hotel costs $250. What do I need?",
        expected_citation="2. Hotels",
        answer_markers=["manager", "$225"],
    ),
    RequiredCase(
        question="Do I need a receipt for a $20 taxi?",
        expected_citation="5. Receipts",
        answer_markers=["receipt"],
    ),
    RequiredCase(
        question="Can I claim a limousine upgrade?",
        expected_citation="4. Ground Transportation",
        answer_markers=["cannot claim"],
    ),
    RequiredCase(
        question="Does the company reimburse gym memberships?",
        expected_citation=None,
        answer_markers=[REFUSAL_ANSWER],
    ),
]


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
    """Ask each of the eight questions and keep the responses for scoring."""
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
    """Run the eight questions and write the report JSON."""
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


def run_required_questions(store: PolicyStore | None = None) -> list[dict]:
    """Ask each required question and return the raw result rows."""
    policy_store = store or ChromaPolicyStore(CHROMA_PATH)
    embedder = OllamaEmbeddingAdapter()
    generator = OllamaChatAdapter()
    results = []
    for case in REQUIRED_CASES:
        response = ask(
            case.question,
            store=policy_store,
            embedder=embedder,
            generator=generator,
        )
        results.append(
            {
                "question": case.question,
                "expected_citation": case.expected_citation,
                "response": response.model_dump(),
            }
        )
    return results


def write_required_questions(
    path: str | Path = EVAL_OUTPUT_PATH,
    *,
    store: PolicyStore | None = None,
) -> Path:
    """Run the required questions and write the JSON results."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(run_required_questions(store=store), indent=2) + "\n")
    return output_path


def load_required_questions(path: str | Path = EVAL_OUTPUT_PATH) -> list[dict]:
    """Load previously saved required-question results."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def response_from_result(result: dict) -> AskResponse:
    """Rebuild an ask response from one saved result row."""
    return AskResponse.model_validate(result["response"])


def main() -> None:
    """Ask the eight questions with Ollama and write outputs/eval_report.json."""
    from adapter.sentence_transformer_embeddings import SentenceTransformerEmbeddingAdapter

    parser = argparse.ArgumentParser(description="Run the eight evaluation questions and save the report.")
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
