"""Run policy questions and score retrieval recall.

Answer markers are recorded on each case. Recall checks section labels only.
Answer-marker checks are added once generation exists.
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
from config import CHROMA_PATH, EVAL_OUTPUT_PATH
from rag.ask import ask
from rag.schema import REFUSAL_ANSWER, AskResponse, RetrievedChunk
from rag.store import PolicyStore


def answer_matches(case: RequiredCase, answer: str) -> bool:
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
    """Parse CLI arguments and write the required-question results."""
    parser = argparse.ArgumentParser(description="Run the six required questions and save JSON output.")
    parser.add_argument(
        "--output",
        type=Path,
        default=EVAL_OUTPUT_PATH,
        help="Path to write the saved results.",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=None,
        help="Optional Chroma persistence directory. Defaults to CHROMA_PATH.",
    )
    args = parser.parse_args()
    store = ChromaPolicyStore(args.chroma_path or CHROMA_PATH)
    path = write_required_questions(args.output, store=store)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
