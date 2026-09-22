"""Run the six required policy questions and save their answers.

Compares each answer with the expected citation and required phrases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from adapter.chroma_store import ChromaPolicyStore
from config import EVAL_OUTPUT_PATH
from rag.ask import ask
from rag.schema import REFUSAL_ANSWER, AskResponse
from rag.store import PolicyStore


def answer_matches(case: "RequiredCase", answer: str) -> bool:
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
        answer_markers=["approv"],
    ),
    RequiredCase(
        question="My hotel costs $250. What do I need?",
        expected_citation="2. Hotels",
        answer_markers=["manager", "approv"],
    ),
    RequiredCase(
        question="Do I need a receipt for a $20 taxi?",
        expected_citation="5. Receipts",
        answer_markers=["receipt"],
    ),
    RequiredCase(
        question="Can I claim a limousine upgrade?",
        expected_citation="4. Ground Transportation",
        answer_markers=["cannot"],
    ),
    RequiredCase(
        question="Does the company reimburse gym memberships?",
        expected_citation=None,
        answer_markers=[REFUSAL_ANSWER],
    ),
]


def run_required_questions(store: PolicyStore | None = None) -> list[dict]:
    """Ask each required question and return the raw result rows."""
    results = []
    for case in REQUIRED_CASES:
        response = ask(case.question, store=store)
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
    store = ChromaPolicyStore(args.chroma_path) if args.chroma_path else None
    path = write_required_questions(args.output, store=store)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
