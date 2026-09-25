"""Check scoring helpers for saved required-question results."""

import json
from pathlib import Path

from rag.eval import (
    RequiredCase,
    answer_matches,
    load_required_questions,
    response_from_result,
    write_required_questions,
)
from rag.schema import REFUSAL_ANSWER


def test_answer_matches_is_case_insensitive_and_requires_every_marker() -> None:
    """Accept an answer only when every marker appears, ignoring case."""
    case = RequiredCase(
        question="How much can I spend on food each day?",
        expected_citation="1. Meals",
        answer_markers=["$65", "Alcohol"],
    )
    assert answer_matches(case, "Alcohol is not reimbursable. The limit is $65.")
    assert not answer_matches(case, "The limit is $65.")


def test_refusal_marker_matches_the_canonical_refusal() -> None:
    """Treat the canonical refusal sentence as the unsupported-question marker."""
    case = RequiredCase(
        question="Does the company reimburse gym memberships?",
        expected_citation=None,
        answer_markers=[REFUSAL_ANSWER],
    )
    assert answer_matches(case, REFUSAL_ANSWER)
    assert not answer_matches(case, "Gym memberships are reimbursable.")


def test_write_required_questions_saves_rows_as_json(tmp_path: Path, monkeypatch) -> None:
    """Create the output directory and write the rows returned by a run."""
    rows = [{"question": "How much can I spend on food each day?", "response": {"answer": "$65"}}]
    monkeypatch.setattr("rag.eval.run_required_questions", lambda store=None: rows)
    path = tmp_path / "nested" / "results.json"

    written = write_required_questions(path)

    assert written == path
    assert json.loads(path.read_text(encoding="utf-8")) == rows


def test_load_required_questions_reads_the_given_file(tmp_path: Path) -> None:
    """Load result rows from the path that was passed in."""
    path = tmp_path / "results.json"
    rows = [{"question": "How much can I spend on food each day?"}]
    path.write_text(json.dumps(rows), encoding="utf-8")
    assert load_required_questions(path) == rows


def test_response_from_result_rebuilds_the_saved_ask_response() -> None:
    """Rebuild an ask response from one saved result row."""
    result = {
        "question": "How much can I spend on food each day?",
        "response": {
            "answer": "Every email must include a joke.",
            "citations": [
                {
                    "document": "HR Policy",
                    "version": "2.0",
                    "section": "3.1 Requirement",
                }
            ],
            "retrieved_chunks": [{"section": "3.1 Requirement", "distance": 0.08}],
        },
    }
    response = response_from_result(result)
    assert response.answer == "Every email must include a joke."
    assert [citation.section for citation in response.citations] == ["3.1 Requirement"]
    assert response.retrieved_chunks[0].distance == 0.08
