from pathlib import Path

import pytest

from rag.ask import ask
from rag.chunking import chunk_policy_file
from rag.eval import (
    DEFAULT_OUTPUT_PATH,
    REQUIRED_CASES,
    answer_matches,
    load_required_questions,
    response_from_result,
)
from rag.ingest import ingest_policy
from rag.schema import REFUSAL_ANSWER, AskResponse
from rag.store import PolicyStore

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "policy.md"


def _assert_response_shape(response: AskResponse) -> None:
    assert 0 < len(response.retrieved_chunks) <= 3
    distances = [chunk.distance for chunk in response.retrieved_chunks]
    assert distances == sorted(distances)
    assert all(isinstance(distance, float) for distance in distances)


def _score_required_run(results: list[tuple]) -> dict[str, int]:
    retrieve_hits = 0
    expected_citations = 0
    supported_with_citation = 0
    refusals = 0

    for case, response in results:
        _assert_response_shape(response)
        retrieved = {chunk.section for chunk in response.retrieved_chunks}

        if case.expected_citation is None:
            assert response.citation is None
            assert response.answer == REFUSAL_ANSWER
            refusals += 1
            continue

        if case.expected_citation in retrieved:
            retrieve_hits += 1
        assert response.citation is not None
        assert response.citation.document == "Employee Expense Policy"
        assert response.citation.version == "2.0"
        assert response.citation.section in retrieved
        supported_with_citation += 1
        if response.citation.section == case.expected_citation:
            expected_citations += 1
            assert answer_matches(case, response.answer)

    return {
        "retrieve_hits": retrieve_hits,
        "expected_citations": expected_citations,
        "supported_with_citation": supported_with_citation,
        "refusals": refusals,
    }


def test_policy_still_has_exactly_six_chunks() -> None:
    assert len(chunk_policy_file(POLICY_PATH)) == 6


def test_saved_output_covers_all_six_required_questions() -> None:
    assert DEFAULT_OUTPUT_PATH.exists(), "run `python -m rag.eval` to create outputs/required_questions.json"
    rows = load_required_questions()
    assert [row["question"] for row in rows] == [case.question for case in REQUIRED_CASES]

    scored = _score_required_run(
        [(case, response_from_result(row)) for case, row in zip(REQUIRED_CASES, rows, strict=True)]
    )
    assert scored["retrieve_hits"] >= 5
    assert scored["supported_with_citation"] == 5
    assert scored["expected_citations"] == 5
    assert scored["refusals"] == 1


@pytest.fixture(scope="module")
def live_store(tmp_path_factory: pytest.TempPathFactory) -> PolicyStore:
    chroma_path = tmp_path_factory.mktemp("required-chroma")
    try:
        ingest_policy(POLICY_PATH, chroma_path=chroma_path)
    except Exception as exc:
        pytest.skip(f"live Ollama ingest unavailable: {exc}")
    return PolicyStore(chroma_path)


def test_live_required_questions_meet_acceptance(live_store: PolicyStore) -> None:
    results = []
    for case in REQUIRED_CASES:
        try:
            results.append((case, ask(case.question, store=live_store)))
        except Exception as exc:
            pytest.skip(f"live ask unavailable: {exc}")

    scored = _score_required_run(results)
    assert scored["retrieve_hits"] >= 5
    assert scored["supported_with_citation"] == 5
    assert scored["expected_citations"] == 5
    assert scored["refusals"] == 1
