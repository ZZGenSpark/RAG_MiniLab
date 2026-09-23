"""Check the six required questions against saved and live answers."""

import pytest

from adapter.chroma_store import ChromaPolicyStore
from adapter.ollama_chat import OllamaChatAdapter
from adapter.ollama_embeddings import OllamaEmbeddingAdapter
from config import EVAL_OUTPUT_PATH, POLICY_PATH
from rag.ask import ask
from rag.chunking import chunk_policy_file
from rag.eval import (
    REQUIRED_CASES,
    answer_matches,
    load_required_questions,
    response_from_result,
)
from rag.ingest import ingest_policy
from rag.schema import REFUSAL_ANSWER, AskResponse
from tests.support import ollama_connection_error


def _assert_response_shape(response: AskResponse) -> None:
    """Require one to three retrieved chunks sorted by distance."""
    assert 0 < len(response.retrieved_chunks) <= 3
    distances = [chunk.distance for chunk in response.retrieved_chunks]
    assert distances == sorted(distances)
    assert all(isinstance(distance, float) for distance in distances)


def _score_required_run(results: list[tuple]) -> dict[str, int]:
    """Count retrieval hits, citations, and refusals for one required-question run."""
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
    """Confirm the policy file still splits into six sections."""
    assert len(chunk_policy_file(POLICY_PATH)) == 6


def test_saved_output_covers_all_six_required_questions() -> None:
    """Confirm the saved JSON covers every required question at the expected scores."""
    assert EVAL_OUTPUT_PATH.exists(), "run `python -m rag.eval` to create outputs/required_questions.json"
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
def live_store(tmp_path_factory: pytest.TempPathFactory) -> ChromaPolicyStore:
    """Ingest the policy into a temporary store, or skip if Ollama cannot be reached."""
    chroma_path = tmp_path_factory.mktemp("required-chroma")
    store = ChromaPolicyStore(chroma_path)
    try:
        ingest_policy(POLICY_PATH, store=store, embedder=OllamaEmbeddingAdapter())
    except Exception as exc:
        if ollama_connection_error(exc):
            pytest.skip(f"live Ollama ingest unavailable: {exc}")
        raise
    return store


def test_live_required_questions_meet_acceptance(live_store: ChromaPolicyStore) -> None:
    """Confirm a live run meets the retrieval, citation, and refusal thresholds."""
    embedder = OllamaEmbeddingAdapter()
    generator = OllamaChatAdapter()
    results = [
        (case, ask(case.question, store=live_store, embedder=embedder, generator=generator))
        for case in REQUIRED_CASES
    ]

    scored = _score_required_run(results)
    assert scored["retrieve_hits"] >= 5
    assert scored["supported_with_citation"] == 5
    assert scored["expected_citations"] == 5
    assert scored["refusals"] == 1
