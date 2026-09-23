"""Check schema validation for chunk ids, vectors, and ask responses."""

import pytest
from pydantic import ValidationError

from rag.schema import AskResponse, Citation, EmbeddedChunk, PolicyChunk, RetrievedChunkRef


def test_policy_chunk_rejects_unstable_chunk_id() -> None:
    """Reject a chunk id that does not encode version and section."""
    with pytest.raises(ValidationError, match="chunk_id"):
        PolicyChunk(
            chunk_id="meals-1",
            document="Employee Expense Policy",
            version="2.0",
            section="1",
            section_title="Meals",
            text="Employees may claim up to $65 per day.",
        )


def test_embedded_chunk_requires_a_complete_vector() -> None:
    """Reject an embedded chunk whose vector is empty."""
    with pytest.raises(ValidationError):
        EmbeddedChunk(
            chunk_id="expense-policy:v2.0:section-1",
            document="Employee Expense Policy",
            version="2.0",
            section="1",
            section_title="Meals",
            text="Employees may claim up to $65 per day.",
            embedding=[],
        )


def test_ask_response_matches_assignment_shape() -> None:
    """Accept an ask response with a citation and one retrieved chunk."""
    response = AskResponse.model_validate(
        {
            "answer": "Employees may claim up to $65 per day for meals.",
            "citation": {
                "document": "Employee Expense Policy",
                "version": "2.0",
                "section": "1. Meals",
            },
            "retrieved_chunks": [{"section": "1. Meals", "distance": 0.08}],
        }
    )
    assert response.citation == Citation(
        document="Employee Expense Policy",
        version="2.0",
        section="1. Meals",
    )
    assert response.retrieved_chunks == [RetrievedChunkRef(section="1. Meals", distance=0.08)]


def test_citation_must_name_a_retrieved_section() -> None:
    """Reject a citation for a section that retrieval did not return."""
    with pytest.raises(ValidationError, match="retrieved chunks"):
        AskResponse(
            answer="Employees may claim up to $65 per day for meals.",
            citation=Citation(
                document="Employee Expense Policy",
                version="2.0",
                section="1. Meals",
            ),
            retrieved_chunks=[RetrievedChunkRef(section="2. Hotels", distance=0.2)],
        )


def test_unsupported_answer_has_no_citation() -> None:
    """Allow a refusal answer to omit the citation."""
    response = AskResponse(
        answer="The provided policy does not answer this question.",
        citation=None,
        retrieved_chunks=[RetrievedChunkRef(section="6. Submission Deadline", distance=0.4)],
    )
    assert response.citation is None


def test_retrieved_chunks_are_capped_and_sorted() -> None:
    """Reject more than three chunks or chunks that are out of distance order."""
    with pytest.raises(ValidationError, match="at most 3"):
        AskResponse(
            answer="Employees may claim up to $65 per day for meals.",
            citation=Citation(
                document="Employee Expense Policy",
                version="2.0",
                section="1. Meals",
            ),
            retrieved_chunks=[
                RetrievedChunkRef(section="1. Meals", distance=0.1),
                RetrievedChunkRef(section="2. Hotels", distance=0.2),
                RetrievedChunkRef(section="3. Airfare", distance=0.3),
                RetrievedChunkRef(section="4. Ground Transportation", distance=0.4),
            ],
        )

    with pytest.raises(ValidationError, match="sorted"):
        AskResponse(
            answer="Employees may claim up to $65 per day for meals.",
            citation=Citation(
                document="Employee Expense Policy",
                version="2.0",
                section="1. Meals",
            ),
            retrieved_chunks=[
                RetrievedChunkRef(section="2. Hotels", distance=0.2),
                RetrievedChunkRef(section="1. Meals", distance=0.1),
            ],
        )
