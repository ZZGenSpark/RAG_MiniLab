"""Check schema validation for chunk ids, vectors, and ask responses."""

import pytest
from pydantic import ValidationError

from rag.schema import (
    AskResponse,
    ChromaRecords,
    ChunkMetadata,
    Citation,
    EmbeddedChunk,
    GroundedModelOutput,
    PolicyChunk,
    RetrievedChunkRef,
    require_uniform_embedding_width,
    to_chroma_records,
)


def test_numbered_rule_chunk_id_keeps_the_rule_number() -> None:
    """Accept a 7.3 id and cite the rule without turning the parent into the label."""
    chunk = PolicyChunk(
        chunk_id="hr-policy:v2.0:section-7.3",
        document="HR Policy",
        version="2.0",
        section="7.3",
        section_title="Weekend Abandonment Consequence",
        parent_heading="7. Shared Refrigerator Policy",
        text="7. Shared Refrigerator Policy\n\nFood left in the shared refrigerator is discarded.",
    )
    assert chunk.citation_section == "7.3 Weekend Abandonment Consequence"
    assert chunk.to_metadata().parent_heading == "7. Shared Refrigerator Policy"


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
            chunk_id="employee-expense-policy:v2.0:section-1",
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


def test_policy_chunk_citation_label_drops_the_stored_text() -> None:
    """Expose the numbered section label without the chunk body."""
    chunk = PolicyChunk(
        chunk_id="employee-expense-policy:v2.0:section-1",
        document="Employee Expense Policy",
        version="2.0",
        section="1",
        section_title="Meals",
        text="Employees may claim up to $65 per day.",
    )
    assert chunk.citation_section == "1. Meals"
    assert chunk.to_metadata() == ChunkMetadata(
        document="Employee Expense Policy",
        version="2.0",
        section="1",
        section_title="Meals",
    )


def test_chunk_metadata_rejects_unknown_fields() -> None:
    """Reject citation metadata that carries fields outside the stored shape."""
    with pytest.raises(ValidationError):
        ChunkMetadata.model_validate(
            {
                "document": "Employee Expense Policy",
                "version": "2.0",
                "section": "1",
                "section_title": "Meals",
                "text": "Employees may claim up to $65 per day.",
            }
        )


def test_embedded_chunk_from_stored_coerces_vector_values() -> None:
    """Rebuild a stored record and keep every vector component as a float."""
    embedded = EmbeddedChunk.from_stored(
        chunk_id="employee-expense-policy:v2.0:section-1",
        text="Employees may claim up to $65 per day.",
        metadata={
            "document": "Employee Expense Policy",
            "version": "2.0",
            "section": "1",
            "section_title": "Meals",
        },
        embedding=(1, 0),
    )
    assert embedded.embedding == [1.0, 0.0]
    assert embedded.text == "Employees may claim up to $65 per day."


def test_chroma_records_reject_unequal_field_lengths() -> None:
    """Reject an upsert payload whose ids, texts, vectors, and metadata differ in length."""
    metadata = ChunkMetadata(
        document="Employee Expense Policy",
        version="2.0",
        section="1",
        section_title="Meals",
    )
    with pytest.raises(ValidationError, match="same length"):
        ChromaRecords(
            ids=["employee-expense-policy:v2.0:section-1"],
            documents=["Employees may claim up to $65 per day.", "Hotels are reimbursable."],
            embeddings=[[1.0, 0.0]],
            metadatas=[metadata],
        )


def test_to_chroma_records_requires_one_vector_per_chunk() -> None:
    """Reject a batch that has a different number of chunks and vectors."""
    chunk = PolicyChunk(
        chunk_id="employee-expense-policy:v2.0:section-1",
        document="Employee Expense Policy",
        version="2.0",
        section="1",
        section_title="Meals",
        text="Employees may claim up to $65 per day.",
    )
    with pytest.raises(ValueError, match="exactly one embedding"):
        to_chroma_records([chunk], [])


def test_as_upsert_serializes_metadata_as_dicts() -> None:
    """Dump citation metadata to plain dicts for the Chroma client."""
    chunk = PolicyChunk(
        chunk_id="employee-expense-policy:v2.0:section-1",
        document="Employee Expense Policy",
        version="2.0",
        section="1",
        section_title="Meals",
        text="Employees may claim up to $65 per day.",
    )
    payload = to_chroma_records([chunk], [[1.0, 0.0]]).as_upsert()
    assert payload["ids"] == ["employee-expense-policy:v2.0:section-1"]
    assert payload["documents"] == ["Employees may claim up to $65 per day."]
    assert payload["embeddings"] == [[1.0, 0.0]]
    assert payload["metadatas"] == [
        {
            "document": "Employee Expense Policy",
            "version": "2.0",
            "section": "1",
            "section_title": "Meals",
            "parent_heading": "",
        }
    ]


def test_uniform_embedding_width_accepts_an_empty_batch() -> None:
    """Treat a batch with no vectors as having no width to compare."""
    assert require_uniform_embedding_width([]) is None


def test_uniform_embedding_width_rejects_an_empty_vector() -> None:
    """Reject a vector that has no components."""
    with pytest.raises(ValueError, match="width"):
        require_uniform_embedding_width([[]])


def test_grounded_model_output_ignores_extra_keys() -> None:
    """Keep an answer when the model adds keys beyond the required JSON shape."""
    parsed = GroundedModelOutput.model_validate(
        {"answerable": True, "answer": "Employees may claim up to $65 per day.", "note": "extra"}
    )
    assert parsed.answerable is True
    assert parsed.answer == "Employees may claim up to $65 per day."


def test_ask_response_rejects_an_empty_answer() -> None:
    """Reject a response whose answer text is empty."""
    with pytest.raises(ValidationError):
        AskResponse(answer="", citation=None, retrieved_chunks=[])


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
