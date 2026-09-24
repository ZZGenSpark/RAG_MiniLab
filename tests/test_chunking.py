"""Check that policy markdown becomes six intact, stably identified chunks."""

import pytest

from config import POLICY_PATH
from rag.chunking import chunk_id_for, chunk_policy, chunk_policy_file
from rag.schema import PolicyChunk

EXPECTED_CHUNKS = [
    {
        "chunk_id": "expense-policy:v2.0:section-1",
        "document": "Employee Expense Policy",
        "version": "2.0",
        "section": "1",
        "section_title": "Meals",
        "text": (
            "Employees may claim up to $65 per day for meals while traveling overnight.\n"
            "Alcohol is not reimbursable."
        ),
    },
    {
        "chunk_id": "expense-policy:v2.0:section-2",
        "document": "Employee Expense Policy",
        "version": "2.0",
        "section": "2",
        "section_title": "Hotels",
        "text": (
            "Hotels are reimbursable up to $225 per night.\n"
            "A manager must approve higher rates before booking."
        ),
    },
    {
        "chunk_id": "expense-policy:v2.0:section-3",
        "document": "Employee Expense Policy",
        "version": "2.0",
        "section": "3",
        "section_title": "Airfare",
        "text": (
            "Employees must purchase economy airfare.\n"
            "Business-class airfare requires written approval from a vice president."
        ),
    },
    {
        "chunk_id": "expense-policy:v2.0:section-4",
        "document": "Employee Expense Policy",
        "version": "2.0",
        "section": "4",
        "section_title": "Ground Transportation",
        "text": (
            "Taxi, rideshare, train, and public-transit expenses are reimbursable.\n"
            "Luxury vehicle upgrades are not reimbursable."
        ),
    },
    {
        "chunk_id": "expense-policy:v2.0:section-5",
        "document": "Employee Expense Policy",
        "version": "2.0",
        "section": "5",
        "section_title": "Receipts",
        "text": "Receipts are required for individual expenses of $25 or more.",
    },
    {
        "chunk_id": "expense-policy:v2.0:section-6",
        "document": "Employee Expense Policy",
        "version": "2.0",
        "section": "6",
        "section_title": "Submission Deadline",
        "text": "Expense reports must be submitted within 30 days after travel ends.",
    },
]


def test_policy_file_produces_exactly_six_chunks() -> None:
    """Confirm the policy file yields exactly six chunks."""
    chunks = chunk_policy_file(POLICY_PATH)
    assert len(chunks) == 6


def test_chunks_have_stable_ids_and_required_metadata() -> None:
    """Confirm each chunk keeps its stable id and citation metadata."""
    chunks = chunk_policy_file(POLICY_PATH)
    assert chunks == [PolicyChunk.model_validate(row) for row in EXPECTED_CHUNKS]


def test_chunk_text_is_the_full_original_section_body() -> None:
    """Confirm chunk text is the original section body without the heading."""
    source = POLICY_PATH.read_text(encoding="utf-8")
    chunks = chunk_policy(source)

    for chunk in chunks:
        assert chunk.text in source
        assert f"## {chunk.section}. {chunk.section_title}" not in chunk.text
        assert chunk.document not in chunk.text


def test_chunks_do_not_cut_sentences_in_half() -> None:
    """Confirm every policy sentence stays whole inside one chunk."""
    chunks = chunk_policy_file(POLICY_PATH)
    sentences = [
        "Employees may claim up to $65 per day for meals while traveling overnight.",
        "Alcohol is not reimbursable.",
        "Hotels are reimbursable up to $225 per night.",
        "A manager must approve higher rates before booking.",
        "Employees must purchase economy airfare.",
        "Business-class airfare requires written approval from a vice president.",
        "Taxi, rideshare, train, and public-transit expenses are reimbursable.",
        "Luxury vehicle upgrades are not reimbursable.",
        "Receipts are required for individual expenses of $25 or more.",
        "Expense reports must be submitted within 30 days after travel ends.",
    ]

    combined = "\n".join(chunk.text for chunk in chunks)
    for sentence in sentences:
        assert sentence in combined
        assert not any(
            sentence[: len(sentence) // 2] in chunk.text and sentence not in chunk.text
            for chunk in chunks
        )

    for chunk in chunks:
        assert chunk.text[-1] in ".!?"
        assert chunk.text[0].isupper()


def test_version_must_include_a_minor_number() -> None:
    """Reject a heading version that is not major.minor."""
    markdown = (
        "# Employee Expense Policy — Version 2\n\n"
        "## 1. Meals\n"
        "Employees may claim up to $65 per day.\n"
    )
    with pytest.raises(ValueError, match="Version"):
        chunk_policy(markdown)


def test_missing_title_is_rejected() -> None:
    """Reject markdown that has no document title and version."""
    with pytest.raises(ValueError, match="missing"):
        chunk_policy("## 1. Meals\nEmployees may claim up to $65 per day.\n")


def test_chunk_id_encodes_version_and_section() -> None:
    """Keep the chunk id tied to the policy version and section number."""
    assert chunk_id_for("2.0", "5") == "expense-policy:v2.0:section-5"


def test_section_that_ends_mid_sentence_is_rejected() -> None:
    """Reject a section body that does not finish a sentence."""
    markdown = (
        "# Employee Expense Policy — Version 2.0\n\n"
        "## 1. Meals\n"
        "Employees may claim up to $65 per day"
    )
    with pytest.raises(ValueError, match="mid-sentence"):
        chunk_policy(markdown)


def test_section_that_starts_mid_sentence_is_rejected() -> None:
    """Reject a section body that begins in the middle of a sentence."""
    markdown = (
        "# Employee Expense Policy — Version 2.0\n\n"
        "## 1. Meals\n"
        "up to $65 per day."
    )
    with pytest.raises(ValueError, match="mid-sentence"):
        chunk_policy(markdown)


def test_hyphenated_version_heading_is_accepted() -> None:
    """Accept a title that separates the version with a hyphen."""
    markdown = (
        "# Employee Expense Policy - Version 2.0\n\n"
        "## 1. Meals\n"
        "Employees may claim up to $65 per day.\n"
    )
    chunks = chunk_policy(markdown)
    assert chunks[0].version == "2.0"
    assert chunks[0].document == "Employee Expense Policy"
    assert chunks[0].chunk_id == "expense-policy:v2.0:section-1"


def test_empty_section_is_rejected() -> None:
    """Reject a numbered section that has no body text."""
    markdown = (
        "# Employee Expense Policy — Version 2.0\n\n"
        "## 1. Meals\n"
        "Employees may claim up to $65 per day.\n\n"
        "## 2. Hotels\n"
    )
    with pytest.raises(ValueError, match="no body text"):
        chunk_policy(markdown)
