"""Check that policy markdown becomes six intact, stably identified chunks."""

import pytest

from config import POLICIES_DIR
from rag.chunking import chunk_id_for, chunk_policy, chunk_policy_file
from rag.schema import PolicyChunk
from tests.support import EXPENSE_POLICY_FIXTURE

EXPECTED_CHUNKS = [
    {
        "chunk_id": "employee-expense-policy:v2.0:section-1",
        "document": "Employee Expense Policy",
        "version": "2.0",
        "section": "1",
        "section_title": "Meals",
        "text": (
            "Employees may claim up to $65 per day for meals while traveling overnight.\nAlcohol is not reimbursable."
        ),
    },
    {
        "chunk_id": "employee-expense-policy:v2.0:section-2",
        "document": "Employee Expense Policy",
        "version": "2.0",
        "section": "2",
        "section_title": "Hotels",
        "text": ("Hotels are reimbursable up to $225 per night.\nA manager must approve higher rates before booking."),
    },
    {
        "chunk_id": "employee-expense-policy:v2.0:section-3",
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
        "chunk_id": "employee-expense-policy:v2.0:section-4",
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
        "chunk_id": "employee-expense-policy:v2.0:section-5",
        "document": "Employee Expense Policy",
        "version": "2.0",
        "section": "5",
        "section_title": "Receipts",
        "text": "Receipts are required for individual expenses of $25 or more.",
    },
    {
        "chunk_id": "employee-expense-policy:v2.0:section-6",
        "document": "Employee Expense Policy",
        "version": "2.0",
        "section": "6",
        "section_title": "Submission Deadline",
        "text": "Expense reports must be submitted within 30 days after travel ends.",
    },
]


def test_policy_file_produces_exactly_six_chunks() -> None:
    """Confirm the policy file yields exactly six chunks."""
    chunks = chunk_policy_file(EXPENSE_POLICY_FIXTURE)
    assert len(chunks) == 6


def test_chunks_have_stable_ids_and_required_metadata() -> None:
    """Confirm each chunk keeps its stable id and citation metadata."""
    chunks = chunk_policy_file(EXPENSE_POLICY_FIXTURE)
    assert chunks == [PolicyChunk.model_validate(row) for row in EXPECTED_CHUNKS]


def test_chunk_text_is_the_full_original_section_body() -> None:
    """Confirm chunk text is the original section body without the heading."""
    source = EXPENSE_POLICY_FIXTURE.read_text(encoding="utf-8")
    chunks = chunk_policy(source)

    for chunk in chunks:
        assert chunk.text in source
        assert f"## {chunk.section}. {chunk.section_title}" not in chunk.text
        assert chunk.document not in chunk.text


def test_chunks_do_not_cut_sentences_in_half() -> None:
    """Confirm every policy sentence stays whole inside one chunk."""
    chunks = chunk_policy_file(EXPENSE_POLICY_FIXTURE)
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
        assert not any(sentence[: len(sentence) // 2] in chunk.text and sentence not in chunk.text for chunk in chunks)

    for chunk in chunks:
        assert chunk.text[-1] in ".!?"
        assert chunk.text[0].isupper()


def test_version_must_include_a_minor_number() -> None:
    """Reject a heading version that is not major.minor."""
    markdown = "# Employee Expense Policy — Version 2\n\n## 1. Meals\nEmployees may claim up to $65 per day.\n"
    with pytest.raises(ValueError, match="Version"):
        chunk_policy(markdown)


def test_missing_title_is_rejected() -> None:
    """Reject markdown that has no document title and version."""
    with pytest.raises(ValueError, match="missing"):
        chunk_policy("## 1. Meals\nEmployees may claim up to $65 per day.\n")


def test_chunk_id_encodes_the_document_slug_version_and_section() -> None:
    """Keep the chunk id tied to the document slug, version, and section number."""
    assert chunk_id_for("Employee Expense Policy", "2.0", "5") == "employee-expense-policy:v2.0:section-5"


def test_deepest_numbered_rule_is_a_chunk_and_its_parent_is_not() -> None:
    """Keep section 7.3, prefix it with the parent heading, and do not chunk section 7."""
    chunks = chunk_policy_file(POLICIES_DIR / "hr-policy-v2.0.md")
    weekend = next(chunk for chunk in chunks if chunk.section == "7.3")

    assert weekend.chunk_id == "hr-policy:v2.0:section-7.3"
    assert weekend.section_title == "Weekend Abandonment Consequence"
    assert weekend.parent_heading == "7. Shared Refrigerator Policy"
    assert weekend.citation_section == "7.3 Weekend Abandonment Consequence"
    assert weekend.text.startswith("7. Shared Refrigerator Policy\n\nFood left in the shared refrigerator")
    assert not any(chunk.section == "7" for chunk in chunks)

    grace = next(chunk for chunk in chunks if chunk.section == "6")
    assert grace.chunk_id == "hr-policy:v2.0:section-6"
    assert grace.parent_heading == ""
    assert grace.text.startswith("When a manager or executive states something incorrect")
    assert "6. Boss Error Grace Period" not in grace.text


def test_sibling_rules_do_not_share_body_text() -> None:
    """Keep overlap at zero between numbered rules under the same parent."""
    markdown = (
        "# HR Policy — Version 2.0\n\n"
        "## 7. Shared Refrigerator Policy\n\n"
        "### 7.1 Ownership\n\n"
        "No food kept in the shared refrigerator belongs to any individual employee.\n\n"
        "### 7.3 Weekend Abandonment Consequence\n\n"
        "Food left in the shared refrigerator over a weekend will be treated as abandoned.\n"
    )
    chunks = chunk_policy(markdown)
    assert [chunk.section for chunk in chunks] == ["7.1", "7.3"]
    assert "individual employee" not in chunks[1].text
    assert "treated as abandoned" not in chunks[0].text
    assert chunks[0].text.startswith("7. Shared Refrigerator Policy\n\n")
    assert chunks[1].chunk_id == "hr-policy:v2.0:section-7.3"


def test_chunk_id_keeps_different_documents_from_colliding() -> None:
    """Give two policies with the same version and section number different ids."""
    hr_id = chunk_id_for("HR Policy", "2.0", "1")
    time_id = chunk_id_for("Time & Usage Policy", "2.0", "1")
    assert hr_id != time_id
    assert hr_id == "hr-policy:v2.0:section-1"
    assert time_id == "time-and-usage-policy:v2.0:section-1"


def test_section_that_ends_mid_sentence_is_rejected() -> None:
    """Reject a section body that does not finish a sentence."""
    markdown = "# Employee Expense Policy — Version 2.0\n\n## 1. Meals\nEmployees may claim up to $65 per day"
    with pytest.raises(ValueError, match="mid-sentence"):
        chunk_policy(markdown)


def test_section_that_starts_mid_sentence_is_rejected() -> None:
    """Reject a section body that begins in the middle of a sentence."""
    markdown = "# Employee Expense Policy — Version 2.0\n\n## 1. Meals\nup to $65 per day."
    with pytest.raises(ValueError, match="mid-sentence"):
        chunk_policy(markdown)


def test_hyphenated_version_heading_is_accepted() -> None:
    """Accept a title that separates the version with a hyphen."""
    markdown = "# Employee Expense Policy - Version 2.0\n\n## 1. Meals\nEmployees may claim up to $65 per day.\n"
    chunks = chunk_policy(markdown)
    assert chunks[0].version == "2.0"
    assert chunks[0].document == "Employee Expense Policy"
    assert chunks[0].chunk_id == "employee-expense-policy:v2.0:section-1"


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
