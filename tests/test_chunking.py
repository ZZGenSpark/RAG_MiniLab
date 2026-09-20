from pathlib import Path

import pytest

from rag.chunking import chunk_policy, chunk_policy_file

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "policy.md"

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
    chunks = chunk_policy_file(POLICY_PATH)
    assert len(chunks) == 6


def test_chunks_have_stable_ids_and_required_metadata() -> None:
    chunks = chunk_policy_file(POLICY_PATH)
    assert chunks == EXPECTED_CHUNKS


def test_chunk_text_is_the_full_original_section_body() -> None:
    source = POLICY_PATH.read_text(encoding="utf-8")
    chunks = chunk_policy(source)

    for chunk in chunks:
        assert chunk["text"] in source
        assert f"## {chunk['section']}. {chunk['section_title']}" not in chunk["text"]
        assert chunk["document"] not in chunk["text"]


def test_chunks_do_not_cut_sentences_in_half() -> None:
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

    combined = "\n".join(chunk["text"] for chunk in chunks)
    for sentence in sentences:
        assert sentence in combined
        assert not any(
            sentence[: len(sentence) // 2] in chunk["text"] and sentence not in chunk["text"]
            for chunk in chunks
        )

    for chunk in chunks:
        assert chunk["text"][-1] in ".!?"
        assert chunk["text"][0].isupper()


def test_missing_title_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing"):
        chunk_policy("## 1. Meals\nEmployees may claim up to $65 per day.\n")


def test_empty_section_is_rejected() -> None:
    markdown = (
        "# Employee Expense Policy — Version 2.0\n\n"
        "## 1. Meals\n"
        "Employees may claim up to $65 per day.\n\n"
        "## 2. Hotels\n"
    )
    with pytest.raises(ValueError, match="no body text"):
        chunk_policy(markdown)
