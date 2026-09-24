"""Check heading splits for one docx rule and one glued pdf fragment."""

from pathlib import Path

import pytest

from config import INGEST_SOURCES, SOURCE_DIR
from rag.preprocess import paragraphs_to_markdown, text_to_markdown, write_policy


def test_docx_rule_paragraph_splits_heading_and_body() -> None:
    """Split `3.1 Requirement. Every email...` into a heading and a body."""
    markdown = paragraphs_to_markdown(
        ["3.1 Requirement. Every email must start with a joke."]
    )
    assert "### 3.1 Requirement" in markdown
    assert "Every email must start with a joke." in markdown
    assert "Requirement. Every" not in markdown


def test_glued_pdf_fragment_splits_purpose_from_body() -> None:
    """Split `1. PurposeThis policy` into a heading and a body."""
    markdown = text_to_markdown("1. PurposeThis policy")
    assert "## 1. Purpose" in markdown
    assert "This policy" in markdown
    assert "PurposeThis" not in markdown


def test_allowlist_has_the_five_ingested_policies() -> None:
    """Leave HR v1 and Preparedness v1 out of preprocessing."""
    names = [path.name for path in INGEST_SOURCES]
    assert len(names) == 5
    assert not any(name.startswith("Doofenshmirtz Evil Inc - HR Policy v1") for name in names)
    assert not any("Preparedness Policy v1" in name for name in names)


def test_excluded_binary_is_rejected(tmp_path: Path) -> None:
    """Refuse to convert a policy that is not on the allowlist."""
    excluded = SOURCE_DIR / "Doofenshmirtz Evil Inc - HR Policy v1.0 1.pdf"
    with pytest.raises(ValueError, match="not an allowlisted policy"):
        write_policy(excluded, tmp_path)
