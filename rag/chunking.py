"""Split policy markdown into one chunk per numbered section.

Each chunk keeps the full section body and a stable id for citations.
"""

from __future__ import annotations

import re
from pathlib import Path

from rag.schema import CHUNK_ID_PREFIX, PolicyChunk

TITLE_RE = re.compile(
    r"^#\s+(?P<document>.+?)\s+[—–-]\s+Version\s+(?P<version>\d+(?:\.\d+)?)\s*$",
    re.MULTILINE,
)
SECTION_HEADING_RE = re.compile(
    r"^##\s+(?P<section>\d+)\.\s+(?P<section_title>.+?)\s*$",
    re.MULTILINE,
)
SENTENCE_END_RE = re.compile(r"[.!?]$")


def chunk_id_for(version: str, section: str) -> str:
    """Build the stable chunk id for a policy version and section."""
    return f"{CHUNK_ID_PREFIX}:v{version}:section-{section}"


def chunk_policy_file(path: str | Path) -> list[PolicyChunk]:
    """Read a markdown file and return one chunk per numbered section."""
    return chunk_policy(Path(path).read_text(encoding="utf-8"))


def chunk_policy(markdown: str) -> list[PolicyChunk]:
    """Split policy markdown into chunks with document, version, and section metadata."""
    document, version = _parse_document_header(markdown)
    headings = list(SECTION_HEADING_RE.finditer(markdown))
    if not headings:
        raise ValueError("policy markdown contains no numbered section headings")

    chunks: list[PolicyChunk] = []
    for index, heading in enumerate(headings):
        body_start = heading.end()
        body_end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        text = markdown[body_start:body_end].strip()
        if not text:
            raise ValueError(f"section {heading.group('section')} has no body text")
        _assert_sentences_intact(text, heading.group("section"))

        section = heading.group("section")
        chunks.append(
            PolicyChunk(
                chunk_id=chunk_id_for(version, section),
                document=document,
                version=version,
                section=section,
                section_title=heading.group("section_title"),
                text=text,
            )
        )
    return chunks


def _parse_document_header(markdown: str) -> tuple[str, str]:
    """Extract the document title and version from the top-level heading."""
    match = TITLE_RE.search(markdown)
    if match is None:
        raise ValueError("policy markdown is missing a '# Title — Version X.Y' heading")
    return match.group("document").strip(), match.group("version")


def _assert_sentences_intact(text: str, section: str) -> None:
    """Reject section text that starts or ends mid-sentence."""
    if not SENTENCE_END_RE.search(text):
        raise ValueError(f"section {section} appears to cut off mid-sentence")
    first_char = text[0]
    if first_char.isalpha() and first_char.islower():
        raise ValueError(f"section {section} appears to start mid-sentence")
