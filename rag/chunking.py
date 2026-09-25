"""Split policy markdown into one chunk per deepest numbered rule.

A `3.1` or `7.3` subsection is its own chunk. A section with no subsections,
such as `6. Boss Error Grace Period`, is one chunk. The parent heading is
metadata and a prefix on each child chunk, not a chunk of its own. Overlap
is 0 because each chunk is already one complete rule.
"""

from __future__ import annotations

import re
from pathlib import Path

from rag.schema import VERSION_PATTERN, PolicyChunk
from rag.slug import slugify

TITLE_RE = re.compile(
    rf"^#\s+(?P<document>.+?)\s+[—–-]\s+Version\s+(?P<version>{VERSION_PATTERN})\s*$",
    re.MULTILINE,
)
HEADING_RE = re.compile(
    r"^(?P<hashes>#{2,3})\s+(?P<section>\d+(?:\.\d+)?)\.?\s+(?P<section_title>.+?)\s*$",
    re.MULTILINE,
)
SENTENCE_MARK_RE = re.compile(r"[.!?]")


def chunk_id_for(document: str, version: str, section: str) -> str:
    """Build the stable chunk id for a policy document, version, and section.

    The document slug is what keeps two different policies from colliding on
    the same id. The section may be a parent number (`6`) or a rule (`7.3`).
    """
    return f"{slugify(document)}:v{version}:section-{section}"


def chunk_policy_file(path: str | Path) -> list[PolicyChunk]:
    """Read a markdown file and return one chunk per deepest numbered rule."""
    return chunk_policy(Path(path).read_text(encoding="utf-8"))


def chunk_policy(markdown: str) -> list[PolicyChunk]:
    """Split policy markdown on the deepest numbered rule, with no overlap."""
    document, version = _parse_document_header(markdown)
    headings = list(HEADING_RE.finditer(markdown))
    if not headings:
        raise ValueError("policy markdown contains no numbered section headings")

    chunks: list[PolicyChunk] = []
    index = 0
    while index < len(headings):
        heading = headings[index]
        if _level(heading) != 2:
            raise ValueError(f"section {heading.group('section')} has no parent section")

        child_end = index + 1
        while child_end < len(headings) and _level(headings[child_end]) == 3:
            child_end += 1
        children = headings[index + 1 : child_end]
        next_parent = headings[child_end] if child_end < len(headings) else None

        if not children:
            body = _body(markdown, heading, next_parent)
            chunks.append(_make_chunk(document, version, heading, "", body))
            index += 1
            continue

        parent_heading = f"{heading.group('section')}. {heading.group('section_title')}"
        intro = _body(markdown, heading, children[0])
        for child_index, child in enumerate(children):
            following = children[child_index + 1] if child_index + 1 < len(children) else next_parent
            body = _body(markdown, child, following)
            if child_index == 0 and intro:
                body = f"{intro}\n\n{body}"
            chunks.append(_make_chunk(document, version, child, parent_heading, body))
        index = child_end
    return chunks


def _make_chunk(
    document: str,
    version: str,
    heading: re.Match[str],
    parent_heading: str,
    body: str,
) -> PolicyChunk:
    """Build one chunk, prefixing a child rule with its parent heading."""
    section = heading.group("section")
    if not body:
        raise ValueError(f"section {section} has no body text")
    _assert_rule_is_intact(body, section)
    text = f"{parent_heading}\n\n{body}" if parent_heading else body
    return PolicyChunk(
        chunk_id=chunk_id_for(document, version, section),
        document=document,
        version=version,
        section=section,
        section_title=heading.group("section_title"),
        parent_heading=parent_heading,
        text=text,
    )


def _level(heading: re.Match[str]) -> int:
    """Return 2 for a parent section and 3 for a numbered rule."""
    return len(heading.group("hashes"))


def _body(markdown: str, heading: re.Match[str], following: re.Match[str] | None) -> str:
    """Return the text between this heading and the next one."""
    end = following.start() if following is not None else len(markdown)
    return markdown[heading.end() : end].strip()


def _parse_document_header(markdown: str) -> tuple[str, str]:
    """Extract the document title and version from the top-level heading."""
    match = TITLE_RE.search(markdown)
    if match is None:
        raise ValueError("policy markdown is missing a '# Title — Version X.Y' heading")
    return match.group("document").strip(), match.group("version")


def _assert_rule_is_intact(text: str, section: str) -> None:
    """Reject a rule that starts or ends before any sentence is complete.

    A rule may end on a table row, such as a protein amount, as long as the
    rule already contains a finished sentence.
    """
    if text[0].isalpha() and text[0].islower():
        raise ValueError(f"section {section} appears to start mid-sentence")
    if SENTENCE_MARK_RE.search(text) is None:
        raise ValueError(f"section {section} appears to cut off mid-sentence")
