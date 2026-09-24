"""Turn the allowlisted policy files into markdown.

Docx text comes from python-docx. Pdf text comes from pypdf. This module
only splits those strings into headings. Ingest reads the markdown, not the
binaries.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document
from pypdf import PdfReader

from config import INGEST_SOURCES, POLICIES_DIR

TITLE_RE = re.compile(
    r"^(?P<title>.+?)\s+[—–-]\s+Version\s+(?P<version>\d+\.\d+)\s*$"
)
RULE_RE = re.compile(r"^(?P<number>\d+\.\d+)\s+(?P<rest>.+)$")
PARENT_RE = re.compile(r"^(?P<number>\d+)\.\s+(?!\d)(?P<title>.+)$")
RULE_BODY_RE = re.compile(r"^(?P<title>.+?)\.\s+(?P<body>.+)$", re.DOTALL)
VERSION_GLUE_RE = re.compile(r"(Version\s+\d+\.\d)(?=\d+\.)")
CAMEL_GLUE_RE = re.compile(r"([a-z])([A-Z])")
SECTION_GLUE_RE = re.compile(r"(?<!\n)(?<![\d.])(\d+\.\d*)(?=\s*[A-Z])")

ALLOWED_NAMES = frozenset(path.name for path in INGEST_SOURCES)


def extract_docx(path: str | Path) -> list[str]:
    """Return one stripped paragraph per non-empty Word paragraph."""
    document = Document(str(path))
    return [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]


def extract_pdf(path: str | Path) -> str:
    """Return the concatenated page text from a PDF."""
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def unglue_pdf(text: str) -> str:
    """Separate a heading that pypdf glued to the following sentence.

    `1. PurposeThis policy` becomes a heading line and a body line. Real
    extracts that already contain newlines pass through unchanged.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = VERSION_GLUE_RE.sub(r"\1\n", text)
    text = CAMEL_GLUE_RE.sub(r"\1\n\2", text)
    text = SECTION_GLUE_RE.sub(r"\n\1", text)
    return text


def paragraphs_to_markdown(paragraphs: list[str]) -> str:
    """Render Word paragraphs as policy markdown."""
    return _lines_to_markdown(paragraphs)


def text_to_markdown(text: str) -> str:
    """Render extracted PDF text as policy markdown."""
    return _lines_to_markdown(unglue_pdf(text).splitlines())


def write_policies(
    sources: tuple[Path, ...] = INGEST_SOURCES,
    dest_dir: str | Path = POLICIES_DIR,
) -> list[Path]:
    """Write one markdown file per allowlisted source and return those paths."""
    destination = Path(dest_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for source in sources:
        written.append(write_policy(source, destination))
    return written


def write_policy(source: str | Path, dest_dir: str | Path) -> Path:
    """Convert one allowlisted file and write its markdown."""
    path = Path(source)
    _require_allowlisted(path)
    if path.suffix.lower() == ".docx":
        markdown = paragraphs_to_markdown(extract_docx(path))
    elif path.suffix.lower() == ".pdf":
        markdown = text_to_markdown(extract_pdf(path))
    else:
        raise ValueError(f"{path.name} is not a pdf or docx file")
    if not markdown.startswith("# "):
        raise ValueError(f"{path.name} did not produce a policy title")
    output = Path(dest_dir) / _markdown_name(markdown)
    output.write_text(markdown, encoding="utf-8")
    return output


def _require_allowlisted(path: Path) -> None:
    """Reject files that are not one of the five ingested policies."""
    if path.name not in ALLOWED_NAMES:
        raise ValueError(f"{path.name} is not an allowlisted policy")


def _lines_to_markdown(lines: list[str]) -> str:
    """Turn heading and body lines into one markdown document."""
    title: str | None = None
    version: str | None = None
    sections: list[dict[str, object]] = []
    current: dict[str, object] | None = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        title_match = TITLE_RE.match(line)
        if title_match and title is None:
            title = title_match.group("title").strip()
            version = title_match.group("version")
            continue
        rule_match = RULE_RE.match(line)
        if rule_match:
            current = _rule_section(rule_match.group("number"), rule_match.group("rest"))
            sections.append(current)
            continue
        parent_match = PARENT_RE.match(line)
        if parent_match:
            current = {
                "level": 2,
                "number": parent_match.group("number"),
                "title": parent_match.group("title").strip(),
                "body": [],
            }
            sections.append(current)
            continue
        if current is None:
            continue
        body = current["body"]
        assert isinstance(body, list)
        if body and _continues_previous_line(str(body[-1]), line):
            body[-1] = f"{body[-1]} {line}"
        else:
            body.append(line)

    parts: list[str] = []
    if title and version:
        parts.append(f"# {title} — Version {version}")
    for section in sections:
        number = section["number"]
        heading = section["title"]
        if section["level"] == 2:
            parts.append(f"## {number}. {heading}")
        else:
            parts.append(f"### {number} {heading}")
        body = section["body"]
        assert isinstance(body, list)
        if body:
            parts.append("\n".join(body))
    return "\n\n".join(parts) + ("\n" if parts else "")


def _continues_previous_line(previous: str, line: str) -> bool:
    """Join a PDF wrap without merging the health-policy protein table."""
    if line[:1].islower():
        return True
    return previous[-1:] not in ".!?" and len(line.split()) == 1


def _rule_section(number: str, rest: str) -> dict[str, object]:
    """Split `3.1 Requirement. Every email...` into a heading and a body."""
    match = RULE_BODY_RE.match(rest)
    if match:
        return {
            "level": 3,
            "number": number,
            "title": match.group("title").strip(),
            "body": [match.group("body").strip()],
        }
    return {"level": 3, "number": number, "title": rest.strip(), "body": []}


def _markdown_name(markdown: str) -> str:
    """Build `time-and-usage-policy-v1.0.md` from the markdown title."""
    first = markdown.splitlines()[0].removeprefix("# ").strip()
    match = TITLE_RE.match(first)
    if match is None:
        raise ValueError("markdown is missing a '# Title — Version X.Y' heading")
    slug = match.group("title").lower().replace("&", "and")
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return f"{slug}-v{match.group('version')}.md"


def main() -> None:
    """Write markdown for the five allowlisted policies."""
    parser = argparse.ArgumentParser(description="Write markdown for the allowlisted policy files.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=POLICIES_DIR,
        help="Directory for the markdown policies.",
    )
    args = parser.parse_args()
    for path in write_policies(dest_dir=args.dest):
        print(path)


if __name__ == "__main__":
    main()
