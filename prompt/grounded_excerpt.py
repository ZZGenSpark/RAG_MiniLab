"""Single-excerpt prompt for a grounded policy answer."""

from __future__ import annotations

from pathlib import Path

INSTRUCTION_PATH = Path(__file__).with_name("grounded_excerpt.txt")
INSTRUCTION = INSTRUCTION_PATH.read_text(encoding="utf-8").strip()


def build_grounded_excerpt_prompt(question: str, section_label: str, text: str) -> str:
    """Build the single-excerpt prompt for one question and chunk."""
    excerpt = f"[Section {section_label}]\n{text}"
    return f"{INSTRUCTION}\n\nQuestion: {question}\n\nPolicy excerpt:\n{excerpt}"
