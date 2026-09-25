"""One prompt for the final policy excerpts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

INSTRUCTION_PATH = Path(__file__).with_name("grounded_excerpt.txt")
INSTRUCTION = INSTRUCTION_PATH.read_text(encoding="utf-8").strip()


def build_grounded_excerpt_prompt(question: str, excerpts: Sequence[str]) -> str:
    """Build one prompt that shows every excerpt the model may cite."""
    body = "\n\n".join(excerpts)
    return f"{INSTRUCTION}\n\nQuestion: {question}\n\nPolicy excerpts:\n{body}"
