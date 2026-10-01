"""Prompt templates used by the RAG pipeline."""

from prompt.grounded_excerpt import INSTRUCTION, build_grounded_excerpt_prompt

__all__ = [
    "INSTRUCTION",
    "build_grounded_excerpt_prompt",
]
