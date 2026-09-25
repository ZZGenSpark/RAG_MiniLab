"""Okapi BM25 over stored chunk text.

The tokenizer keeps a section code such as 7.3 as one token. Splitting it into
7 and 3 would match every numbered rule.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from rag.schema import PolicyChunk

# A dotted rule is one token. Other words split on punctuation.
_TOKEN = re.compile(r"\d+\.\d+|[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Split text into BM25 tokens, keeping a dotted section code intact."""
    return [match.group(0).lower() for match in _TOKEN.finditer(text)]


def searchable_text(chunk: PolicyChunk) -> str:
    """Return the chunk text plus its section code, so a query for 7.3 can match."""
    return f"{chunk.section} {chunk.section_title}\n{chunk.text}"


def bm25_top(question: str, chunks: Sequence[PolicyChunk], n: int) -> list[PolicyChunk]:
    """Return up to n chunks with the highest Okapi BM25 score for the question."""
    if n < 1 or not chunks:
        return []
    documents = [tokenize(searchable_text(chunk)) for chunk in chunks]
    if all(len(tokens) == 0 for tokens in documents):
        return []
    scores = [float(score) for score in BM25Okapi(documents).get_scores(tokenize(question))]
    order = sorted(range(len(chunks)), key=lambda index: (-scores[index], chunks[index].chunk_id))
    return [chunks[index] for index in order if scores[index] > 0][:n]
