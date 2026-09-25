"""Prove embed, store, and retrieve on two known texts without a live model."""

from collections.abc import Sequence
from pathlib import Path

from adapter.chroma_store import ChromaPolicyStore
from scripts.minimal_loop import KNOWN_TEXTS, QUESTION, run_minimal_loop
from tests.support import KeepingReranker

TOKEN_TEXT = KNOWN_TEXTS[0][2]
FOOSBALL_TEXT = KNOWN_TEXTS[1][2]


class TwoTextEmbedder:
    """Map the two known texts and the question onto fixed, separable vectors."""

    def __init__(self) -> None:
        """Record every text that is embedded."""
        self.texts: list[list[str]] = []
        self.queries: list[str] = []

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text. The token sentence points along the first axis."""
        self.texts.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return the question vector, near the token sentence."""
        self.queries.append(text)
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        """Place the token text and its question near each other, and foosball apart."""
        if text in {TOKEN_TEXT, QUESTION}:
            return [1.0, 0.0]
        if text == FOOSBALL_TEXT:
            return [0.0, 1.0]
        raise AssertionError(f"unexpected text: {text}")


def test_minimal_loop_retrieves_the_matching_text(tmp_path: Path) -> None:
    """Store both known texts and retrieve the token allotment for the token question."""
    embedder = TwoTextEmbedder()
    store = ChromaPolicyStore(tmp_path / "chroma")

    hits = run_minimal_loop(store, embedder, reranker=KeepingReranker())

    assert store.count() == 2
    assert [hit.text for hit in hits] == [TOKEN_TEXT, FOOSBALL_TEXT]
    assert hits[0].citation_section == "1. Token Allotment"
    assert hits[0].distance < hits[1].distance
    assert embedder.texts == [[TOKEN_TEXT, FOOSBALL_TEXT]]
    assert embedder.queries == [QUESTION]
