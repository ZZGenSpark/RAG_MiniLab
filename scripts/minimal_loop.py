"""Embed two known texts, store them, and retrieve the closer one.

This is the embed-store-retrieve proof that runs before the full corpus.
The screenshot is the printed ranking. Tests call run_minimal_loop with a
fake embedder so CI does not download MiniLM weights.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from adapter.chroma_store import ChromaPolicyStore
from adapter.sentence_transformer_embeddings import SentenceTransformerEmbeddingAdapter
from rag.embeddings import Embedder
from rag.rerank import Reranker
from rag.retrieve import retrieve
from rag.schema import PolicyChunk, RetrievedChunk
from rag.slug import slugify
from rag.store import PolicyStore

DOCUMENT = "Minimal Loop"
VERSION = "1.0"
QUESTION = "How many tokens does an employee receive?"

# Two texts that must not be confused: a token allotment and a foosball cap.
KNOWN_TEXTS: tuple[tuple[str, str, str], ...] = (
    (
        "1",
        "Token Allotment",
        "Every employee is issued 1,000,000 tokens at the start of each six-hour cycle.",
    ),
    (
        "2",
        "Foosball",
        "Foosball is capped at 20 minutes per employee per day.",
    ),
)


def known_chunks() -> list[PolicyChunk]:
    """Build one chunk per known text."""
    return [
        PolicyChunk(
            chunk_id=f"{slugify(DOCUMENT)}:v{VERSION}:section-{section}",
            document=DOCUMENT,
            version=VERSION,
            section=section,
            section_title=title,
            text=text,
        )
        for section, title, text in KNOWN_TEXTS
    ]


def run_minimal_loop(
    store: PolicyStore,
    embedder: Embedder,
    question: str = QUESTION,
    reranker: Reranker | None = None,
) -> list[RetrievedChunk]:
    """Embed the two known texts, store them, and return the nearest chunks."""
    chunks = known_chunks()
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
    store.upsert_chunks(chunks, embeddings)
    return retrieve(question, store=store, embedder=embedder, reranker=reranker)


def main() -> None:
    """Embed, store, and retrieve the two known texts, then print the ranking."""
    parser = argparse.ArgumentParser(description="Prove embed, store, and retrieve on two known texts.")
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=None,
        help="Chroma directory for this proof. Defaults to a temporary directory.",
    )
    args = parser.parse_args()

    if args.chroma_path is None:
        with tempfile.TemporaryDirectory() as chroma_name:
            _run(Path(chroma_name))
        return
    _run(args.chroma_path)


def _run(chroma_path: Path) -> None:
    """Run the loop against one Chroma directory and require the token text to rank first."""
    hits = run_minimal_loop(ChromaPolicyStore(chroma_path), SentenceTransformerEmbeddingAdapter())
    _print_ranking(hits)
    if not hits or hits[0].text != KNOWN_TEXTS[0][2]:
        raise SystemExit("retrieval did not rank the token allotment first")


def _print_ranking(hits: list[RetrievedChunk]) -> None:
    """Print the two stored texts and the retrieved order."""
    print("Stored 2 texts:")
    for _section, title, text in KNOWN_TEXTS:
        print(f"  {title}: {text}")
    print(f"Question: {QUESTION}")
    print("Retrieved:")
    for index, hit in enumerate(hits, start=1):
        print(f"  {index}. {hit.citation_section} distance={hit.distance:.4f}")
        print(f"     {hit.text}")


if __name__ == "__main__":
    main()
