"""Command-line entry point that loads source/policies into Chroma."""

from __future__ import annotations

import argparse
from pathlib import Path

from adapter.chroma_store import ChromaPolicyStore
from adapter.ollama_embeddings import OllamaEmbeddingAdapter
from config import CHROMA_PATH, POLICIES_DIR
from rag.ingest import ingest_policy


def main() -> None:
    """Parse CLI arguments and ingest the markdown policies."""
    parser = argparse.ArgumentParser(
        description="Ingest markdown policies from source/policies into Chroma with Ollama embeddings."
    )
    parser.add_argument(
        "--policies",
        type=Path,
        default=POLICIES_DIR,
        help="Directory of markdown policies. Defaults to source/policies.",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=None,
        help="Optional Chroma persistence directory. Defaults to CHROMA_PATH.",
    )
    args = parser.parse_args()

    store = ChromaPolicyStore(args.chroma_path or CHROMA_PATH)
    chunk_ids = ingest_policy(args.policies, store=store, embedder=OllamaEmbeddingAdapter())
    print(f"Ingested {len(chunk_ids)} policy chunks:")
    for chunk_id in chunk_ids:
        print(f"  {chunk_id}")


if __name__ == "__main__":
    main()
