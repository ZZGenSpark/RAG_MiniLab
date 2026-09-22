"""Command-line entry point that loads source/policy.md into Chroma."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import POLICY_PATH
from rag.ingest import ingest_policy


def main() -> None:
    """Parse CLI arguments and ingest the policy file."""
    parser = argparse.ArgumentParser(description="Ingest source/policy.md into Chroma with Ollama embeddings.")
    parser.add_argument(
        "--policy",
        type=Path,
        default=POLICY_PATH,
        help="Path to the policy markdown file.",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=None,
        help="Optional Chroma persistence directory. Defaults to CHROMA_PATH.",
    )
    args = parser.parse_args()

    chunk_ids = ingest_policy(args.policy, chroma_path=args.chroma_path)
    print(f"Ingested {len(chunk_ids)} policy chunks:")
    for chunk_id in chunk_ids:
        print(f"  {chunk_id}")


if __name__ == "__main__":
    main()
