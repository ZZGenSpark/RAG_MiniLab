"""Command-line entry point that asks one grounded policy question."""

from __future__ import annotations

import argparse

from adapter.chroma_store import ChromaPolicyStore
from rag.ask import ask


def main() -> None:
    """Parse a question and print the structured answer as JSON."""
    parser = argparse.ArgumentParser(description="Ask a grounded question against the ingested policy.")
    parser.add_argument("question", help="Question to answer from the policy excerpts.")
    parser.add_argument(
        "--chroma-path",
        default=None,
        help="Optional Chroma persistence directory. Defaults to CHROMA_PATH.",
    )
    args = parser.parse_args()

    store = ChromaPolicyStore(args.chroma_path) if args.chroma_path else None
    response = ask(args.question, store=store)
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
