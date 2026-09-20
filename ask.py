from __future__ import annotations

import argparse

from rag.ask import ask
from rag.store import PolicyStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a grounded question against the ingested policy.")
    parser.add_argument("question", help="Question to answer from the policy excerpts.")
    parser.add_argument(
        "--chroma-path",
        default=None,
        help="Optional Chroma persistence directory. Defaults to CHROMA_PATH.",
    )
    args = parser.parse_args()

    store = PolicyStore(args.chroma_path) if args.chroma_path else None
    response = ask(args.question, store=store)
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
