"""Command-line entry point that asks one grounded policy question."""

from __future__ import annotations

import argparse

from adapter.chroma_store import ChromaPolicyStore
from adapter.ollama_chat import OllamaChatAdapter
from adapter.sentence_transformer_embeddings import SentenceTransformerEmbeddingAdapter
from config import CHROMA_PATH
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

    store = ChromaPolicyStore(args.chroma_path or CHROMA_PATH)
    response = ask(
        args.question,
        store=store,
        embedder=SentenceTransformerEmbeddingAdapter(),
        generator=OllamaChatAdapter(),
    )
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
