"""Project-wide configuration.

Add shared paths, model names, and other settings here.
Values read from the environment (and `.env`) fall back to the defaults below.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent

# Paths
SOURCE_DIR = REPO_ROOT / "source"
POLICY_PATH = Path(os.getenv("POLICY_PATH", SOURCE_DIR / "policy.md"))
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "chroma_db"))
EVAL_OUTPUT_PATH = REPO_ROOT / "outputs" / "required_questions.json"

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b")

# Chroma collection
COLLECTION_NAME = "expense_policy"
DISTANCE_SPACE = "cosine"
