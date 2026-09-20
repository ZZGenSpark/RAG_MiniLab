from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_POLICY_PATH = Path("policy.md")
DEFAULT_CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "chroma_db"))
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
DEFAULT_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b")
