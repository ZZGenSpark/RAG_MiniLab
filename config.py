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

# Paths. The lab corpus is the markdown under source/policies/.
SOURCE_DIR = REPO_ROOT / "source"
POLICIES_DIR = SOURCE_DIR / "policies"
# Five policies are preprocessed into markdown. HR v1 and Preparedness v1 stay binaries.
INGEST_SOURCES = (
    SOURCE_DIR / "Doofenshmirtz Evil Inc - HR Policy v2.0.docx",
    SOURCE_DIR / "Doofenshmirtz Evil Inc - Health Policy v1.0 1.pdf",
    SOURCE_DIR / "Doofenshmirtz Evil Inc - Preparedness Policy v2.0 1.docx",
    SOURCE_DIR / "Doofenshmirtz Evil Inc - Time and Usage Policy v2.0 1.docx",
    SOURCE_DIR / "Doofenshmirtz Evil Inc - Time and Usage Policy v1.0 1.pdf",
)
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", REPO_ROOT / "chroma_db"))
EVAL_OUTPUT_PATH = REPO_ROOT / "outputs" / "required_questions.json"

# Ollama. Tags name one release. Unqualified names follow latest.
# nomic-embed-text:v1.5 — library digest 0a109f422b47
# qwen3:8b-q4_K_M — library digest 500a1f067a9f (the weights behind qwen3:8b)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
EMBED_MODEL_NAME = "nomic-embed-text"
EMBED_MODEL_VERSION = "v1.5"
CHAT_MODEL_NAME = "qwen3"
CHAT_MODEL_VERSION = "8b-q4_K_M"
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", f"{EMBED_MODEL_NAME}:{EMBED_MODEL_VERSION}")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", f"{CHAT_MODEL_NAME}:{CHAT_MODEL_VERSION}")
CHAT_TEMPERATURE = 0

# Local embeddings for the lab. all-MiniLM-L6-v2 is 384-dimensional and is stored in cosine space.
# The Ollama embedder above stays available and is not this path.
SENTENCE_TRANSFORMER_MODEL = os.getenv(
    "SENTENCE_TRANSFORMER_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# Retrieval
TOP_K = 3

# Chroma collection. The lab corpus lives in company_policies.
COLLECTION_NAME = "company_policies"
DISTANCE_SPACE = "cosine"
