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
EVAL_REPORT_PATH = REPO_ROOT / "outputs" / "eval_report.json"
AUDIT_PATH = REPO_ROOT / "outputs" / "audit.jsonl"

# Ollama chat. Tags name one release. Unqualified names follow latest.
# qwen3:8b-q4_K_M — library digest 500a1f067a9f (the weights behind qwen3:8b)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
CHAT_MODEL_NAME = "qwen3"
CHAT_MODEL_VERSION = "8b-q4_K_M"
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", f"{CHAT_MODEL_NAME}:{CHAT_MODEL_VERSION}")
CHAT_TEMPERATURE = 0

# Local embeddings for the lab. all-MiniLM-L6-v2 is 384-dimensional and is stored in cosine space.
SENTENCE_TRANSFORMER_MODEL = os.getenv(
    "SENTENCE_TRANSFORMER_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

# Retrieval. Vector search fetches 5. Hybrid fetches 10 on each leg, then equal-weight RRF keeps 5.
# The cross-encoder reranks that shortlist and keeps TOP_K.
TOP_K = 3
VECTOR_CANDIDATES = 5
HYBRID_CANDIDATES = 10
FUSED_CANDIDATES = 5
RRF_K = 60

# Chroma collection. The lab corpus lives in company_policies.
COLLECTION_NAME = "company_policies"
DISTANCE_SPACE = "cosine"


def typesafe_api_key() -> str:
    """Return the TypeSafe API key, or an empty string when it is unset.

    An empty value selects the section-code fallback instead of Jev.
    """
    return os.getenv("TYPESAFE_API_KEY", "").strip()
