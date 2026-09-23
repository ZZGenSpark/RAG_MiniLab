"""Public API for the expense-policy RAG pipeline.

Re-exports ingest, retrieval, generation, and the shared schema types.
"""

from rag.ask import ask
from rag.chunking import chunk_policy, chunk_policy_file
from rag.embeddings import Embedder
from rag.generate import generate_answer
from rag.ingest import ingest_policy
from rag.retrieve import retrieve
from rag.schema import (
    COLLECTION_NAME,
    DISTANCE_SPACE,
    REFUSAL_ANSWER,
    AskResponse,
    Citation,
    EmbeddedChunk,
    PolicyChunk,
    RetrievedChunk,
    RetrievedChunkRef,
)
from rag.store import PolicyStore

__all__ = [
    "COLLECTION_NAME",
    "DISTANCE_SPACE",
    "AskResponse",
    "Citation",
    "Embedder",
    "EmbeddedChunk",
    "PolicyChunk",
    "PolicyStore",
    "REFUSAL_ANSWER",
    "RetrievedChunk",
    "RetrievedChunkRef",
    "ask",
    "chunk_policy",
    "chunk_policy_file",
    "generate_answer",
    "ingest_policy",
    "retrieve",
]
