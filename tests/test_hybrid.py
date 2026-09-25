"""Fuse BM25 and vector ranks. The Section 7.3 comparison does not call Jev."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from adapter.chroma_store import ChromaPolicyStore
from adapter.jev_router import JevRouter
from config import POLICIES_DIR
from rag.ask import ask
from rag.chunking import chunk_policy_file
from rag.lexical import bm25_top, tokenize
from rag.retrieve import fuse_reciprocal_ranks, retrieve
from rag.route import RetrievalDecision
from rag.schema import EmbeddedChunk, PolicyChunk, RetrievedChunk
from tests.support import KeepingReranker

QUESTION = "Section 7.3"
SECTION = "7.3 Weekend Abandonment Consequence"


class FixedEmbedder:
    """Return one query vector and record the embed call."""

    def __init__(self, vector: list[float]) -> None:
        """Store the vector used for every query."""
        self.vector = vector

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return the query vector once per text."""
        return [list(self.vector) for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return the query vector."""
        return list(self.vector)


class RecordingStore:
    """Delegate to Chroma and record how retrieval searches."""

    def __init__(self, store: ChromaPolicyStore) -> None:
        """Wrap a store that already holds the chunks."""
        self.store = store
        self.queries: list[int] = []
        self.get_all_calls = 0

    def upsert_chunks(
        self,
        chunks: Sequence[PolicyChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> list[str]:
        """Store chunks on the wrapped database."""
        return self.store.upsert_chunks(chunks, embeddings)

    def count(self) -> int:
        """Return how many chunks are stored."""
        return self.store.count()

    def query_similar(self, embedding: Sequence[float], n_results: int = 3) -> list[RetrievedChunk]:
        """Record the vector-search width and return the Chroma hits."""
        self.queries.append(n_results)
        return self.store.query_similar(embedding, n_results=n_results)

    def get_all(self) -> list[EmbeddedChunk]:
        """Record the keyword-index load and return every stored chunk."""
        self.get_all_calls += 1
        return self.store.get_all()


def _hit(section: str) -> RetrievedChunk:
    """Build a retrieved chunk whose id encodes the section."""
    return RetrievedChunk(
        chunk_id=f"hr-policy:v2.0:section-{section}",
        document="HR Policy",
        version="2.0",
        section=section,
        section_title="Rule",
        text="Employees follow this rule.",
        distance=0.2,
    )


def test_tokenizer_keeps_a_dotted_section_code_intact() -> None:
    """Keep 7.3 as one token so it does not match every rule numbered 7 or 3."""
    assert tokenize("See Section 7.3.") == ["see", "section", "7.3"]
    assert "7" not in tokenize("7.3")
    assert "3" not in tokenize("7.3")


def test_bm25_ranks_the_intact_section_code_first() -> None:
    """Match 7.3 as one token. A chunk that merely contains 7 and 3 does not take its place."""
    target = PolicyChunk(
        chunk_id="hr-policy:v2.0:section-7.3",
        document="HR Policy",
        version="2.0",
        section="7.3",
        section_title="Weekend Abandonment Consequence",
        text="Food left in the refrigerator is abandoned.",
    )
    lookalike = PolicyChunk(
        chunk_id="hr-policy:v2.0:section-7",
        document="HR Policy",
        version="2.0",
        section="7",
        section_title="Shared Refrigerator Policy",
        text="See item 3 and rule 7 today.",
    )
    unrelated = PolicyChunk(
        chunk_id="hr-policy:v2.0:section-1",
        document="HR Policy",
        version="2.0",
        section="1",
        section_title="Purpose",
        text="Employees receive a meal allowance.",
    )
    ranked = bm25_top("Section 7.3", [lookalike, unrelated, target], n=2)
    assert [chunk.section for chunk in ranked][0] == "7.3"


def test_rrf_weights_both_legs_equally() -> None:
    """A rank on the keyword leg adds the same 1/(60+rank) as that rank on the vector leg."""
    shared = _hit("7.3")
    vector_only = _hit("1")
    keyword_first = fuse_reciprocal_ranks([[vector_only], [shared, vector_only]])
    vector_first = fuse_reciprocal_ranks([[shared, vector_only], [vector_only]])
    assert [hit.section for hit in keyword_first] == [hit.section for hit in vector_first]
    assert keyword_first[0].section == "1"
    both = fuse_reciprocal_ranks([[shared, vector_only], [shared]])
    assert [hit.section for hit in both] == ["7.3", "1"]


def test_vector_search_does_not_read_the_keyword_index(tmp_path: Path) -> None:
    """Ask Chroma for 5 chunks and do not load the corpus for BM25."""
    store = _hr_store(tmp_path)
    recorded = RecordingStore(store)
    retrieve(
        "How long is the grace period?",
        store=recorded,
        embedder=FixedEmbedder([1.0, 0.0]),
        reranker=KeepingReranker(),
    )
    assert recorded.queries == [5]
    assert recorded.get_all_calls == 0


def test_hybrid_returns_section_7_3_when_vector_search_misses_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vector-only misses Section 7.3. Hybrid returns that subsection. Jev is not called."""
    monkeypatch.setattr(JevRouter, "choose", _refuse_jev)

    store = _hr_store(tmp_path)
    recorded = RecordingStore(store)
    embedder = FixedEmbedder([1.0, 0.0])
    keeper = KeepingReranker()
    vector_hits = retrieve(
        QUESTION,
        store=recorded,
        embedder=embedder,
        decision=RetrievalDecision(strategy="vector"),
        reranker=keeper,
    )
    hybrid_hits = retrieve(
        QUESTION,
        store=recorded,
        embedder=embedder,
        decision=RetrievalDecision(strategy="hybrid"),
        reranker=keeper,
    )

    assert recorded.queries == [5, 10]
    assert recorded.get_all_calls == 1
    assert SECTION not in [hit.citation_section for hit in vector_hits]
    assert SECTION in [hit.citation_section for hit in hybrid_hits]
    assert [len(shortlist) for shortlist in keeper.shortlists] == [5, 5]
    assert len(hybrid_hits) == 3


def test_ask_uses_the_injected_router(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The ask path asks the supplied router and does not call Jev."""
    monkeypatch.setattr(JevRouter, "choose", _refuse_jev)
    router = _VectorRouter()
    store = _hr_store(tmp_path)

    ask(
        "How long is the grace period?",
        store=store,
        embedder=FixedEmbedder([1.0, 0.0]),
        generator=_NotAnswerable(),
        router=router,
        reranker=KeepingReranker(),
    )

    assert router.questions == ["How long is the grace period?"]


class _VectorRouter:
    """Record the question and choose vector search."""

    def __init__(self) -> None:
        """Start with no questions seen."""
        self.questions: list[str] = []

    def choose(self, question: str) -> RetrievalDecision:
        """Record the question and return vector search."""
        self.questions.append(question)
        return RetrievalDecision(strategy="vector")


class _NotAnswerable:
    """Tell generation that the excerpt does not answer."""

    def complete(self, prompt: str) -> str:
        """Return one unanswerable model payload."""
        return '{"answerable": false, "answer": ""}'


def _refuse_jev(self: JevRouter, question: str) -> RetrievalDecision:
    """Fail the test when the comparison reaches Jev."""
    raise AssertionError("Jev")


def _hr_store(tmp_path: Path) -> ChromaPolicyStore:
    """Store the HR policy with Section 7.3 just outside the vector top 5."""
    chunks = chunk_policy_file(POLICIES_DIR / "hr-policy-v2.0.md")
    store = ChromaPolicyStore(tmp_path / "chroma")
    store.upsert_chunks(chunks, _embeddings_with_7_3_at_rank_six(chunks))
    return store


def _embeddings_with_7_3_at_rank_six(chunks: Sequence[PolicyChunk]) -> list[list[float]]:
    """Place five unrelated rules ahead of 7.3, so vector-only stops before it."""
    front = ["1", "3.1", "3.2", "4.1", "4.2"]
    by_section = {chunk.section: chunk for chunk in chunks}
    ordered = [by_section[section] for section in front]
    ordered.append(by_section["7.3"])
    ordered.extend(chunk for chunk in chunks if chunk.section not in {*front, "7.3"})
    rank = {chunk.chunk_id: index for index, chunk in enumerate(ordered)}
    return [[1.0, rank[chunk.chunk_id] * 0.05] for chunk in chunks]
