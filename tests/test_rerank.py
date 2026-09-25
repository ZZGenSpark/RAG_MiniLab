"""Rerank the shortlist of five once. A fake scorer can flip that order."""

from collections.abc import Sequence
from pathlib import Path

from adapter.chroma_store import ChromaPolicyStore
from adapter.cross_encoder import CrossEncoderReranker
from config import CROSS_ENCODER_MODEL
from rag.chunking import chunk_policy_file
from rag.retrieve import retrieve
from rag.route import RetrievalDecision
from rag.schema import RetrievedChunk
from tests.support import EXPENSE_POLICY_FIXTURE


class FlippingReranker:
    """Score the last chunk highest and record each shortlist."""

    def __init__(self) -> None:
        """Start with no scored shortlists."""
        self.shortlists: list[list[RetrievedChunk]] = []

    def score(self, question: str, chunks: Sequence[RetrievedChunk]) -> list[float]:
        """Record the shortlist and prefer the chunk that arrived last."""
        self.shortlists.append(list(chunks))
        return [float(index) for index in range(len(chunks))]


class FakeEmbedder:
    """Return one fixed query vector."""

    def __init__(self, vector: list[float]) -> None:
        """Store the vector used for the query."""
        self.vector = vector

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return the query vector once per text."""
        return [list(self.vector) for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return the query vector."""
        return list(self.vector)


class FakePairScorer:
    """Return scripted scores and record the pairs sent to the model."""

    def __init__(self, scores: list[float]) -> None:
        """Store the scores predict will return."""
        self.scores = scores
        self.pairs: list[Sequence[tuple[str, str]]] = []
        self.kwargs: list[dict[str, object]] = []

    def predict(self, inputs: Sequence[tuple[str, str]], **kwargs: object) -> list[float]:
        """Record the question-passage pairs and return the scripted scores."""
        self.pairs.append(list(inputs))
        self.kwargs.append(kwargs)
        return list(self.scores)


def test_fake_scorer_flips_the_shortlist_on_both_paths(tmp_path: Path) -> None:
    """Each path scores its five chunks once. The fake scorer moves the last chunk first."""
    chunks = chunk_policy_file(EXPENSE_POLICY_FIXTURE)
    store = ChromaPolicyStore(tmp_path / "chroma")
    store.upsert_chunks(
        chunks,
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
    )
    reranker = FlippingReranker()
    embedder = FakeEmbedder([0.95, 0.2, 0.1, 0.0, 0.0, 0.0])

    vector_hits = retrieve(
        "How much can I spend on food each day?",
        store=store,
        embedder=embedder,
        decision=RetrievalDecision(strategy="vector"),
        reranker=reranker,
    )
    hybrid_hits = retrieve(
        "How much can I spend on food each day?",
        store=store,
        embedder=embedder,
        decision=RetrievalDecision(strategy="hybrid"),
        reranker=reranker,
    )

    assert [len(shortlist) for shortlist in reranker.shortlists] == [5, 5]
    assert len(vector_hits) == 3
    assert len(hybrid_hits) == 3
    assert vector_hits[0].chunk_id == reranker.shortlists[0][-1].chunk_id
    assert hybrid_hits[0].chunk_id == reranker.shortlists[1][-1].chunk_id
    assert vector_hits[0].chunk_id != reranker.shortlists[0][0].chunk_id


def test_cross_encoder_scores_the_pairs_it_is_given() -> None:
    """Map chunk text to passage pairs. The injected scorer is the only model."""
    meals = _chunk("1", "Meals", "Employees may claim up to $65 per day for meals.")
    hotels = _chunk("2", "Hotels", "Hotels are reimbursable up to $225 per night.")
    scorer = FakePairScorer([0.1, 0.9])
    reranker = CrossEncoderReranker(model=scorer)

    scores = reranker.score("Where can I stay?", [meals, hotels])

    assert reranker.model_name == CROSS_ENCODER_MODEL
    assert scores == [0.1, 0.9]
    assert scorer.pairs == [[("Where can I stay?", meals.text), ("Where can I stay?", hotels.text)]]
    assert scorer.kwargs == [{"show_progress_bar": False}]
    assert reranker._model is scorer


def _chunk(section: str, title: str, text: str) -> RetrievedChunk:
    """Build one retrieved chunk for the adapter test."""
    return RetrievedChunk(
        chunk_id=f"employee-expense-policy:v2.0:section-{section}",
        document="Employee Expense Policy",
        version="2.0",
        section=section,
        section_title=title,
        text=text,
        distance=0.2,
    )
