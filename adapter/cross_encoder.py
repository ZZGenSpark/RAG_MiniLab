"""Local cross-encoder implementation of the reranker port."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

from config import CROSS_ENCODER_MODEL
from rag.schema import RetrievedChunk


def _scores(predicted: object) -> list[float]:
    """Turn a predict result into one float per pair. A numpy array uses tolist."""
    tolist = getattr(predicted, "tolist", None)
    values = tolist() if callable(tolist) else predicted
    if isinstance(values, int | float):
        return [float(values)]
    if not isinstance(values, list):
        raise ValueError("reranker must return one score per chunk")
    return [float(value) for value in values]


class PairScorer(Protocol):
    """The predict method CrossEncoder provides."""

    def predict(self, inputs: Sequence[tuple[str, str]], **kwargs: object) -> object:
        """Return one score for each question-passage pair."""
        ...


class CrossEncoderReranker:
    """Rerank passages locally with ms-marco-MiniLM-L-6-v2.

    This is the only reranker. It scores the shortlist it is given and does
    not call a second model.
    """

    def __init__(self, model_name: str | None = None, model: PairScorer | None = None) -> None:
        """Configure the model name and an optional already-built scorer."""
        self.model_name = model_name or CROSS_ENCODER_MODEL
        self._model = model

    def score(self, question: str, chunks: Sequence[RetrievedChunk]) -> list[float]:
        """Return one cross-encoder score per chunk. Higher means more relevant."""
        if not chunks:
            return []
        pairs = [(question, chunk.text) for chunk in chunks]
        predicted = self._model_or_live().predict(pairs, show_progress_bar=False)
        scores = _scores(predicted)
        if len(scores) != len(chunks):
            raise ValueError("reranker must return one score per chunk")
        return scores

    def _model_or_live(self) -> PairScorer:
        """Load the cross-encoder the first time a shortlist is scored."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = cast(PairScorer, CrossEncoder(self.model_name))
        return self._model
