"""Score retrieval choices on the policy questions and write comparison tables.

A labeled question is recalled when every expected section is in the top 3.
Retrieval scoring does not call Ollama. The router table does. The run does
not change ingest or ask.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from adapter.chroma_store import ChromaPolicyStore
from adapter.cross_encoder import CrossEncoderReranker
from adapter.sentence_transformer_embeddings import SentenceTransformerEmbeddingAdapter
from config import CHAT_MODEL, CHAT_TEMPERATURE, OLLAMA_HOST, REPO_ROOT, typesafe_api_key
from rag.embeddings import Embedder
from rag.eval import RETRIEVAL_CASES, RetrievalCase, case_recalled
from rag.ingest import ingest_corpus
from rag.rerank import Reranker
from rag.retrieve import retrieve
from rag.route import FallbackRouter, RetrievalDecision, Router, Strategy, question_names_section
from rag.schema import RetrievedChunk
from rag.store import PolicyStore

_OUTPUT = REPO_ROOT / "docs" / "comparison_tables.md"
_STRATEGY_LABEL = re.compile(r"\b(vector|hybrid)\b", re.IGNORECASE)
_QWEN_INSTRUCTIONS = (
    "Choose hybrid when the question names a policy section code such as Section 7.3. "
    "Choose vector for every other question.\n"
    "vector: Search stored policy text by semantic similarity.\n"
    "hybrid: Search by semantic similarity and by keywords, including a section code.\n"
    "Reply with one word: vector or hybrid."
)


@dataclass(frozen=True)
class Recall:
    """How many labeled questions retrieved every expected section."""

    hit: int
    total: int
    missed: tuple[str, ...]

    def render(self) -> str:
        """Return the score as a fraction."""
        return f"{self.hit}/{self.total}"


@dataclass(frozen=True)
class RouteTiming:
    """How often a router matched the section-code rule, and how long it took."""

    name: str
    section_hybrid: int
    section_total: int
    other_vector: int
    other_total: int
    mean_ms: float
    differed: tuple[str, ...]


class ConstantRouter:
    """Return the same strategy for every question."""

    def __init__(self, strategy: Strategy) -> None:
        """Store the strategy this router always chooses."""
        self._strategy = strategy

    def choose(self, question: str) -> RetrievalDecision:
        """Return the stored strategy."""
        if not question.strip():
            raise ValueError("question must not be empty")
        return RetrievalDecision(strategy=self._strategy)


class QwenRouter:
    """Ask Qwen for the same vector-or-hybrid choice Jev is given."""

    def __init__(self) -> None:
        """Open the local Ollama chat client used for answers."""
        import ollama

        self._client = ollama.Client(host=OLLAMA_HOST)

    def choose(self, question: str) -> RetrievalDecision:
        """Return vector or hybrid from one Qwen reply."""
        if not question.strip():
            raise ValueError("question must not be empty")
        response = self._client.chat(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": f"{_QWEN_INSTRUCTIONS}\n\nQuestion: {question}"}],
            think=False,
            options={"temperature": CHAT_TEMPERATURE},
        )
        text = _chat_text(response)
        label = _one_strategy(text)
        if label is None:
            raise ValueError("reply was not vector or hybrid")
        return RetrievalDecision(strategy=label)


class ShortlistOrderReranker:
    """Keep the shortlist in the order retrieval already produced."""

    def score(self, question: str, chunks: Sequence[RetrievedChunk]) -> list[float]:
        """Give the earliest chunk the highest score so sorting keeps this order."""
        if not question.strip():
            raise ValueError("question must not be empty")
        total = len(chunks)
        return [float(total - index) for index in range(total)]


def recall_of(
    cases: Sequence[RetrievalCase],
    *,
    store: PolicyStore,
    embedder: Embedder,
    router: Router,
    reranker: Reranker,
) -> Recall:
    """Score labeled questions. A question counts when every expected section is in the top 3."""
    labeled = [case for case in cases if case.expected_labels]
    missed: list[str] = []
    hits = 0
    for case in labeled:
        chunks = retrieve(
            case.question,
            store=store,
            embedder=embedder,
            decision=router.choose(case.question),
            reranker=reranker,
        )
        if case_recalled(case, chunks):
            hits += 1
        else:
            missed.append(case.question)
    return Recall(hit=hits, total=len(labeled), missed=tuple(missed))


def run_comparisons() -> str:
    """Ingest the numbered-rule index, score the questions, and return the markdown."""
    total = len(RETRIEVAL_CASES)
    labeled = sum(1 for case in RETRIEVAL_CASES if case.expected_labels)
    embedder = SentenceTransformerEmbeddingAdapter()
    reranker = CrossEncoderReranker()
    with tempfile.TemporaryDirectory(prefix="compare-rules-", ignore_cleanup_errors=True) as rules_name:
        rules = ChromaPolicyStore(rules_name)
        _log("Ingesting numbered rules")
        ingest_corpus(store=rules, embedder=embedder)
        vector = _score(
            "Always vector",
            RETRIEVAL_CASES,
            store=rules,
            embedder=embedder,
            router=ConstantRouter("vector"),
            reranker=reranker,
        )
        hybrid = _score(
            "Always hybrid",
            RETRIEVAL_CASES,
            store=rules,
            embedder=embedder,
            router=ConstantRouter("hybrid"),
            reranker=reranker,
        )
        routed = _score(
            "Section-code router",
            RETRIEVAL_CASES,
            store=rules,
            embedder=embedder,
            router=FallbackRouter(),
            reranker=reranker,
        )
        rerank_off = _score(
            "Reranker off",
            RETRIEVAL_CASES,
            store=rules,
            embedder=embedder,
            router=FallbackRouter(),
            reranker=ShortlistOrderReranker(),
        )
    return _render(total, labeled, vector, hybrid, routed, rerank_off, _router_section())


def main() -> None:
    """Write the comparison tables and print them."""
    parser = argparse.ArgumentParser(
        description="Score retrieval choices on the policy questions and write comparison tables."
    )
    parser.add_argument(
        "--output", type=Path, default=_OUTPUT, help="Markdown file to write. Defaults to docs/comparison_tables.md."
    )
    args = parser.parse_args()
    report = run_comparisons()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report, end="")
    _log(f"Wrote {args.output}")


def _score(
    label: str,
    cases: Sequence[RetrievalCase],
    *,
    store: PolicyStore,
    embedder: Embedder,
    router: Router,
    reranker: Reranker,
) -> Recall:
    """Score one setup and print the fraction while the run is still going."""
    _log(f"Scoring {label}")
    scored = recall_of(cases, store=store, embedder=embedder, router=router, reranker=reranker)
    _log(f"{label}: {scored.render()}")
    return scored


def _router_section() -> str:
    """Time the section-code rule, Jev, and Qwen on the same choice."""
    notes: list[str] = []
    rows = [_time_router(FallbackRouter(), "Section-code rule")]
    if typesafe_api_key():
        _log("Calling Jev")
        try:
            from adapter.jev_router import JevRouter

            rows.append(_time_router(JevRouter(), "Jev"))
        except Exception as exc:
            notes.append(f"Jev was not called because the request failed ({type(exc).__name__}).")
    else:
        notes.append("Jev was not called because TYPESAFE_API_KEY is unset.")
    _log("Calling Qwen")
    try:
        rows.append(_time_router(QwenRouter(), "Qwen"))
    except Exception as exc:
        notes.append(f"Qwen was not called because the request failed ({type(exc).__name__}).")
    return _route_timing_table(rows, notes)


def _time_router(router: Router, name: str) -> RouteTiming:
    """Count agreement with the section-code rule and the mean choose time."""
    section_hybrid = 0
    other_vector = 0
    elapsed = 0.0
    differed: list[str] = []
    section_total = 0
    other_total = 0
    for case in RETRIEVAL_CASES:
        named = question_names_section(case.question)
        started = time.perf_counter()
        try:
            decision = router.choose(case.question)
        except ValueError:
            elapsed += time.perf_counter() - started
            differed.append(f"{case.question} -> unparsed")
            if named:
                section_total += 1
            else:
                other_total += 1
            continue
        elapsed += time.perf_counter() - started
        if named:
            section_total += 1
            if decision.strategy == "hybrid":
                section_hybrid += 1
            else:
                differed.append(f"{case.question} -> {decision.strategy}")
        else:
            other_total += 1
            if decision.strategy == "vector":
                other_vector += 1
            else:
                differed.append(f"{case.question} -> {decision.strategy}")
    mean_ms = elapsed / len(RETRIEVAL_CASES) * 1000
    return RouteTiming(
        name,
        section_hybrid,
        section_total,
        other_vector,
        other_total,
        mean_ms,
        tuple(differed),
    )


def _render(
    total: int,
    labeled: int,
    vector: Recall,
    hybrid: Recall,
    routed: Recall,
    rerank_off: Recall,
    routers: str,
) -> str:
    """Build the markdown document from the measured scores."""
    parts = [
        "# Comparison tables",
        (
            "A labeled question is recalled when every expected section is in the top 3. "
            f"{labeled} of the {total} questions have an expected section."
        ),
        _block(
            "Route",
            "Same numbered-rule index, MiniLM, and cross-encoder.",
            _table(
                ("Setup", "Recalled"),
                (
                    ("Always vector", vector.render()),
                    ("Always hybrid", hybrid.render()),
                    ("Section-code router", routed.render()),
                ),
            ),
            _misses((("always vector", vector), ("always hybrid", hybrid), ("the section-code router", routed))),
        ),
        _block(
            "Reranker",
            (
                "Section-code router on the numbered-rule index. "
                "Off keeps the shortlist order. On scores it with the cross-encoder."
            ),
            _table(("Reranker", "Recalled"), (("Off", rerank_off.render()), ("On", routed.render()))),
            _misses((("reranker off", rerank_off), ("reranker on", routed))),
        ),
    ]
    parts.append(routers if routers.startswith("##") else f"## Jev\n\n{routers}")
    parts.append("Produced by `scripts/compare_retrieval.py`.")
    return "\n\n".join(parts).rstrip() + "\n"


def _route_timing_table(rows: Sequence[RouteTiming], notes: Sequence[str]) -> str:
    """Render the router comparison, including any question whose label differed."""
    intro = f"Each router chooses vector or hybrid. Qwen is {CHAT_MODEL}."
    body = _table(
        ("Router", "Section questions hybrid", "Other questions vector", "Mean call time"),
        tuple(
            (
                row.name,
                f"{row.section_hybrid}/{row.section_total}",
                f"{row.other_vector}/{row.other_total}",
                f"{row.mean_ms:.1f} ms",
            )
            for row in rows
        ),
    )
    extra = list(notes)
    for row in rows:
        if not row.differed:
            continue
        lines = "\n".join(f"- {item}" for item in row.differed)
        extra.append(f"{row.name} chose a different strategy for:\n\n{lines}")
    tail = "\n\n".join(extra)
    if tail:
        return f"## Jev\n\n{intro}\n\n{body}\n\n{tail}"
    return f"## Jev\n\n{intro}\n\n{body}"


def _block(title: str, intro: str, table: str, misses: str) -> str:
    """One section: a sentence, a table, and the questions that missed."""
    body = f"## {title}\n\n{intro}\n\n{table}"
    if misses:
        body = f"{body}\n\n{misses}"
    return body


def _table(headers: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> str:
    """Render a markdown table."""
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def _misses(rows: Sequence[tuple[str, Recall]]) -> str:
    """Name the questions a setup missed. An empty string means every row was complete."""
    blocks: list[str] = []
    for name, scored in rows:
        if not scored.missed:
            continue
        items = "\n".join(f"- {question}" for question in scored.missed)
        blocks.append(f"Missed by {name}:\n\n{items}")
    return "\n\n".join(blocks)


def _chat_text(response: object) -> str:
    """Return the text from an Ollama chat response."""
    content = getattr(response, "message", None)
    text = getattr(content, "content", None) if content is not None else None
    if not text and isinstance(response, dict):
        message = response.get("message", {})
        text = message.get("content") if isinstance(message, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise ValueError("reply was not vector or hybrid")
    return text


def _one_strategy(text: str) -> Strategy | None:
    """Return the single vector or hybrid label in a reply."""
    labels = {match.group(1).lower() for match in _STRATEGY_LABEL.finditer(text)}
    if labels == {"vector"}:
        return "vector"
    if labels == {"hybrid"}:
        return "hybrid"
    return None


def _log(message: str) -> None:
    """Print progress without mixing it into the table document."""
    print(message, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
