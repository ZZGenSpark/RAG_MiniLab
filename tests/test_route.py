"""Check that routing chooses only vector or hybrid, without calling Jev."""

from collections.abc import Mapping
from types import SimpleNamespace

import pytest
from typesafe_sdk import Choice

from adapter.jev_router import JevRouter
from rag.route import (
    FallbackRouter,
    RetrievalDecision,
    Strategy,
    default_router,
    fallback_decision,
    route,
)


class FakeRouter:
    """Return one scripted strategy and record the question."""

    def __init__(self, strategy: Strategy) -> None:
        """Store the strategy this router will return."""
        self.strategy = strategy
        self.questions: list[str] = []

    def choose(self, question: str) -> RetrievalDecision:
        """Record the question and return the scripted decision."""
        self.questions.append(question)
        return RetrievalDecision(strategy=self.strategy)


class FakeSystemOne:
    """Return one scripted TypeSafe choice and record the request."""

    def __init__(self, choice: str) -> None:
        """Store the label the fake System One call will select."""
        self.choice = choice
        self.states: list[object] = []
        self.questions: list[Mapping[str, object]] = []

    def system_one(
        self,
        state: str | Mapping[str, str],
        questions: Mapping[str, object],
    ) -> SimpleNamespace:
        """Record the request and return the scripted choice."""
        self.states.append(state)
        self.questions.append(questions)
        return SimpleNamespace(choices={"strategy": SimpleNamespace(choice=self.choice)})


def test_fallback_selects_hybrid_only_for_a_section_code() -> None:
    """A section code selects hybrid. Every other question selects vector."""
    assert fallback_decision("What does Section 7.3 say about leftover food?").strategy == "hybrid"
    assert fallback_decision("Explain 7.3.").strategy == "hybrid"
    assert fallback_decision("What does section 6 require?").strategy == "hybrid"
    assert fallback_decision("How long can an employee play foosball each day?").strategy == "vector"
    assert fallback_decision("What happens to food left in the shared refrigerator over the weekend?").strategy == (
        "vector"
    )
    assert fallback_decision("What changed in version 2.0?").strategy == "vector"


def test_decision_is_only_vector_or_hybrid() -> None:
    """Reject a score, a weight, and any strategy other than the two names."""
    assert set(RetrievalDecision.model_fields) == {"strategy"}
    with pytest.raises(ValueError):
        RetrievalDecision.model_validate({"strategy": "vector", "keyword_weight": 0.5})
    with pytest.raises(ValueError):
        RetrievalDecision.model_validate({"strategy": "lexical"})


def test_route_uses_the_injected_router() -> None:
    """A fake router decides, including for a question the fallback would send to vector."""
    router = FakeRouter("hybrid")
    decision = route("How long can an employee play foosball each day?", router=router)
    assert decision.strategy == "hybrid"
    assert router.questions == ["How long can an employee play foosball each day?"]


def test_route_uses_the_fallback_when_the_key_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty key selects the section-code rule and does not construct Jev."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "  ")
    assert fallback_decision("Section 7.3").strategy == route("Section 7.3").strategy == "hybrid"
    assert route("How long can an employee play foosball each day?").strategy == "vector"
    assert isinstance(default_router(), FallbackRouter)


def test_default_router_is_jev_when_the_key_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """A present key selects the Jev router. Constructing it does not call the API."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    assert isinstance(default_router(), JevRouter)


def test_jev_router_fills_the_decision_from_one_choice() -> None:
    """Map Jev's single strategy choice onto the decision. Do not send a score."""
    client = FakeSystemOne("vector")
    decision = JevRouter(client=client).choose("How long can an employee play foosball each day?")
    assert decision == RetrievalDecision(strategy="vector")
    assert client.states == [{"question": "How long can an employee play foosball each day?"}]
    question = client.questions[0]["strategy"]
    assert isinstance(question, Choice)
    assert set(question.criteria) == {"vector", "hybrid"}


def test_jev_router_rejects_a_choice_outside_the_two_strategies() -> None:
    """Refuse a label that is not vector or hybrid."""
    with pytest.raises(ValueError, match="vector or hybrid"):
        JevRouter(client=FakeSystemOne("lexical")).choose("Section 7.3")


def test_route_rejects_an_empty_question() -> None:
    """Refuse a blank question before choosing a strategy."""
    with pytest.raises(ValueError, match="question must not be empty"):
        route("   ")
