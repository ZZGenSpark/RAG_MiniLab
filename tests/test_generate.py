"""Test grounded answers that cite every source used in one prompt."""

from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from adapter.chroma_store import ChromaPolicyStore
from adapter.ollama_chat import OllamaChatAdapter
from config import CHAT_TEMPERATURE
from prompt.grounded_excerpt import INSTRUCTION, build_grounded_excerpt_prompt
from rag.ask import ask
from rag.generate import REFUSAL_ANSWER, build_prompt, generate_answer
from rag.route import FallbackRouter, RetrievalDecision, Strategy
from rag.schema import PolicyChunk, RetrievedChunk
from tests.support import KeepingReranker


def _chunk(
    section: str,
    title: str,
    text: str,
    distance: float,
) -> RetrievedChunk:
    """Build a retrieved HR chunk."""
    return RetrievedChunk(
        chunk_id=f"hr-policy:v2.0:section-{section}",
        document="HR Policy",
        version="2.0",
        section=section,
        section_title=title,
        text=text,
        distance=distance,
    )


JOKE = _chunk(
    "3.1",
    "Requirement",
    "Every email must begin or end with a joke.",
    0.08,
)
LEAVE = _chunk(
    "5.1",
    "Leave Entitlement",
    "Employees receive 5 days of paid leave when they adopt a pet.",
    0.21,
)
GRACE = _chunk(
    "6",
    "Boss Error Grace Period",
    "Employees must wait 30 minutes before correcting a boss.",
    0.41,
)


def _answer(answer: str, sources: list[int], *, answerable: bool = True) -> str:
    """Build one model payload."""
    return (
        '{"answerable": '
        + str(answerable).lower()
        + ', "answer": "'
        + answer
        + '", "sources": '
        + str(sources).replace("'", "")
        + "}"
    )


class FakeGenerator:
    """Return one scripted response and record the prompt."""

    def __init__(self, response: str) -> None:
        """Store the only response this generator returns."""
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        """Record the prompt and return the scripted response."""
        self.prompts.append(prompt)
        return self.response


class FakeEmbedder:
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return a fixed unit vector for each input text."""
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return the embedding vector for a single question."""
        return self.embed_texts([text])[0]


class ChoosingRouter:
    """Return one scripted retrieval strategy."""

    def __init__(self, strategy: Strategy) -> None:
        """Store the strategy."""
        self.decision = RetrievalDecision(strategy=strategy)

    def choose(self, question: str) -> RetrievalDecision:
        """Return the scripted decision."""
        return self.decision


def test_one_prompt_cites_every_source_the_model_uses() -> None:
    """Send every final chunk once and cite each excerpt the answer names."""
    chat = FakeGenerator(
        _answer("Emails need a joke, and a correction waits 30 minutes.", [1, 3]),
    )

    answer, citations = generate_answer(
        "What are the joke and correction rules?",
        [JOKE, LEAVE, GRACE],
        generator=chat,
    )

    assert "joke" in answer
    assert [citation.section for citation in citations] == ["3.1 Requirement", "6. Boss Error Grace Period"]
    assert {citation.document for citation in citations} == {"HR Policy"}
    assert len(chat.prompts) == 1
    assert "Every email must begin or end with a joke." in chat.prompts[0]
    assert "Employees receive 5 days of paid leave" in chat.prompts[0]
    assert "Employees must wait 30 minutes" in chat.prompts[0]


def test_refusal_uses_one_prompt_and_cites_nothing() -> None:
    """Refuse unsupported questions without a citation after one prompt."""
    chat = FakeGenerator('{"answerable": false, "answer": "", "sources": []}')
    answer, citations = generate_answer(
        "Does the company match retirement contributions?",
        [JOKE, LEAVE, GRACE],
        generator=chat,
    )
    assert answer == REFUSAL_ANSWER
    assert citations == []
    assert len(chat.prompts) == 1


def test_named_source_must_support_the_numbers_in_the_answer() -> None:
    """Refuse an answer whose number is absent from the excerpts it cites."""
    chat = FakeGenerator(_answer("A correction waits 30 minutes.", [1]))
    answer, citations = generate_answer(
        "How long must I wait before correcting a boss?",
        [JOKE, GRACE],
        generator=chat,
    )
    assert answer == REFUSAL_ANSWER
    assert citations == []


def test_number_from_the_question_is_allowed_when_that_excerpt_is_cited() -> None:
    """Allow a number that appears in the question when the supporting excerpt is cited."""
    chat = FakeGenerator(_answer("Wait 30 minutes before the correction.", [2]))
    answer, citations = generate_answer(
        "How long is the 30 minute grace period?",
        [JOKE, GRACE],
        generator=chat,
    )
    assert citations[0].section == "6. Boss Error Grace Period"
    assert "30" in answer


def test_prompt_lists_every_final_excerpt() -> None:
    """Put the instruction, question, and every chunk into the one prompt."""
    prompt = build_prompt("What are the joke and leave rules?", [JOKE, LEAVE])
    assert prompt == build_grounded_excerpt_prompt(
        "What are the joke and leave rules?",
        [
            "\n".join(
                [
                    "Excerpt 1",
                    "Document: HR Policy",
                    "Version: 2.0",
                    "Section: 3.1 Requirement",
                    JOKE.text,
                ]
            ),
            "\n".join(
                [
                    "Excerpt 2",
                    "Document: HR Policy",
                    "Version: 2.0",
                    "Section: 5.1 Leave Entitlement",
                    LEAVE.text,
                ]
            ),
        ],
    )
    assert INSTRUCTION in prompt
    assert "Question: What are the joke and leave rules?" in prompt


def test_fenced_json_is_parsed_as_the_model_answer() -> None:
    """Accept a JSON object wrapped in a markdown fence."""
    chat = FakeGenerator(
        '```json\n{"answerable": true, "answer": "Every email must include a joke.", "sources": [1]}\n```'
    )
    answer, citations = generate_answer("Does every company email have to include a joke?", [JOKE], generator=chat)
    assert "joke" in answer
    assert citations[0].section == "3.1 Requirement"


def test_empty_retrieval_refuses_without_calling_the_model() -> None:
    """Refuse immediately when retrieval returns no chunks."""

    class ExplodingChat:
        def complete(self, prompt: str) -> str:
            """Fail if generation runs with no retrieved excerpts."""
            raise AssertionError("model should not run without retrieved excerpts")

    answer, citations = generate_answer(
        "Does the company match retirement contributions?",
        [],
        generator=ExplodingChat(),
    )
    assert answer == REFUSAL_ANSWER
    assert citations == []


def test_unparseable_model_text_is_not_cited() -> None:
    """Refuse when the model returns text that is not the expected JSON."""
    chat = FakeGenerator("Every email must include a joke.")
    answer, citations = generate_answer("Does every company email have to include a joke?", [JOKE], generator=chat)
    assert answer == REFUSAL_ANSWER
    assert citations == []


def test_ask_returns_every_citation_and_the_strategy(tmp_path) -> None:
    """Return each source the model used and the strategy chosen for the ask."""
    store = _store(tmp_path)
    chat = FakeGenerator(_answer("Every email must include a joke.", [1]))

    response = ask(
        "Does every company email have to include a joke?",
        store=store,
        embedder=FakeEmbedder(),
        generator=chat,
        router=ChoosingRouter("hybrid"),
        reranker=KeepingReranker(),
        audit_path=tmp_path / "audit.jsonl",
    )

    assert [citation.section for citation in response.citations] == ["3.1 Requirement"]
    assert response.retrieval.strategy == "hybrid"
    assert len(chat.prompts) == 1
    assert "1. Meals" not in chat.prompts[0]


def test_ask_refusal_keeps_the_retrieved_chunks(tmp_path) -> None:
    """Return the refusal, no citations, and the retrieved excerpts."""
    store = _store(tmp_path)
    chat = FakeGenerator('{"answerable": false, "answer": "", "sources": []}')

    response = ask(
        "Does the company match retirement contributions?",
        store=store,
        embedder=FakeEmbedder(),
        generator=chat,
        router=FallbackRouter(),
        reranker=KeepingReranker(),
        audit_path=tmp_path / "audit.jsonl",
    )

    assert response.answer == REFUSAL_ANSWER
    assert response.citations == []
    assert response.retrieval.strategy == "vector"
    assert {chunk.section for chunk in response.retrieved_chunks} == {
        "3.1 Requirement",
        "5.1 Leave Entitlement",
        "6. Boss Error Grace Period",
    }


def test_ollama_chat_adapter_returns_message_text() -> None:
    """Read the Ollama message body and return it as plain text."""

    class RecordingClient:
        def chat(self, model: str, messages: list, **kwargs):
            """Return one scripted chat message and record the call."""
            self.messages = messages
            self.kwargs = kwargs
            return SimpleNamespace(
                message=SimpleNamespace(content='{"answerable": true, "answer": "joke", "sources": [1]}'),
            )

    client = RecordingClient()
    text = OllamaChatAdapter(model="qwen3:8b", client=client).complete("Question: email")

    assert text == '{"answerable": true, "answer": "joke", "sources": [1]}'
    assert client.messages == [{"role": "user", "content": "Question: email"}]
    assert client.kwargs["think"] is False
    assert client.kwargs["options"] == {"temperature": CHAT_TEMPERATURE}
    assert "answerable" in client.kwargs["format"]["properties"]
    assert "sources" in client.kwargs["format"]["properties"]


def test_ollama_chat_adapter_reads_a_dict_response() -> None:
    """Read message text when the client returns a plain dictionary."""

    class DictClient:
        def chat(self, model: str, messages: list, **kwargs):
            """Return one scripted dictionary response."""
            return {"message": {"content": '{"answerable": false, "answer": "", "sources": []}'}}

    text = OllamaChatAdapter(client=DictClient()).complete("Question: retirement")
    assert text == '{"answerable": false, "answer": "", "sources": []}'


def test_ollama_chat_adapter_rejects_an_empty_response() -> None:
    """Reject a chat response that has no message text."""

    class EmptyClient:
        def chat(self, model: str, messages: list, **kwargs):
            """Return a message with no content."""
            return SimpleNamespace(message=SimpleNamespace(content=""))

    with pytest.raises(ValueError, match="empty response"):
        OllamaChatAdapter(client=EmptyClient()).complete("Question: email")


def _store(tmp_path) -> ChromaPolicyStore:
    """Store the three HR excerpts used by the ask tests."""
    store = ChromaPolicyStore(tmp_path / "chroma")
    store.upsert_chunks(
        [_as_policy_chunk(JOKE), _as_policy_chunk(LEAVE), _as_policy_chunk(GRACE)],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    return store


def _as_policy_chunk(chunk: RetrievedChunk) -> PolicyChunk:
    """Drop retrieval distance so a chunk can be stored."""
    return PolicyChunk(
        chunk_id=chunk.chunk_id,
        document=chunk.document,
        version=chunk.version,
        section=chunk.section,
        section_title=chunk.section_title,
        text=chunk.text,
    )
