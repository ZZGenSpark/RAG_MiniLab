"""Test grounded answers, fallbacks, and refusals with fake models."""

from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from adapter.chroma_store import ChromaPolicyStore
from adapter.ollama_chat import OllamaChatAdapter
from config import CHAT_TEMPERATURE
from prompt.grounded_excerpt import INSTRUCTION, build_grounded_excerpt_prompt
from rag.ask import ask
from rag.generate import REFUSAL_ANSWER, build_prompt, generate_answer
from rag.schema import PolicyChunk, RetrievedChunk


def _chunk(
    section: str,
    title: str,
    text: str,
    distance: float,
) -> RetrievedChunk:
    """Build a retrieved chunk fixture for generation tests."""
    return RetrievedChunk(
        chunk_id=f"employee-expense-policy:v2.0:section-{section}",
        document="Employee Expense Policy",
        version="2.0",
        section=section,
        section_title=title,
        text=text,
        distance=distance,
    )


MEALS = _chunk(
    "1",
    "Meals",
    "Employees may claim up to $65 per day for meals while traveling overnight.\nAlcohol is not reimbursable.",
    0.08,
)
HOTELS = _chunk(
    "2",
    "Hotels",
    "Hotels are reimbursable up to $225 per night.\nA manager must approve higher rates before booking.",
    0.21,
)
DEADLINE = _chunk(
    "6",
    "Submission Deadline",
    "Expense reports must be submitted within 30 days after travel ends.",
    0.41,
)


class FakeGenerator:
    """Returns one response per call, in order. The last response repeats if
    more calls happen than responses were supplied."""

    def __init__(self, *responses: str) -> None:
        """Store the scripted responses in call order."""
        self.responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        """Record the prompt and return the next scripted response."""
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self.responses) - 1)
        return self.responses[index]


class FakeEmbedder:
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return a fixed unit vector for each input text."""
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return the embedding vector for a single question."""
        return self.embed_texts([text])[0]


NOT_ANSWERABLE = '{"answerable": false, "answer": ""}'


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


def _store_with_meals(tmp_path) -> ChromaPolicyStore:
    """Create a store containing the meals, hotels, and deadline fixtures."""
    store = ChromaPolicyStore(tmp_path / "chroma")
    store.upsert_chunks(
        [_as_policy_chunk(MEALS), _as_policy_chunk(HOTELS), _as_policy_chunk(DEADLINE)],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    return store


def test_closest_chunk_answers_in_a_single_call() -> None:
    """Cite the closest chunk when it alone answers the question."""
    chat = FakeGenerator('{"answerable": true, "answer": "Employees may claim up to $65 per day for meals."}')

    answer, citation = generate_answer(
        "How much can I spend on food each day?",
        [MEALS, HOTELS, DEADLINE],
        generator=chat,
    )

    assert "$65" in answer
    assert citation is not None
    assert citation.document == "Employee Expense Policy"
    assert citation.version == "2.0"
    assert citation.section == "1. Meals"
    # Only the closest chunk needed to be checked.
    assert len(chat.prompts) == 1


def test_falls_back_to_next_closest_chunk_when_first_cannot_answer() -> None:
    """Use the next chunk when the closest one cannot answer."""
    chat = FakeGenerator(
        NOT_ANSWERABLE,
        '{"answerable": true, "answer": "A manager must approve rates above $225 per night."}',
    )

    answer, citation = generate_answer(
        "My hotel costs $250. What do I need?",
        [MEALS, HOTELS, DEADLINE],
        generator=chat,
    )

    assert "manager" in answer.lower()
    assert citation is not None
    assert citation.section == "2. Hotels"
    # First (closest) chunk was tried and rejected before falling back.
    assert len(chat.prompts) == 2
    assert "1. Meals" in chat.prompts[0]
    assert "2. Hotels" in chat.prompts[1]


def test_unsupported_question_refuses_without_citation() -> None:
    """Refuse without a citation when no chunk can answer."""
    chat = FakeGenerator(NOT_ANSWERABLE)
    answer, citation = generate_answer(
        "Does the company reimburse professional conference tickets?",
        [MEALS, HOTELS, DEADLINE],
        generator=chat,
    )
    assert answer == REFUSAL_ANSWER
    assert citation is None
    # Every chunk was tried before refusing.
    assert len(chat.prompts) == 3


def test_gym_membership_is_an_unsupported_question_example() -> None:
    """Refuse the gym-membership question when no excerpt supports it."""
    chat = FakeGenerator(NOT_ANSWERABLE)
    answer, citation = generate_answer(
        "Does the company reimburse gym memberships?",
        [MEALS, HOTELS, DEADLINE],
        generator=chat,
    )
    assert answer == REFUSAL_ANSWER
    assert citation is None
    assert len(chat.prompts) == 3


def test_fenced_json_is_parsed_as_the_model_answer() -> None:
    """Accept a JSON object wrapped in a markdown fence."""
    chat = FakeGenerator(
        '```json\n{"answerable": true, "answer": "Employees may claim up to $65 per day for meals."}\n```'
    )
    answer, citation = generate_answer(
        "How much can I spend on food each day?",
        [MEALS],
        generator=chat,
    )
    assert "$65" in answer
    assert citation is not None
    assert citation.section == "1. Meals"


def test_answerable_flag_with_an_empty_answer_is_not_cited() -> None:
    """Refuse a chunk the model marks answerable but leaves blank."""
    chat = FakeGenerator('{"answerable": true, "answer": "   "}')
    answer, citation = generate_answer(
        "How much can I spend on food each day?",
        [MEALS],
        generator=chat,
    )
    assert answer == REFUSAL_ANSWER
    assert citation is None


def test_number_from_the_question_is_allowed_in_the_answer() -> None:
    """Allow an amount that appears in the question even when the excerpt uses another amount."""
    chat = FakeGenerator('{"answerable": true, "answer": "A manager must approve a hotel that costs $250."}')
    answer, citation = generate_answer(
        "My hotel costs $250. What do I need?",
        [HOTELS],
        generator=chat,
    )
    assert citation is not None
    assert citation.section == "2. Hotels"
    assert "$250" in answer


def test_number_missing_from_excerpt_and_question_is_not_cited() -> None:
    """Skip an answer that introduces a number present in neither the excerpt nor the question."""
    chat = FakeGenerator('{"answerable": true, "answer": "You need approval from 9 managers."}')
    answer, citation = generate_answer(
        "My hotel costs $250. What do I need?",
        [HOTELS],
        generator=chat,
    )
    assert answer == REFUSAL_ANSWER
    assert citation is None


def test_prompt_quotes_only_the_selected_excerpt() -> None:
    """Put the instruction, question, section label, and excerpt text in the prompt."""
    prompt = build_prompt("How much can I spend on food each day?", MEALS)
    assert prompt == build_grounded_excerpt_prompt(
        "How much can I spend on food each day?",
        "1. Meals",
        MEALS.text,
    )
    assert INSTRUCTION in prompt
    assert "Question: How much can I spend on food each day?" in prompt
    assert "[Section 1. Meals]" in prompt
    assert MEALS.text in prompt
    assert "Hotels" not in prompt


def test_unparseable_model_text_is_not_cited() -> None:
    """Refuse when the model returns text that is not the expected JSON."""
    chat = FakeGenerator("Employees may claim up to $65 per day for meals.")
    answer, citation = generate_answer(
        "How much can I spend on food each day?",
        [MEALS],
        generator=chat,
    )
    assert answer == REFUSAL_ANSWER
    assert citation is None


def test_answer_that_introduces_an_amount_is_not_cited() -> None:
    """Skip a chunk whose answer uses an amount missing from the excerpt and question."""
    chat = FakeGenerator(
        '{"answerable": true, "answer": "Employees may claim up to $80 per day for meals."}',
        '{"answerable": true, "answer": "A manager must approve rates above $225 per night."}',
    )

    answer, citation = generate_answer(
        "How much can I spend on food each day?",
        [MEALS, HOTELS],
        generator=chat,
    )

    assert citation is not None
    assert citation.section == "2. Hotels"
    assert "$225" in answer
    assert len(chat.prompts) == 2


def test_empty_retrieval_refuses_without_calling_the_model() -> None:
    """Refuse immediately when retrieval returns no chunks."""

    class ExplodingChat:
        def complete(self, prompt: str) -> str:
            """Fail if generation runs with no retrieved excerpts."""
            raise AssertionError("model should not run without retrieved excerpts")

    answer, citation = generate_answer(
        "Does the company reimburse gym memberships?",
        [],
        generator=ExplodingChat(),
    )
    assert answer == REFUSAL_ANSWER
    assert citation is None


def test_ask_builds_structured_response_from_retrieval_not_the_model(tmp_path) -> None:
    """Build the cited response from retrieved chunks, not from the model JSON."""
    store = _store_with_meals(tmp_path)
    chat = FakeGenerator('{"answerable": true, "answer": "Employees may claim up to $65 per day for meals."}')

    response = ask(
        "How much can I spend on food each day?",
        store=store,
        embedder=FakeEmbedder(),
        generator=chat,
    )

    assert response.citation is not None
    assert response.citation.section == "1. Meals"
    assert len(response.retrieved_chunks) <= 3
    assert response.retrieved_chunks == sorted(response.retrieved_chunks, key=lambda chunk: chunk.distance)
    assert all(isinstance(chunk.distance, float) for chunk in response.retrieved_chunks)
    assert "1. Meals" in {chunk.section for chunk in response.retrieved_chunks}


def test_ask_refusal_keeps_the_retrieved_chunks(tmp_path) -> None:
    """Return the refusal and the retrieved excerpts when no chunk can answer."""
    store = _store_with_meals(tmp_path)
    chat = FakeGenerator(NOT_ANSWERABLE)

    response = ask(
        "Does the company reimburse gym memberships?",
        store=store,
        embedder=FakeEmbedder(),
        generator=chat,
    )

    assert response.answer == REFUSAL_ANSWER
    assert response.citation is None
    assert [chunk.section for chunk in response.retrieved_chunks][0] == "1. Meals"
    assert {chunk.section for chunk in response.retrieved_chunks} == {
        "1. Meals",
        "2. Hotels",
        "6. Submission Deadline",
    }
    distances = [chunk.distance for chunk in response.retrieved_chunks]
    assert distances == sorted(distances)


def test_ollama_chat_adapter_returns_message_text() -> None:
    """Read the Ollama message body and return it as plain text."""

    class RecordingClient:
        def chat(self, model: str, messages: list, **kwargs):
            """Return one scripted chat message and record the call."""
            self.messages = messages
            self.kwargs = kwargs
            return SimpleNamespace(message=SimpleNamespace(content='{"answerable": true, "answer": "$65"}'))

    client = RecordingClient()
    text = OllamaChatAdapter(model="qwen3:8b", client=client).complete("Question: meals")

    assert text == '{"answerable": true, "answer": "$65"}'
    assert client.messages == [{"role": "user", "content": "Question: meals"}]
    assert client.kwargs["think"] is False
    assert client.kwargs["options"] == {"temperature": CHAT_TEMPERATURE}
    assert "answerable" in client.kwargs["format"]["properties"]


def test_ollama_chat_adapter_reads_a_dict_response() -> None:
    """Read message text when the client returns a plain dictionary."""

    class DictClient:
        def chat(self, model: str, messages: list, **kwargs):
            """Return one scripted dictionary response."""
            return {"message": {"content": '{"answerable": false, "answer": ""}'}}

    text = OllamaChatAdapter(client=DictClient()).complete("Question: gym")
    assert text == '{"answerable": false, "answer": ""}'


def test_ollama_chat_adapter_rejects_an_empty_response() -> None:
    """Reject a chat response that has no message text."""

    class EmptyClient:
        def chat(self, model: str, messages: list, **kwargs):
            """Return a message with no content."""
            return SimpleNamespace(message=SimpleNamespace(content=""))

    with pytest.raises(ValueError, match="empty response"):
        OllamaChatAdapter(client=EmptyClient()).complete("Question: meals")
