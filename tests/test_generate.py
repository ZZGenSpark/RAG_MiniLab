from types import SimpleNamespace

from rag.ask import ask
from rag.embeddings import Embedder
from rag.generate import REFUSAL_ANSWER, Generator, generate_answer
from rag.schema import PolicyChunk, RetrievedChunk
from rag.store import PolicyStore


def _chunk(
    section: str,
    title: str,
    text: str,
    distance: float,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"expense-policy:v2.0:section-{section}",
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


class FakeChat:
    def __init__(self, content: str) -> None:
        self.content = content
        self.prompts: list[str] = []

    def chat(self, model: str, messages: list, **kwargs):
        self.prompts.append(messages[0]["content"])
        return SimpleNamespace(message=SimpleNamespace(content=self.content))


class FakeEmbedderClient:
    def embed(self, model: str, input: str | list[str]):
        texts = input if isinstance(input, list) else [input]
        return SimpleNamespace(embeddings=[[1.0, 0.0, 0.0] for _ in texts])


def _as_policy_chunk(chunk: RetrievedChunk) -> PolicyChunk:
    return PolicyChunk(
        chunk_id=chunk.chunk_id,
        document=chunk.document,
        version=chunk.version,
        section=chunk.section,
        section_title=chunk.section_title,
        text=chunk.text,
    )


def _store_with_meals(tmp_path) -> PolicyStore:
    store = PolicyStore(tmp_path / "chroma")
    store.upsert_chunks(
        [_as_policy_chunk(MEALS), _as_policy_chunk(HOTELS), _as_policy_chunk(DEADLINE)],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    return store


def test_supported_answer_uses_only_retrieved_evidence_and_cites_section() -> None:
    generator = Generator(
        client=FakeChat(
            '{"answer": "Employees may claim up to $65 per day for meals.", "section": "1. Meals"}'
        )
    )
    answer, citation = generate_answer(
        "How much can I spend on food each day?",
        [MEALS, HOTELS, DEADLINE],
        generator=generator,
    )

    assert "$65" in answer
    assert citation is not None
    assert citation.document == "Employee Expense Policy"
    assert citation.version == "2.0"
    assert citation.section == "1. Meals"


def test_unsupported_question_refuses_without_citation() -> None:
    answer, citation = generate_answer(
        "Does the company reimburse professional conference tickets?",
        [MEALS, HOTELS, DEADLINE],
        generator=Generator(
            client=FakeChat(
                '{"answer": "The provided policy does not answer this question.", "section": null}'
            )
        ),
    )
    assert answer == REFUSAL_ANSWER
    assert citation is None


def test_gym_membership_is_an_unsupported_question_example() -> None:
    answer, citation = generate_answer(
        "Does the company reimburse gym memberships?",
        [MEALS, HOTELS, DEADLINE],
        generator=Generator(
            client=FakeChat(
                '{"answer": "The provided policy does not answer this question.", "section": null}'
            )
        ),
    )
    assert answer == REFUSAL_ANSWER
    assert citation is None


def test_invented_section_is_treated_as_unsupported() -> None:
    answer, citation = generate_answer(
        "How much can I spend on food each day?",
        [MEALS, HOTELS, DEADLINE],
        generator=Generator(
            client=FakeChat('{"answer": "Employees get a gym stipend.", "section": "9. Wellness"}')
        ),
    )
    assert answer == REFUSAL_ANSWER
    assert citation is None


def test_empty_retrieval_refuses_without_calling_the_model() -> None:
    class ExplodingChat:
        def chat(self, *args, **kwargs):
            raise AssertionError("model should not run without retrieved excerpts")

    answer, citation = generate_answer(
        "Does the company reimburse gym memberships?",
        [],
        generator=Generator(client=ExplodingChat()),
    )
    assert answer == REFUSAL_ANSWER
    assert citation is None


def test_ask_builds_structured_response_from_retrieval_not_the_model(tmp_path) -> None:
    store = _store_with_meals(tmp_path)
    chat = FakeChat(
        '{"answer": "Employees may claim up to $65 per day for meals.", "section": "1. Meals"}'
    )

    response = ask(
        "How much can I spend on food each day?",
        store=store,
        embedder=Embedder(client=FakeEmbedderClient()),
        generator=Generator(client=chat),
    )

    assert response.citation is not None
    assert response.citation.section == "1. Meals"
    assert len(response.retrieved_chunks) <= 3
    assert response.retrieved_chunks == sorted(
        response.retrieved_chunks, key=lambda chunk: chunk.distance
    )
    assert all(isinstance(chunk.distance, float) for chunk in response.retrieved_chunks)
    assert "1. Meals" in {chunk.section for chunk in response.retrieved_chunks}
    assert "9. Wellness" not in chat.prompts[0]
