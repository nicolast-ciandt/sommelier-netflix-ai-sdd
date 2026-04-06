"""Task 1.3 — Verify port interfaces and their supporting data types."""

import pytest

from sommelier.domain.models import (
    DurationInfo,
    Message,
    NetflixTitle,
    ScoredTitle,
    Session,
)
from sommelier.ports.interfaces import (
    ConversationPort,
    DatasetFilter,
    DatasetPort,
    LLMPort,
    LLMRequest,
    LLMResponse,
)


# ── DatasetFilter ─────────────────────────────────────────────────────────────


class TestDatasetFilter:
    def test_all_fields_default_to_none(self):
        f = DatasetFilter()
        assert f.content_type is None
        assert f.genres is None
        assert f.year_min is None
        assert f.year_max is None
        assert f.maturity_ceiling is None
        assert f.country is None

    def test_instantiation_with_all_fields(self):
        f = DatasetFilter(
            content_type="Movie",
            genres=["Drama", "Thriller"],
            year_min=2000,
            year_max=2022,
            maturity_ceiling="PG-13",
            country="United States",
        )
        assert f.content_type == "Movie"
        assert f.genres == ["Drama", "Thriller"]
        assert f.year_min == 2000
        assert f.year_max == 2022
        assert f.maturity_ceiling == "PG-13"
        assert f.country == "United States"

    def test_partial_instantiation(self):
        f = DatasetFilter(content_type="TV Show", year_min=2015)
        assert f.content_type == "TV Show"
        assert f.year_min == 2015
        assert f.genres is None

    def test_tv_show_content_type(self):
        f = DatasetFilter(content_type="TV Show")
        assert f.content_type == "TV Show"


# ── LLMRequest ────────────────────────────────────────────────────────────────


class TestLLMRequest:
    def test_instantiation_with_required_fields(self):
        req = LLMRequest(
            system_prompt="You are a sommelier.",
            messages=[],
            model="extraction",
            max_tokens=300,
        )
        assert req.system_prompt == "You are a sommelier."
        assert req.messages == []
        assert req.model == "extraction"
        assert req.max_tokens == 300

    def test_default_temperature(self):
        req = LLMRequest(
            system_prompt="prompt",
            messages=[],
            model="extraction",
            max_tokens=100,
        )
        assert req.temperature == 0.3

    def test_custom_temperature(self):
        req = LLMRequest(
            system_prompt="prompt",
            messages=[Message(role="user", content="hi")],
            model="generation",
            max_tokens=1000,
            temperature=0.7,
        )
        assert req.temperature == 0.7
        assert req.model == "generation"
        assert len(req.messages) == 1

    def test_extraction_and_generation_model_types(self):
        extraction = LLMRequest("p", [], "extraction", 200)
        generation = LLMRequest("p", [], "generation", 500)
        assert extraction.model == "extraction"
        assert generation.model == "generation"

    def test_messages_are_mutable(self):
        req = LLMRequest("p", [], "extraction", 100)
        req.messages.append(Message(role="user", content="test"))
        assert len(req.messages) == 1


# ── LLMResponse ───────────────────────────────────────────────────────────────


class TestLLMResponse:
    def test_instantiation(self):
        resp = LLMResponse(
            content='{"genres": ["Drama"]}',
            input_tokens=120,
            output_tokens=45,
        )
        assert resp.content == '{"genres": ["Drama"]}'
        assert resp.input_tokens == 120
        assert resp.output_tokens == 45

    def test_is_frozen(self):
        from dataclasses import FrozenInstanceError

        resp = LLMResponse(content="text", input_tokens=10, output_tokens=5)
        with pytest.raises(FrozenInstanceError):
            resp.content = "other"  # type: ignore[misc]

    def test_equality(self):
        r1 = LLMResponse("text", 10, 5)
        r2 = LLMResponse("text", 10, 5)
        assert r1 == r2


# ── DatasetPort Protocol ──────────────────────────────────────────────────────


class TestDatasetPort:
    def _make_title(self) -> NetflixTitle:
        return NetflixTitle(
            show_id="s1", type="Movie", title="Film", director=None,
            cast=(), country=None, release_year=2020, rating=None,
            duration=None, genres=(), description="desc",
        )

    def test_complete_implementation_satisfies_protocol(self):
        title = self._make_title()

        class MockDatasetStore:
            def filter(self, criteria: DatasetFilter) -> list[NetflixTitle]:
                return [title]

            def get_by_id(self, show_id: str) -> NetflixTitle | None:
                return title if show_id == "s1" else None

            def tfidf_similarity(
                self, query: str, candidates: list[NetflixTitle]
            ) -> list[ScoredTitle]:
                return [ScoredTitle(title=title, similarity_score=0.9)]

            def title_count(self) -> int:
                return 1

        store = MockDatasetStore()
        assert isinstance(store, DatasetPort)

    def test_missing_filter_fails_protocol_check(self):
        class IncompleteStore:
            def get_by_id(self, show_id: str) -> NetflixTitle | None:
                return None

            def tfidf_similarity(self, query: str, candidates: list) -> list:
                return []

            def title_count(self) -> int:
                return 0

        assert not isinstance(IncompleteStore(), DatasetPort)

    def test_missing_get_by_id_fails_protocol_check(self):
        class IncompleteStore:
            def filter(self, criteria: DatasetFilter) -> list:
                return []

            def tfidf_similarity(self, query: str, candidates: list) -> list:
                return []

            def title_count(self) -> int:
                return 0

        assert not isinstance(IncompleteStore(), DatasetPort)

    def test_completely_empty_class_fails_protocol_check(self):
        assert not isinstance(object(), DatasetPort)


# ── LLMPort Protocol ──────────────────────────────────────────────────────────


class TestLLMPort:
    def test_complete_implementation_satisfies_protocol(self):
        class MockLLMAdapter:
            def complete(self, request: LLMRequest) -> LLMResponse:
                return LLMResponse(content="ok", input_tokens=10, output_tokens=5)

        adapter = MockLLMAdapter()
        assert isinstance(adapter, LLMPort)

    def test_missing_complete_method_fails_protocol_check(self):
        class NullAdapter:
            pass

        assert not isinstance(NullAdapter(), LLMPort)

    def test_mock_complete_is_callable(self):
        class MockLLMAdapter:
            def complete(self, request: LLMRequest) -> LLMResponse:
                return LLMResponse(content=request.system_prompt, input_tokens=5, output_tokens=3)

        adapter = MockLLMAdapter()
        req = LLMRequest(system_prompt="hello", messages=[], model="extraction", max_tokens=50)
        resp = adapter.complete(req)
        assert resp.content == "hello"


# ── ConversationPort Protocol ─────────────────────────────────────────────────


class TestConversationPort:
    def test_complete_implementation_satisfies_protocol(self):
        class MockOrchestrator:
            def start_session(self) -> tuple[Session, str]:
                return Session(id="s1"), "Hello! What would you like to watch?"

            def handle_turn(
                self, user_message: str, session: Session
            ) -> tuple[Session, str]:
                return session, "Here are some suggestions."

        orch = MockOrchestrator()
        assert isinstance(orch, ConversationPort)

    def test_missing_handle_turn_fails_protocol_check(self):
        class IncompleteOrchestrator:
            def start_session(self) -> tuple[Session, str]:
                return Session(id="x"), "Hi"

        assert not isinstance(IncompleteOrchestrator(), ConversationPort)

    def test_missing_start_session_fails_protocol_check(self):
        class IncompleteOrchestrator:
            def handle_turn(self, user_message: str, session: Session) -> tuple[Session, str]:
                return session, "ok"

        assert not isinstance(IncompleteOrchestrator(), ConversationPort)

    def test_mock_start_session_returns_session_and_string(self):
        class MockOrchestrator:
            def start_session(self) -> tuple[Session, str]:
                return Session(id="abc"), "Welcome!"

            def handle_turn(
                self, user_message: str, session: Session
            ) -> tuple[Session, str]:
                return session, "response"

        orch = MockOrchestrator()
        session, greeting = orch.start_session()
        assert isinstance(session, Session)
        assert isinstance(greeting, str)
        assert greeting == "Welcome!"
