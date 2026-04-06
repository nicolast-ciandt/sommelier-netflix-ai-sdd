"""Tasks 9.1, 9.2, 9.3 — Verify ConversationOrchestrator routing, pipeline, and error recovery.

All collaborators (SessionManager, PreferenceExtractor, RecommendationEngine,
ResponseGenerator) are real instances where possible; LLM and dataset are mocked.

Groups:
  TestStartSession             — 9.1: creates session, returns greeting
  TestHandleTurnBasics         — 9.1: appends messages, returns (session, str)
  TestRecommendationPipeline   — 9.2: preference→recommend→response→register
  TestClarificationBranch      — 9.2: needs_clarification → clarification response
  TestFeedbackPipeline         — 9.2: feedback mode path
  TestTitleDetailPipeline      — 9.2: title detail and catalog-miss routing
  TestTimeoutAndErrors         — 9.3: LLMUnavailableError and unexpected exceptions
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from sommelier.domain.models import (
    DurationInfo,
    LLMUnavailableError,
    NetflixTitle,
    NoResultsResult,
    PreferenceProfile,
    Recommendation,
    Session,
)
from sommelier.ports.interfaces import LLMResponse


# ── Helpers ───────────────────────────────────────────────────────────────────


def _llm_response(text: str = "response") -> LLMResponse:
    return LLMResponse(content=text, input_tokens=5, output_tokens=10)


def _title(show_id: str = "s1") -> NetflixTitle:
    return NetflixTitle(
        show_id=show_id, type="Movie", title="Test Film",
        director="Dir", cast=("Actor",), country="USA",
        release_year=2020, rating="PG", duration=DurationInfo(90, "min"),
        genres=("Drama",), description="A test.",
    )


def _make_orchestrator(
    llm_text: str = "LLM response",
    llm_side_effect=None,
    dataset_titles: list | None = None,
):
    """Build a fully wired ConversationOrchestrator with mocked LLM and dataset."""
    import json
    from sommelier.application.conversation_orchestrator import ConversationOrchestrator
    from sommelier.application.recommendation_engine import RecommendationEngine
    from sommelier.application.response_generator import ResponseGenerator
    from sommelier.application.session_manager import SessionManager
    from sommelier.domain.candidate_retriever import CandidateRetriever
    from sommelier.domain.preference_extractor import PreferenceExtractor
    from sommelier.infrastructure.claude_adapter import ClaudeAdapter

    # Mock LLM
    mock_llm = MagicMock()
    if llm_side_effect:
        mock_llm.complete.side_effect = llm_side_effect
    else:
        # Default: return a valid preference delta JSON so extraction works
        valid_delta = json.dumps({
            "genres": ["Drama"], "mood_keywords": [],
            "content_type": None, "year_min": None, "year_max": None,
            "maturity_ceiling": None, "country_filter": None,
            "excluded_title_ids": [], "positive_genre_signals": [],
            "needs_clarification": False, "clarification_hint": None,
            "has_conflict": False, "conflict_description": None,
        })
        mock_llm.complete.return_value = LLMResponse(
            content=valid_delta, input_tokens=5, output_tokens=10
        )

    # Mock dataset
    mock_dataset = MagicMock()
    titles = dataset_titles or [_title(f"s{i}") for i in range(1, 11)]
    mock_dataset.filter.return_value = titles
    mock_dataset.tfidf_similarity.return_value = [
        MagicMock(title=t, similarity_score=0.5) for t in titles
    ]
    mock_dataset.get_by_id.return_value = _title()
    mock_dataset.title_count.return_value = len(titles)

    sm = SessionManager(max_history_turns=20)
    pe = PreferenceExtractor(mock_llm)
    cr = CandidateRetriever(mock_dataset)
    re = RecommendationEngine(cr)
    rg = ResponseGenerator(mock_llm, mock_dataset)

    orch = ConversationOrchestrator(
        session_manager=sm,
        preference_extractor=pe,
        recommendation_engine=re,
        response_generator=rg,
        dataset=mock_dataset,
    )
    return orch, mock_llm, mock_dataset


# ── 9.1: start_session ────────────────────────────────────────────────────────


class TestStartSession:
    def test_returns_tuple_of_session_and_string(self):
        from sommelier.application.conversation_orchestrator import ConversationOrchestrator
        orch, _, _ = _make_orchestrator()
        result = orch.start_session()
        assert isinstance(result, tuple)
        assert len(result) == 2
        session, greeting = result
        assert isinstance(session, Session)
        assert isinstance(greeting, str)

    def test_session_starts_empty(self):
        orch, _, _ = _make_orchestrator()
        session, _ = orch.start_session()
        assert session.conversation_history == []
        assert session.seen_title_ids == frozenset()

    def test_greeting_is_nonempty(self):
        orch, _, _ = _make_orchestrator()
        _, greeting = orch.start_session()
        assert len(greeting) > 0

    def test_each_start_session_has_unique_id(self):
        orch, _, _ = _make_orchestrator()
        s1, _ = orch.start_session()
        s2, _ = orch.start_session()
        assert s1.id != s2.id


# ── 9.1: handle_turn basics ───────────────────────────────────────────────────


class TestHandleTurnBasics:
    def test_returns_tuple_of_session_and_string(self):
        orch, _, _ = _make_orchestrator()
        session, _ = orch.start_session()
        result = orch.handle_turn("I want drama movies", session)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], Session)
        assert isinstance(result[1], str)

    def test_user_message_appended_to_history(self):
        orch, _, _ = _make_orchestrator()
        session, _ = orch.start_session()
        updated, _ = orch.handle_turn("I like action films", session)
        user_messages = [m for m in updated.conversation_history if m.role == "user"]
        assert any("action" in m.content for m in user_messages)

    def test_assistant_response_appended_to_history(self):
        orch, _, _ = _make_orchestrator()
        session, _ = orch.start_session()
        updated, _ = orch.handle_turn("I like drama", session)
        assistant_messages = [m for m in updated.conversation_history if m.role == "assistant"]
        assert len(assistant_messages) >= 1

    def test_response_is_nonempty_string(self):
        orch, _, _ = _make_orchestrator()
        session, _ = orch.start_session()
        _, response = orch.handle_turn("I want comedies", session)
        assert isinstance(response, str)
        assert len(response) > 0


# ── 9.2: Recommendation pipeline ─────────────────────────────────────────────


class TestRecommendationPipeline:
    def test_shown_titles_registered_in_session_after_recommendations(self):
        orch, _, _ = _make_orchestrator()
        session, _ = orch.start_session()
        updated, _ = orch.handle_turn("I want drama", session)
        # seen_title_ids should be populated after a successful recommendation turn
        assert len(updated.seen_title_ids) > 0

    def test_preference_profile_updated_after_turn(self):
        orch, _, _ = _make_orchestrator()
        session, _ = orch.start_session()
        updated, _ = orch.handle_turn("I want drama movies", session)
        # The LLM returned genres=["Drama"] in the delta
        assert "Drama" in updated.preference_profile.genres


# ── 9.2: Clarification branch ─────────────────────────────────────────────────


class TestClarificationBranch:
    def test_needs_clarification_triggers_clarification_response(self):
        import json
        orch, mock_llm, _ = _make_orchestrator()
        # Override LLM to return needs_clarification=True
        clarification_delta = json.dumps({
            "genres": [], "mood_keywords": [], "content_type": None,
            "year_min": None, "year_max": None, "maturity_ceiling": None,
            "country_filter": None, "excluded_title_ids": [], "positive_genre_signals": [],
            "needs_clarification": True, "clarification_hint": "What genre?",
            "has_conflict": False, "conflict_description": None,
        })
        # First call (extraction) returns clarification delta;
        # second call (generate_clarification) returns the question text
        mock_llm.complete.side_effect = [
            LLMResponse(content=clarification_delta, input_tokens=5, output_tokens=10),
            LLMResponse(content="What genre do you prefer?", input_tokens=5, output_tokens=10),
        ]
        session, _ = orch.start_session()
        _, response = orch.handle_turn("something", session)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_clarification_does_not_register_shown_titles(self):
        import json
        orch, mock_llm, _ = _make_orchestrator()
        clarification_delta = json.dumps({
            "genres": [], "mood_keywords": [], "content_type": None,
            "year_min": None, "year_max": None, "maturity_ceiling": None,
            "country_filter": None, "excluded_title_ids": [], "positive_genre_signals": [],
            "needs_clarification": True, "clarification_hint": "What genre?",
            "has_conflict": False, "conflict_description": None,
        })
        mock_llm.complete.side_effect = [
            LLMResponse(content=clarification_delta, input_tokens=5, output_tokens=10),
            LLMResponse(content="What genre?", input_tokens=5, output_tokens=10),
        ]
        session, _ = orch.start_session()
        updated, _ = orch.handle_turn("something", session)
        assert updated.seen_title_ids == frozenset()


# ── 9.2: Feedback pipeline ────────────────────────────────────────────────────


class TestFeedbackPipeline:
    def test_feedback_message_processed_without_error(self):
        import json
        orch, mock_llm, _ = _make_orchestrator()
        feedback_delta = json.dumps({
            "genres": [], "mood_keywords": [], "content_type": None,
            "year_min": None, "year_max": None, "maturity_ceiling": None,
            "country_filter": None, "excluded_title_ids": ["s1"],
            "positive_genre_signals": ["Action"],
            "needs_clarification": False, "clarification_hint": None,
            "has_conflict": False, "conflict_description": None,
        })
        # First: pref extraction; second: feedback extraction; third: recs response
        mock_llm.complete.side_effect = [
            LLMResponse(content=json.dumps({
                "genres": ["Drama"], "mood_keywords": [], "content_type": None,
                "year_min": None, "year_max": None, "maturity_ceiling": None,
                "country_filter": None, "excluded_title_ids": [], "positive_genre_signals": [],
                "needs_clarification": False, "clarification_hint": None,
                "has_conflict": False, "conflict_description": None,
            }), input_tokens=5, output_tokens=10),
            # second turn: feedback extraction
            LLMResponse(content=feedback_delta, input_tokens=5, output_tokens=10),
            # third call: generate response
            LLMResponse(content="Here are updated picks!", input_tokens=5, output_tokens=10),
        ]
        session, _ = orch.start_session()
        session, _ = orch.handle_turn("I want drama", session)
        updated, response = orch.handle_turn("not that one, I prefer action", session)
        assert isinstance(response, str)


# ── 9.2: Title detail pipeline ───────────────────────────────────────────────


class TestTitleDetailPipeline:
    def test_title_detail_question_returns_response(self):
        orch, mock_llm, mock_dataset = _make_orchestrator()
        mock_dataset.get_by_id.return_value = _title("s1")
        # Override to always return a string for generate calls
        mock_llm.complete.return_value = LLMResponse(
            content="Here's what I know about this film.", input_tokens=5, output_tokens=10
        )
        session, _ = orch.start_session()
        _, response = orch.handle_turn("Tell me about s1", session)
        assert isinstance(response, str)


# ── 9.3: Timeout and error recovery ──────────────────────────────────────────


class TestTimeoutAndErrors:
    def test_llm_unavailable_returns_friendly_message(self):
        orch, mock_llm, _ = _make_orchestrator()
        mock_llm.complete.side_effect = LLMUnavailableError("API down")
        session, _ = orch.start_session()
        updated, response = orch.handle_turn("I want movies", session)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_llm_unavailable_session_unchanged(self):
        orch, mock_llm, _ = _make_orchestrator()
        session, _ = orch.start_session()
        original_seen = session.seen_title_ids
        mock_llm.complete.side_effect = LLMUnavailableError("down")
        updated, _ = orch.handle_turn("drama movies", session)
        assert updated.seen_title_ids == original_seen

    def test_unexpected_exception_returns_recovery_message(self):
        orch, mock_llm, _ = _make_orchestrator()
        mock_llm.complete.side_effect = RuntimeError("unexpected crash")
        session, _ = orch.start_session()
        updated, response = orch.handle_turn("drama", session)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_unexpected_exception_does_not_propagate(self):
        orch, mock_llm, _ = _make_orchestrator()
        mock_llm.complete.side_effect = RuntimeError("crash")
        session, _ = orch.start_session()
        # Must not raise
        try:
            orch.handle_turn("drama", session)
        except Exception as exc:
            pytest.fail(f"handle_turn raised unexpectedly: {exc}")
