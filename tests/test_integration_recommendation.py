"""Task 12.4 — Integration test: full recommendation turn cycle.

Uses a real DatasetStore with a 50-title fixture and a mocked LLMPort so that
ConversationOrchestrator.handle_turn() exercises the complete filter → rank →
generate pipeline without real API calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from sommelier.application.conversation_orchestrator import ConversationOrchestrator
from sommelier.application.recommendation_engine import RecommendationEngine
from sommelier.application.response_generator import ResponseGenerator
from sommelier.application.session_manager import SessionManager
from sommelier.domain.candidate_retriever import CandidateRetriever
from sommelier.ports.interfaces import DatasetFilter, LLMResponse
from sommelier.domain.preference_extractor import PreferenceExtractor
from sommelier.infrastructure.dataset_store import DatasetStore
from tests.fixtures import write_fixture_csv


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def fifty_csv(tmp_path):
    return write_fixture_csv(tmp_path / "fifty.csv", n=50)


@pytest.fixture()
def dataset(fifty_csv):
    ds = DatasetStore()
    ds.load_and_index(fifty_csv)
    return ds


def _llm_response(payload: dict) -> LLMResponse:
    return LLMResponse(content=json.dumps(payload), input_tokens=10, output_tokens=20)


def _mock_llm(extraction_payload: dict, generation_text: str = "Here are your picks!") -> MagicMock:
    """Return a mock LLMPort that returns extraction_payload for extraction calls
    and generation_text for generation calls."""
    llm = MagicMock()

    def complete(request):
        if request.model == "extraction":
            return _llm_response(extraction_payload)
        return LLMResponse(content=generation_text, input_tokens=5, output_tokens=30)

    llm.complete.side_effect = complete
    return llm


@pytest.fixture()
def orchestrator(dataset):
    """Wire a full orchestrator with 50-title real DatasetStore + mocked LLM."""
    extraction_delta = {
        "genres": ["Drama"],
        "mood_keywords": ["compelling", "adventure"],
        "content_type": None,
        "year_min": None,
        "year_max": None,
        "maturity_ceiling": None,
        "country_filter": None,
        "excluded_title_ids": [],
        "positive_genre_signals": [],
        "needs_clarification": False,
        "clarification_hint": None,
        "has_conflict": False,
        "conflict_description": None,
    }
    llm = _mock_llm(extraction_delta, generation_text="Here are your drama picks!")
    sm = SessionManager()
    extractor = PreferenceExtractor(llm)
    retriever = CandidateRetriever(dataset)
    engine = RecommendationEngine(retriever)
    generator = ResponseGenerator(llm, dataset)
    return ConversationOrchestrator(
        session_manager=sm,
        preference_extractor=extractor,
        recommendation_engine=engine,
        response_generator=generator,
        dataset=dataset,
    )


# ── 12.4 Tests ────────────────────────────────────────────────────────────────


class TestRecommendationTurnCycle:
    def test_handle_turn_returns_tuple(self, orchestrator):
        session, _ = orchestrator.start_session()
        result = orchestrator.handle_turn("I want drama films", session)
        assert isinstance(result, tuple) and len(result) == 2

    def test_handle_turn_updates_session_history(self, orchestrator):
        session, _ = orchestrator.start_session()
        session, _ = orchestrator.handle_turn("I want drama films", session)
        roles = [m.role for m in session.conversation_history]
        assert "user" in roles
        assert "assistant" in roles

    def test_handle_turn_returns_response_string(self, orchestrator):
        session, _ = orchestrator.start_session()
        _, response = orchestrator.handle_turn("I want drama films", session)
        assert isinstance(response, str) and len(response) > 0

    def test_shown_titles_registered_after_turn(self, orchestrator):
        session, _ = orchestrator.start_session()
        session, _ = orchestrator.handle_turn("I want drama films", session)
        # After a recommendation turn, seen_title_ids should be populated
        assert len(session.seen_title_ids) >= 3

    def test_shown_title_count_between_3_and_10(self, orchestrator):
        session, _ = orchestrator.start_session()
        session, _ = orchestrator.handle_turn("I want drama films", session)
        assert 3 <= len(session.seen_title_ids) <= 10

    def test_maturity_ceiling_respected_end_to_end(self, dataset):
        """Titles rated TV-MA must not appear when ceiling is PG-13."""
        extraction_delta = {
            "genres": ["Drama"],
            "mood_keywords": [],
            "content_type": None,
            "year_min": None,
            "year_max": None,
            "maturity_ceiling": "PG-13",
            "country_filter": None,
            "excluded_title_ids": [],
            "positive_genre_signals": [],
            "needs_clarification": False,
            "clarification_hint": None,
            "has_conflict": False,
            "conflict_description": None,
        }
        llm = _mock_llm(extraction_delta, "Here are family-friendly picks!")
        sm = SessionManager()
        extractor = PreferenceExtractor(llm)
        retriever = CandidateRetriever(dataset)
        engine = RecommendationEngine(retriever)
        generator = ResponseGenerator(llm, dataset)
        orch = ConversationOrchestrator(
            session_manager=sm,
            preference_extractor=extractor,
            recommendation_engine=engine,
            response_generator=generator,
            dataset=dataset,
        )

        session, _ = orch.start_session()
        session, _ = orch.handle_turn("family friendly drama please", session)

        # Verify none of the shown titles are TV-MA
        for sid in session.seen_title_ids:
            title = dataset.get_by_id(sid)
            if title and title.rating:
                assert title.rating != "TV-MA", (
                    f"TV-MA title {title.title} should have been filtered out"
                )

    def test_preference_profile_updated_after_turn(self, orchestrator):
        session, _ = orchestrator.start_session()
        session, _ = orchestrator.handle_turn("I want drama films", session)
        assert "Drama" in session.preference_profile.genres
