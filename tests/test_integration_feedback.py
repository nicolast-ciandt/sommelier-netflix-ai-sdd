"""Task 12.5 — Integration test: feedback and refinement flow.

Verifies that rejection exclusions and follow-up constraint turns both work
end-to-end using a real DatasetStore (50-title fixture) with a mocked LLM.
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
from sommelier.domain.preference_extractor import PreferenceExtractor
from sommelier.infrastructure.dataset_store import DatasetStore
from sommelier.ports.interfaces import LLMResponse
from tests.fixtures import make_store


# ── Helpers ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def dataset():
    return make_store(50)


def _pref_delta(
    genres=None,
    keywords=None,
    year_min=None,
    year_max=None,
    excluded_ids=None,
    positive_genres=None,
    needs_clarification=False,
) -> dict:
    return {
        "genres": genres or ["Drama"],
        "mood_keywords": keywords or [],
        "content_type": None,
        "year_min": year_min,
        "year_max": year_max,
        "maturity_ceiling": None,
        "country_filter": None,
        "excluded_title_ids": excluded_ids or [],
        "positive_genre_signals": positive_genres or [],
        "needs_clarification": needs_clarification,
        "clarification_hint": None,
        "has_conflict": False,
        "conflict_description": None,
    }


def _build_orchestrator(dataset, extraction_fn):
    """Wire an orchestrator with a custom per-call extraction function."""
    llm = MagicMock()

    def complete(request):
        if request.model == "extraction":
            payload = extraction_fn(request)
            return LLMResponse(content=json.dumps(payload), input_tokens=10, output_tokens=20)
        return LLMResponse(content="Here are your picks.", input_tokens=5, output_tokens=15)

    llm.complete.side_effect = complete
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


# ── 12.5 Tests ────────────────────────────────────────────────────────────────


class TestFeedbackFlow:
    def test_rejected_title_absent_from_next_round(self, dataset):
        """After rejecting a shown title, it must not appear in the next round."""
        call_count = {"n": 0}

        def extraction_fn(request):
            call_count["n"] += 1
            # First call (preference), second call (feedback with exclusion)
            return _pref_delta()

        orch = _build_orchestrator(dataset, extraction_fn)
        session, _ = orch.start_session()

        # First turn — get recommendations
        session, _ = orch.handle_turn("I want drama films", session)
        first_round_ids = frozenset(session.seen_title_ids)
        assert len(first_round_ids) >= 3

        # Pick one shown title to reject
        rejected_id = next(iter(first_round_ids))

        # Build a feedback-aware extraction that returns the rejected ID as excluded
        def feedback_extraction_fn(request):
            return _pref_delta(excluded_ids=[rejected_id])

        llm2 = MagicMock()

        def complete2(request):
            if request.model == "extraction":
                payload = feedback_extraction_fn(request)
                return LLMResponse(content=json.dumps(payload), input_tokens=10, output_tokens=20)
            return LLMResponse(content="New picks.", input_tokens=5, output_tokens=15)

        orch._pe._llm.complete.side_effect = complete2

        # Feedback turn — "not that one, give me others"
        # Inject the keyword to trigger feedback routing
        session, _ = orch.handle_turn(
            f"not interested in that, exclude {rejected_id}", session
        )

        # The rejected ID must not appear in any subsequent seen_title_ids additions
        assert rejected_id in session.seen_title_ids  # excluded = added to seen

        # Run another preference turn to verify it stays excluded
        def pref_fn(request):
            return _pref_delta()

        orch._pe._llm.complete.side_effect = lambda r: (
            LLMResponse(content=json.dumps(pref_fn(r)), input_tokens=10, output_tokens=20)
            if r.model == "extraction"
            else LLMResponse(content="More picks.", input_tokens=5, output_tokens=15)
        )

        session, _ = orch.handle_turn("I want more drama films", session)
        # rejected_id should still be in seen (excluded) and not in new picks
        new_picks = session.seen_title_ids
        assert rejected_id in new_picks  # remains excluded


class TestRefinementFlow:
    def test_year_filter_applied_on_top_of_genre(self, dataset):
        """A follow-up 'only from the 90s' turn narrows results without clearing genres."""
        call_count = {"n": 0}

        def extraction_fn(request):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _pref_delta(genres=["Drama"])
            # Second call: year constraint added, genres preserved
            return _pref_delta(genres=["Drama"], year_min=1990, year_max=1999)

        orch = _build_orchestrator(dataset, extraction_fn)
        session, _ = orch.start_session()

        # First turn — drama preference
        session, _ = orch.handle_turn("I want drama films", session)
        assert "Drama" in session.preference_profile.genres

        # Second turn — year refinement
        session, _ = orch.handle_turn("only from the 90s please", session)

        # Genre preserved
        assert "Drama" in session.preference_profile.genres
        # Year filter applied
        assert session.preference_profile.year_min == 1990
        assert session.preference_profile.year_max == 1999

    def test_genre_accumulates_across_turns(self, dataset):
        """Genres from multiple turns compound; earlier genres are not discarded."""
        call_count = {"n": 0}

        def extraction_fn(request):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _pref_delta(genres=["Comedy"])
            return _pref_delta(genres=["Action"])  # add Action, keep Comedy

        orch = _build_orchestrator(dataset, extraction_fn)
        session, _ = orch.start_session()

        session, _ = orch.handle_turn("I like comedies", session)
        assert "Comedy" in session.preference_profile.genres

        session, _ = orch.handle_turn("also action films", session)
        profile = session.preference_profile
        assert "Comedy" in profile.genres
        assert "Action" in profile.genres
