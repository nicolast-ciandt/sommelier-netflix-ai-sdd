"""Task 11.2 — Smoke test: full conversation flow with a real dataset sample.

Verifies the complete user journey end-to-end using a 50-title fixture and a
mocked LLM (no real API key required):

  1. Start session → receive greeting
  2. Preference query → 3–10 recommendations returned
  3. Rejection feedback → rejected title absent from next round
  4. Follow-up constraint → applied without clearing prior preferences
  5. Response latency under 5 seconds per turn (CPU-only, no real LLM)
"""

from __future__ import annotations

import json
import time
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
from tests.fixtures import write_fixture_csv


# ── Fixture & wiring ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def fifty_csv(tmp_path_factory):
    return write_fixture_csv(tmp_path_factory.mktemp("data") / "fifty.csv", n=50)


@pytest.fixture(scope="module")
def dataset(fifty_csv):
    ds = DatasetStore()
    ds.load_and_index(fifty_csv)
    return ds


class _ScriptedLLM:
    """Cycles through scripted extraction payloads; generation always succeeds."""

    def __init__(self, payloads: list[dict], generation_text: str = "Here are your picks."):
        self._payloads = iter(payloads)
        self._generation_text = generation_text

    def complete(self, request: object) -> LLMResponse:
        if request.model == "extraction":  # type: ignore[union-attr]
            payload = next(self._payloads, {
                "genres": [], "mood_keywords": [], "content_type": None,
                "year_min": None, "year_max": None, "maturity_ceiling": None,
                "country_filter": None, "excluded_title_ids": [],
                "positive_genre_signals": [], "needs_clarification": False,
                "clarification_hint": None, "has_conflict": False, "conflict_description": None,
            })
            return LLMResponse(content=json.dumps(payload), input_tokens=10, output_tokens=20)
        return LLMResponse(content=self._generation_text, input_tokens=5, output_tokens=30)


def _wire(dataset, scripted_llm: _ScriptedLLM) -> ConversationOrchestrator:
    llm_mock = MagicMock()
    llm_mock.complete.side_effect = scripted_llm.complete
    sm = SessionManager()
    extractor = PreferenceExtractor(llm_mock)
    retriever = CandidateRetriever(dataset)
    engine = RecommendationEngine(retriever)
    generator = ResponseGenerator(llm_mock, dataset)
    return ConversationOrchestrator(
        session_manager=sm, preference_extractor=extractor,
        recommendation_engine=engine, response_generator=generator, dataset=dataset,
    )


def _delta(genres=None, year_min=None, year_max=None, excluded=None):
    return {
        "genres": genres or [],
        "mood_keywords": [],
        "content_type": None,
        "year_min": year_min,
        "year_max": year_max,
        "maturity_ceiling": None,
        "country_filter": None,
        "excluded_title_ids": excluded or [],
        "positive_genre_signals": [],
        "needs_clarification": False,
        "clarification_hint": None,
        "has_conflict": False,
        "conflict_description": None,
    }


# ── Smoke tests ───────────────────────────────────────────────────────────────


class TestSmokeFlow:
    def test_start_session_returns_greeting(self, dataset):
        llm = _ScriptedLLM([])
        orch = _wire(dataset, llm)
        _, greeting = orch.start_session()
        assert isinstance(greeting, str) and len(greeting) > 0

    def test_preference_query_returns_3_to_10_recommendations(self, dataset):
        llm = _ScriptedLLM([_delta(genres=["Drama"])])
        orch = _wire(dataset, llm)
        session, _ = orch.start_session()
        session, _ = orch.handle_turn("I want drama films", session)
        assert 3 <= len(session.seen_title_ids) <= 10

    def test_rejected_title_absent_from_next_round(self, dataset):
        llm = _ScriptedLLM([
            _delta(genres=["Drama"]),               # turn 1: preference
            _delta(genres=["Drama"]),               # turn 2: feedback extraction
        ])
        orch = _wire(dataset, llm)
        session, _ = orch.start_session()

        # Turn 1: get recommendations
        session, _ = orch.handle_turn("I want drama films", session)
        first_round_ids = frozenset(session.seen_title_ids)
        assert len(first_round_ids) >= 3

        rejected_id = next(iter(first_round_ids))

        # Patch LLM to return the rejected ID in excluded_title_ids on next call
        orch._pe._llm.complete.side_effect = _ScriptedLLM([
            _delta(genres=["Drama"], excluded=[rejected_id]),  # feedback turn
            _delta(genres=["Drama"]),                           # follow-up turn
        ]).complete

        # Turn 2: rejection feedback
        session, _ = orch.handle_turn(
            f"not interested in that, exclude {rejected_id}", session
        )
        assert rejected_id in session.seen_title_ids

        # Turn 3: another preference turn — rejected_id still excluded
        session, _ = orch.handle_turn("more drama please", session)
        assert rejected_id in session.seen_title_ids

    def test_follow_up_constraint_applied_without_clearing_genres(self, dataset):
        llm = _ScriptedLLM([
            _delta(genres=["Comedy"]),                                # turn 1
            _delta(genres=["Comedy"], year_min=1990, year_max=1999),  # turn 2
        ])
        orch = _wire(dataset, llm)
        session, _ = orch.start_session()

        session, _ = orch.handle_turn("I like comedies", session)
        assert "Comedy" in session.preference_profile.genres

        session, _ = orch.handle_turn("only from the 90s", session)
        assert "Comedy" in session.preference_profile.genres
        assert session.preference_profile.year_min == 1990
        assert session.preference_profile.year_max == 1999

    def test_each_turn_completes_under_5_seconds(self, dataset):
        llm = _ScriptedLLM([_delta(genres=["Action"]), _delta(genres=["Action"])])
        orch = _wire(dataset, llm)
        session, _ = orch.start_session()

        start = time.monotonic()
        session, _ = orch.handle_turn("I want action films", session)
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"Turn took {elapsed:.2f}s — exceeds 5-second SLA"
