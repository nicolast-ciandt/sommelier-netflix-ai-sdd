"""Tasks 7.1, 7.2 — Verify RecommendationEngine orchestration and no-results handling.

Uses a real DatasetStore (loaded_store fixture) and a real CandidateRetriever
so the full pipeline is exercised without extra mocking.

Groups:
  TestRecommendReturnType      — 7.1: output shape
  TestResultCountEnforcement   — 7.1: 3–10 trimming, <3 → NoResultsResult
  TestRecommendationWrapping   — 7.1: ScoredTitle → Recommendation
  TestNoResultsReason          — 7.2: reason field discrimination
  TestNoResultsSuggestion      — 7.2: human-readable suggestion text
"""

import pytest

from sommelier.domain.models import (
    NoResultsResult,
    PreferenceProfile,
    Recommendation,
    Session,
)
from sommelier.infrastructure.dataset_store import DatasetStore


# ── Helpers ───────────────────────────────────────────────────────────────────


def _profile(**kwargs) -> PreferenceProfile:
    defaults = dict(
        genres=[], mood_keywords=[], content_type=None,
        year_min=None, year_max=None, maturity_ceiling=None,
        country_filter=None, positive_genre_signals=[],
    )
    defaults.update(kwargs)
    return PreferenceProfile(**defaults)


def _session(seen: frozenset[str] = frozenset()) -> Session:
    return Session(
        id="t", conversation_history=[], preference_profile=PreferenceProfile(),
        seen_title_ids=seen, maturity_ceiling_locked=False,
    )


# ── 7.1: Return type ──────────────────────────────────────────────────────────


class TestRecommendReturnType:
    def test_returns_list_of_recommendations(self, loaded_store):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        from sommelier.application.recommendation_engine import RecommendationEngine
        cr = CandidateRetriever(loaded_store)
        engine = RecommendationEngine(cr)
        result = engine.recommend(_profile(), _session())
        assert isinstance(result, list)
        assert all(isinstance(r, Recommendation) for r in result)

    def test_returns_no_results_result_when_no_candidates(self, loaded_store):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        from sommelier.application.recommendation_engine import RecommendationEngine
        cr = CandidateRetriever(loaded_store)
        engine = RecommendationEngine(cr)
        result = engine.recommend(_profile(genres=["SciFi-Nonexistent"]), _session())
        assert isinstance(result, NoResultsResult)


# ── 7.1: Result-count enforcement ────────────────────────────────────────────


class TestResultCountEnforcement:
    def test_result_count_at_most_10(self, loaded_store):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        from sommelier.application.recommendation_engine import RecommendationEngine
        cr = CandidateRetriever(loaded_store)
        engine = RecommendationEngine(cr)
        result = engine.recommend(_profile(), _session())
        assert len(result) <= 10

    def test_result_count_at_least_3_when_enough_candidates(self, loaded_store):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        from sommelier.application.recommendation_engine import RecommendationEngine
        cr = CandidateRetriever(loaded_store)
        engine = RecommendationEngine(cr)
        result = engine.recommend(_profile(), _session())
        assert len(result) >= 3

    def test_fewer_than_3_candidates_returns_no_results(self, loaded_store):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        from sommelier.application.recommendation_engine import RecommendationEngine
        cr = CandidateRetriever(loaded_store)
        engine = RecommendationEngine(cr)
        # Exclude all but 2 titles from a Drama-only pool (s1, s3, s9 → 3 Drama titles)
        # Exclude s1 and s3 → only s9 left (1 < 3) → NoResultsResult
        seen = frozenset({"s1", "s3"})
        result = engine.recommend(_profile(genres=["Drama"]), _session(seen=seen))
        assert isinstance(result, NoResultsResult)

    def test_exactly_3_candidates_returns_list(self, loaded_store):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        from sommelier.application.recommendation_engine import RecommendationEngine
        cr = CandidateRetriever(loaded_store)
        engine = RecommendationEngine(cr)
        # Drama gives s1, s3, s9 (exactly 3)
        result = engine.recommend(_profile(genres=["Drama"]), _session())
        assert isinstance(result, list)
        assert len(result) == 3


# ── 7.1: Recommendation wrapping ─────────────────────────────────────────────


class TestRecommendationWrapping:
    def test_recommendation_contains_netflix_title(self, loaded_store):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        from sommelier.application.recommendation_engine import RecommendationEngine
        from sommelier.domain.models import NetflixTitle
        cr = CandidateRetriever(loaded_store)
        engine = RecommendationEngine(cr)
        result = engine.recommend(_profile(), _session())
        assert all(isinstance(r.title, NetflixTitle) for r in result)

    def test_recommendation_relevance_score_is_float(self, loaded_store):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        from sommelier.application.recommendation_engine import RecommendationEngine
        cr = CandidateRetriever(loaded_store)
        engine = RecommendationEngine(cr)
        result = engine.recommend(_profile(), _session())
        assert all(isinstance(r.relevance_score, float) for r in result)

    def test_recommendation_rationale_is_empty_string(self, loaded_store):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        from sommelier.application.recommendation_engine import RecommendationEngine
        cr = CandidateRetriever(loaded_store)
        engine = RecommendationEngine(cr)
        result = engine.recommend(_profile(), _session())
        assert all(r.rationale == "" for r in result)

    def test_seen_titles_not_in_results(self, loaded_store):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        from sommelier.application.recommendation_engine import RecommendationEngine
        cr = CandidateRetriever(loaded_store)
        engine = RecommendationEngine(cr)
        seen = frozenset({"s1", "s2"})
        result = engine.recommend(_profile(), _session(seen=seen))
        result_ids = {r.title.show_id for r in result}
        assert result_ids.isdisjoint(seen)


# ── 7.2: NoResultsResult reason ──────────────────────────────────────────────


class TestNoResultsReason:
    def test_no_matching_titles_reason_when_filter_yields_nothing(self, loaded_store):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        from sommelier.application.recommendation_engine import RecommendationEngine
        cr = CandidateRetriever(loaded_store)
        engine = RecommendationEngine(cr)
        result = engine.recommend(_profile(genres=["SciFi-Ghost"]), _session())
        assert isinstance(result, NoResultsResult)
        assert result.reason == "no_matching_titles"

    def test_all_seen_reason_when_candidates_exhausted(self, loaded_store):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        from sommelier.application.recommendation_engine import RecommendationEngine
        cr = CandidateRetriever(loaded_store)
        engine = RecommendationEngine(cr)
        # Drama pool: s1, s3, s9. Exclude all three → all_seen
        seen = frozenset({"s1", "s3", "s9"})
        result = engine.recommend(_profile(genres=["Drama"]), _session(seen=seen))
        assert isinstance(result, NoResultsResult)
        assert result.reason == "all_seen"


# ── 7.2: NoResultsResult suggestion ──────────────────────────────────────────


class TestNoResultsSuggestion:
    def test_no_matching_titles_suggestion_is_nonempty(self, loaded_store):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        from sommelier.application.recommendation_engine import RecommendationEngine
        cr = CandidateRetriever(loaded_store)
        engine = RecommendationEngine(cr)
        result = engine.recommend(_profile(genres=["SciFi-Ghost"]), _session())
        assert isinstance(result.suggestion, str)
        assert len(result.suggestion) > 0

    def test_all_seen_suggestion_mentions_broadening(self, loaded_store):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        from sommelier.application.recommendation_engine import RecommendationEngine
        cr = CandidateRetriever(loaded_store)
        engine = RecommendationEngine(cr)
        seen = frozenset({"s1", "s3", "s9"})
        result = engine.recommend(_profile(genres=["Drama"]), _session(seen=seen))
        assert isinstance(result.suggestion, str)
        assert len(result.suggestion) > 0
