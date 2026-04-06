"""Tasks 5.1, 5.2, 5.3 — Verify PreferenceExtractor.

All LLM calls are mocked — no real API calls.

Groups:
  TestExtractBasicFields       — 5.1: valid JSON parsing, field mapping
  TestJsonFallback             — 5.1: malformed JSON fallback
  TestAmbiguityDetection       — 5.2: needs_clarification flag
  TestConflictDetection        — 5.2: has_conflict flag
  TestMaturityValidation       — 5.2: invalid maturity_ceiling → None
  TestFeedbackMode             — 5.3: mode="feedback" populates excluded/positive
"""

import json
from unittest.mock import MagicMock

import pytest

from sommelier.domain.models import (
    Message,
    PreferenceProfileDelta,
    Session,
    PreferenceProfile,
    NETFLIX_RATINGS_ORDERED,
)
from sommelier.ports.interfaces import LLMResponse


# ── Helpers ───────────────────────────────────────────────────────────────────


def _session(messages: list | None = None) -> Session:
    return Session(
        id="test-session",
        conversation_history=messages or [],
        preference_profile=PreferenceProfile(),
        seen_title_ids=frozenset(),
        maturity_ceiling_locked=False,
    )


def _llm_response(content: str) -> LLMResponse:
    return LLMResponse(content=content, input_tokens=10, output_tokens=20)


def _mock_llm(response_text: str) -> MagicMock:
    llm = MagicMock()
    llm.complete.return_value = _llm_response(response_text)
    return llm


def _valid_delta_json(**overrides) -> str:
    base = {
        "genres": [],
        "mood_keywords": [],
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
    base.update(overrides)
    return json.dumps(base)


# ── 5.1: Basic field extraction ───────────────────────────────────────────────


class TestExtractBasicFields:
    def test_extract_returns_preference_profile_delta(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        llm = _mock_llm(_valid_delta_json())
        pe = PreferenceExtractor(llm)
        result = pe.extract("I like drama", _session(), mode="preference")
        assert isinstance(result, PreferenceProfileDelta)

    def test_genres_populated_from_llm_response(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        llm = _mock_llm(_valid_delta_json(genres=["Drama", "Thriller"]))
        pe = PreferenceExtractor(llm)
        result = pe.extract("dark drama", _session(), mode="preference")
        assert "Drama" in result.genres
        assert "Thriller" in result.genres

    def test_mood_keywords_populated(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        llm = _mock_llm(_valid_delta_json(mood_keywords=["dark", "suspenseful"]))
        pe = PreferenceExtractor(llm)
        result = pe.extract("something dark", _session(), mode="preference")
        assert "dark" in result.mood_keywords

    def test_content_type_movie_populated(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        llm = _mock_llm(_valid_delta_json(content_type="Movie"))
        pe = PreferenceExtractor(llm)
        result = pe.extract("a movie", _session(), mode="preference")
        assert result.content_type == "Movie"

    def test_year_range_populated(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        llm = _mock_llm(_valid_delta_json(year_min=1990, year_max=2000))
        pe = PreferenceExtractor(llm)
        result = pe.extract("90s films", _session(), mode="preference")
        assert result.year_min == 1990
        assert result.year_max == 2000

    def test_country_filter_populated(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        llm = _mock_llm(_valid_delta_json(country_filter="Japan"))
        pe = PreferenceExtractor(llm)
        result = pe.extract("Japanese anime", _session(), mode="preference")
        assert result.country_filter == "Japan"

    def test_valid_maturity_ceiling_preserved(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        llm = _mock_llm(_valid_delta_json(maturity_ceiling="PG-13"))
        pe = PreferenceExtractor(llm)
        result = pe.extract("family friendly", _session(), mode="preference")
        assert result.maturity_ceiling == "PG-13"

    def test_llm_called_with_extraction_model(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        from sommelier.ports.interfaces import LLMRequest
        llm = _mock_llm(_valid_delta_json())
        pe = PreferenceExtractor(llm)
        pe.extract("drama", _session(), mode="preference")
        request: LLMRequest = llm.complete.call_args[0][0]
        assert request.model == "extraction"

    def test_llm_called_once_per_extract(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        llm = _mock_llm(_valid_delta_json())
        pe = PreferenceExtractor(llm)
        pe.extract("drama", _session(), mode="preference")
        assert llm.complete.call_count == 1

    def test_system_prompt_contains_schema_hint(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        from sommelier.ports.interfaces import LLMRequest
        llm = _mock_llm(_valid_delta_json())
        pe = PreferenceExtractor(llm)
        pe.extract("drama", _session(), mode="preference")
        request: LLMRequest = llm.complete.call_args[0][0]
        assert "genres" in request.system_prompt
        assert "needs_clarification" in request.system_prompt

    def test_user_message_included_in_request_messages(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        from sommelier.ports.interfaces import LLMRequest
        llm = _mock_llm(_valid_delta_json())
        pe = PreferenceExtractor(llm)
        pe.extract("I want drama films", _session(), mode="preference")
        request: LLMRequest = llm.complete.call_args[0][0]
        all_content = " ".join(m.content for m in request.messages)
        assert "I want drama films" in all_content


# ── 5.1: Malformed JSON fallback ──────────────────────────────────────────────


class TestJsonFallback:
    def test_malformed_json_returns_delta(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        llm = _mock_llm("not valid json at all")
        pe = PreferenceExtractor(llm)
        result = pe.extract("drama", _session(), mode="preference")
        assert isinstance(result, PreferenceProfileDelta)

    def test_malformed_json_sets_needs_clarification(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        llm = _mock_llm("{broken json}")
        pe = PreferenceExtractor(llm)
        result = pe.extract("drama", _session(), mode="preference")
        assert result.needs_clarification is True

    def test_malformed_json_treats_message_as_keyword(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        llm = _mock_llm("oops not json")
        pe = PreferenceExtractor(llm)
        result = pe.extract("romantic comedy", _session(), mode="preference")
        assert "romantic comedy" in result.mood_keywords

    def test_extract_never_raises_on_llm_failure(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        from sommelier.domain.models import LLMUnavailableError
        llm = MagicMock()
        llm.complete.side_effect = LLMUnavailableError("API down")
        pe = PreferenceExtractor(llm)
        result = pe.extract("drama", _session(), mode="preference")
        assert isinstance(result, PreferenceProfileDelta)
        assert result.needs_clarification is True

    def test_empty_json_object_returns_empty_delta(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        llm = _mock_llm("{}")
        pe = PreferenceExtractor(llm)
        result = pe.extract("drama", _session(), mode="preference")
        assert isinstance(result, PreferenceProfileDelta)
        assert result.genres == ()


# ── 5.2: Ambiguity detection ──────────────────────────────────────────────────


class TestAmbiguityDetection:
    def test_needs_clarification_true_when_llm_sets_it(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        llm = _mock_llm(_valid_delta_json(needs_clarification=True, clarification_hint="What genre?"))
        pe = PreferenceExtractor(llm)
        result = pe.extract("something", _session(), mode="preference")
        assert result.needs_clarification is True

    def test_clarification_hint_populated(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        llm = _mock_llm(_valid_delta_json(needs_clarification=True, clarification_hint="What genre do you prefer?"))
        pe = PreferenceExtractor(llm)
        result = pe.extract("something", _session(), mode="preference")
        assert result.clarification_hint == "What genre do you prefer?"

    def test_needs_clarification_false_for_clear_query(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        llm = _mock_llm(_valid_delta_json(genres=["Drama"], needs_clarification=False))
        pe = PreferenceExtractor(llm)
        result = pe.extract("I love drama", _session(), mode="preference")
        assert result.needs_clarification is False


# ── 5.2: Conflict detection ───────────────────────────────────────────────────


class TestConflictDetection:
    def test_has_conflict_true_when_llm_sets_it(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        llm = _mock_llm(_valid_delta_json(
            has_conflict=True,
            conflict_description="Wants both G-rated and R-rated content"
        ))
        pe = PreferenceExtractor(llm)
        result = pe.extract("family friendly but violent", _session(), mode="preference")
        assert result.has_conflict is True

    def test_conflict_description_populated(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        desc = "Requested both old and new films simultaneously"
        llm = _mock_llm(_valid_delta_json(has_conflict=True, conflict_description=desc))
        pe = PreferenceExtractor(llm)
        result = pe.extract("90s films from 2020", _session(), mode="preference")
        assert result.conflict_description == desc

    def test_has_conflict_false_for_clean_query(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        llm = _mock_llm(_valid_delta_json(genres=["Comedy"], has_conflict=False))
        pe = PreferenceExtractor(llm)
        result = pe.extract("funny movies", _session(), mode="preference")
        assert result.has_conflict is False


# ── 5.2: Maturity ceiling validation ─────────────────────────────────────────


class TestMaturityValidation:
    def test_valid_rating_preserved(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        for rating in ("G", "PG", "PG-13", "TV-MA"):
            llm = _mock_llm(_valid_delta_json(maturity_ceiling=rating))
            pe = PreferenceExtractor(llm)
            result = pe.extract("test", _session(), mode="preference")
            assert result.maturity_ceiling == rating

    def test_unrecognized_rating_replaced_with_none(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        llm = _mock_llm(_valid_delta_json(maturity_ceiling="X-RATED-BOGUS"))
        pe = PreferenceExtractor(llm)
        result = pe.extract("test", _session(), mode="preference")
        assert result.maturity_ceiling is None

    def test_all_known_ratings_are_valid(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        for rating in NETFLIX_RATINGS_ORDERED:
            llm = _mock_llm(_valid_delta_json(maturity_ceiling=rating))
            pe = PreferenceExtractor(llm)
            result = pe.extract("test", _session(), mode="preference")
            assert result.maturity_ceiling == rating


# ── 5.3: Feedback mode ────────────────────────────────────────────────────────


class TestFeedbackMode:
    def test_feedback_mode_populates_excluded_title_ids(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        llm = _mock_llm(_valid_delta_json(excluded_title_ids=["s1", "s3"]))
        pe = PreferenceExtractor(llm)
        result = pe.extract("not that one", _session(), mode="feedback")
        assert "s1" in result.excluded_title_ids
        assert "s3" in result.excluded_title_ids

    def test_feedback_mode_populates_positive_genre_signals(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        llm = _mock_llm(_valid_delta_json(positive_genre_signals=["Action", "Thriller"]))
        pe = PreferenceExtractor(llm)
        result = pe.extract("I liked that one", _session(), mode="feedback")
        assert "Action" in result.positive_genre_signals

    def test_feedback_mode_uses_extraction_model(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        from sommelier.ports.interfaces import LLMRequest
        llm = _mock_llm(_valid_delta_json())
        pe = PreferenceExtractor(llm)
        pe.extract("liked it", _session(), mode="feedback")
        request: LLMRequest = llm.complete.call_args[0][0]
        assert request.model == "extraction"

    def test_feedback_system_prompt_differs_from_preference(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        from sommelier.ports.interfaces import LLMRequest
        llm = _mock_llm(_valid_delta_json())
        pe = PreferenceExtractor(llm)
        pe.extract("drama", _session(), mode="preference")
        pref_prompt = llm.complete.call_args[0][0].system_prompt
        pe.extract("not that", _session(), mode="feedback")
        fb_prompt = llm.complete.call_args[0][0].system_prompt
        assert pref_prompt != fb_prompt

    def test_feedback_mode_never_raises(self):
        from sommelier.domain.preference_extractor import PreferenceExtractor
        from sommelier.domain.models import LLMUnavailableError
        llm = MagicMock()
        llm.complete.side_effect = LLMUnavailableError("down")
        pe = PreferenceExtractor(llm)
        result = pe.extract("not that", _session(), mode="feedback")
        assert isinstance(result, PreferenceProfileDelta)
