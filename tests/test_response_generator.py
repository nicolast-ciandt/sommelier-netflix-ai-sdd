"""Tasks 8.1, 8.2, 8.3 — Verify ResponseGenerator LLM-backed response generation.

All LLM calls are mocked — no real API calls.

Groups:
  TestRecommendationsResponse  — 8.1: per-title rationale, language injection
  TestTitleDetailResponse      — 8.2: get_by_id lookup, catalog-miss handling
  TestClarificationResponse    — 8.3: hint forwarded to LLM
  TestNoResultsResponse        — 8.3: NoResultsResult reason + suggestion
  TestLanguageInjection        — 8.1/8.3: language in every system prompt
  TestModelUsage               — uses generation model for all calls
"""

import json
from unittest.mock import MagicMock

import pytest

from sommelier.domain.models import (
    NetflixTitle,
    DurationInfo,
    NoResultsResult,
    PreferenceProfile,
    Recommendation,
    Session,
)
from sommelier.ports.interfaces import LLMResponse


# ── Helpers ───────────────────────────────────────────────────────────────────


def _llm(text: str = "Generated response") -> MagicMock:
    mock = MagicMock()
    mock.complete.return_value = LLMResponse(content=text, input_tokens=10, output_tokens=20)
    return mock


def _title(show_id: str = "s1", title: str = "Test Film") -> NetflixTitle:
    return NetflixTitle(
        show_id=show_id, type="Movie", title=title,
        director="Dir", cast=("Actor A",), country="USA",
        release_year=2020, rating="PG", duration=DurationInfo(90, "min"),
        genres=("Drama",), description="A test film.",
    )


def _recommendation(show_id: str = "s1") -> Recommendation:
    return Recommendation(title=_title(show_id), relevance_score=0.8, rationale="")


def _session() -> Session:
    return Session(
        id="t", conversation_history=[], preference_profile=PreferenceProfile(),
        seen_title_ids=frozenset(), maturity_ceiling_locked=False,
    )


def _dataset(title: NetflixTitle | None = None) -> MagicMock:
    ds = MagicMock()
    ds.get_by_id.return_value = title
    return ds


# ── 8.1: Recommendation response ─────────────────────────────────────────────


class TestRecommendationsResponse:
    def test_returns_string(self):
        from sommelier.application.response_generator import ResponseGenerator
        rg = ResponseGenerator(_llm(), _dataset())
        result = rg.generate_recommendations_response(
            [_recommendation()], _session(), user_language="English"
        )
        assert isinstance(result, str)

    def test_calls_llm_once(self):
        from sommelier.application.response_generator import ResponseGenerator
        llm = _llm()
        rg = ResponseGenerator(llm, _dataset())
        rg.generate_recommendations_response([_recommendation()], _session(), "English")
        assert llm.complete.call_count == 1

    def test_uses_generation_model(self):
        from sommelier.application.response_generator import ResponseGenerator
        from sommelier.ports.interfaces import LLMRequest
        llm = _llm()
        rg = ResponseGenerator(llm, _dataset())
        rg.generate_recommendations_response([_recommendation()], _session(), "English")
        req: LLMRequest = llm.complete.call_args[0][0]
        assert req.model == "generation"

    def test_title_name_included_in_request(self):
        from sommelier.application.response_generator import ResponseGenerator
        from sommelier.ports.interfaces import LLMRequest
        llm = _llm()
        rg = ResponseGenerator(llm, _dataset())
        rg.generate_recommendations_response(
            [_recommendation()], _session(), "English"
        )
        req: LLMRequest = llm.complete.call_args[0][0]
        full_text = req.system_prompt + " ".join(m.content for m in req.messages)
        assert "Test Film" in full_text

    def test_release_year_included_in_request(self):
        from sommelier.application.response_generator import ResponseGenerator
        from sommelier.ports.interfaces import LLMRequest
        llm = _llm()
        rg = ResponseGenerator(llm, _dataset())
        rg.generate_recommendations_response([_recommendation()], _session(), "English")
        req: LLMRequest = llm.complete.call_args[0][0]
        full_text = req.system_prompt + " ".join(m.content for m in req.messages)
        assert "2020" in full_text

    def test_genre_included_in_request(self):
        from sommelier.application.response_generator import ResponseGenerator
        from sommelier.ports.interfaces import LLMRequest
        llm = _llm()
        rg = ResponseGenerator(llm, _dataset())
        rg.generate_recommendations_response([_recommendation()], _session(), "English")
        req: LLMRequest = llm.complete.call_args[0][0]
        full_text = req.system_prompt + " ".join(m.content for m in req.messages)
        assert "Drama" in full_text

    def test_returns_llm_content(self):
        from sommelier.application.response_generator import ResponseGenerator
        rg = ResponseGenerator(_llm("Here are my picks!"), _dataset())
        result = rg.generate_recommendations_response([_recommendation()], _session(), "English")
        assert result == "Here are my picks!"

    def test_empty_recommendations_list_still_calls_llm(self):
        from sommelier.application.response_generator import ResponseGenerator
        llm = _llm()
        rg = ResponseGenerator(llm, _dataset())
        rg.generate_recommendations_response([], _session(), "English")
        assert llm.complete.call_count == 1


# ── 8.2: Title detail response ────────────────────────────────────────────────


class TestTitleDetailResponse:
    def test_returns_string(self):
        from sommelier.application.response_generator import ResponseGenerator
        rg = ResponseGenerator(_llm(), _dataset(_title()))
        result = rg.generate_title_detail_response(
            _title(), "What is this film about?", "English"
        )
        assert isinstance(result, str)

    def test_uses_generation_model(self):
        from sommelier.application.response_generator import ResponseGenerator
        from sommelier.ports.interfaces import LLMRequest
        llm = _llm()
        rg = ResponseGenerator(llm, _dataset(_title()))
        rg.generate_title_detail_response(_title(), "Tell me about it", "English")
        req: LLMRequest = llm.complete.call_args[0][0]
        assert req.model == "generation"

    def test_title_description_included(self):
        from sommelier.application.response_generator import ResponseGenerator
        from sommelier.ports.interfaces import LLMRequest
        llm = _llm()
        rg = ResponseGenerator(llm, _dataset(_title()))
        rg.generate_title_detail_response(_title(), "What genre?", "English")
        req: LLMRequest = llm.complete.call_args[0][0]
        full_text = req.system_prompt + " ".join(m.content for m in req.messages)
        assert "A test film." in full_text

    def test_user_question_included(self):
        from sommelier.application.response_generator import ResponseGenerator
        from sommelier.ports.interfaces import LLMRequest
        llm = _llm()
        rg = ResponseGenerator(llm, _dataset(_title()))
        rg.generate_title_detail_response(_title(), "Is it family friendly?", "English")
        req: LLMRequest = llm.complete.call_args[0][0]
        full_text = req.system_prompt + " ".join(m.content for m in req.messages)
        assert "Is it family friendly?" in full_text


# ── 8.2: Catalog-miss handling ────────────────────────────────────────────────


class TestCatalogMissHandling:
    def test_catalog_miss_returns_string(self):
        from sommelier.application.response_generator import ResponseGenerator
        rg = ResponseGenerator(_llm(), _dataset(title=None))
        result = rg.generate_catalog_miss_response("Unknown Film", "English")
        assert isinstance(result, str)

    def test_catalog_miss_does_not_call_llm(self):
        from sommelier.application.response_generator import ResponseGenerator
        llm = _llm()
        rg = ResponseGenerator(llm, _dataset(title=None))
        rg.generate_catalog_miss_response("Unknown Film", "English")
        assert llm.complete.call_count == 0

    def test_catalog_miss_message_mentions_title(self):
        from sommelier.application.response_generator import ResponseGenerator
        rg = ResponseGenerator(_llm(), _dataset(title=None))
        result = rg.generate_catalog_miss_response("Phantom Movie", "English")
        assert "Phantom Movie" in result


# ── 8.3: Clarification response ───────────────────────────────────────────────


class TestClarificationResponse:
    def test_returns_string(self):
        from sommelier.application.response_generator import ResponseGenerator
        rg = ResponseGenerator(_llm("What genre?"), _dataset())
        result = rg.generate_clarification("What genre do you prefer?", _session(), "English")
        assert isinstance(result, str)

    def test_uses_generation_model(self):
        from sommelier.application.response_generator import ResponseGenerator
        from sommelier.ports.interfaces import LLMRequest
        llm = _llm()
        rg = ResponseGenerator(llm, _dataset())
        rg.generate_clarification("What genre?", _session(), "English")
        req: LLMRequest = llm.complete.call_args[0][0]
        assert req.model == "generation"

    def test_hint_included_in_request(self):
        from sommelier.application.response_generator import ResponseGenerator
        from sommelier.ports.interfaces import LLMRequest
        llm = _llm()
        rg = ResponseGenerator(llm, _dataset())
        rg.generate_clarification("Prefer movies or series?", _session(), "English")
        req: LLMRequest = llm.complete.call_args[0][0]
        full_text = req.system_prompt + " ".join(m.content for m in req.messages)
        assert "Prefer movies or series?" in full_text

    def test_returns_llm_content(self):
        from sommelier.application.response_generator import ResponseGenerator
        rg = ResponseGenerator(_llm("Could you clarify?"), _dataset())
        result = rg.generate_clarification("hint", _session(), "English")
        assert result == "Could you clarify?"


# ── 8.3: No-results response ─────────────────────────────────────────────────


class TestNoResultsResponse:
    def test_returns_string(self):
        from sommelier.application.response_generator import ResponseGenerator
        no_result = NoResultsResult(reason="no_matching_titles", suggestion="Try relaxing filters.")
        rg = ResponseGenerator(_llm("No titles found."), _dataset())
        result = rg.generate_no_results_response(no_result, "English")
        assert isinstance(result, str)

    def test_uses_generation_model(self):
        from sommelier.application.response_generator import ResponseGenerator
        from sommelier.ports.interfaces import LLMRequest
        llm = _llm()
        no_result = NoResultsResult(reason="all_seen", suggestion="Broaden your search.")
        rg = ResponseGenerator(llm, _dataset())
        rg.generate_no_results_response(no_result, "English")
        req: LLMRequest = llm.complete.call_args[0][0]
        assert req.model == "generation"

    def test_suggestion_included_in_request(self):
        from sommelier.application.response_generator import ResponseGenerator
        from sommelier.ports.interfaces import LLMRequest
        llm = _llm()
        no_result = NoResultsResult(reason="all_seen", suggestion="Try a different genre.")
        rg = ResponseGenerator(llm, _dataset())
        rg.generate_no_results_response(no_result, "English")
        req: LLMRequest = llm.complete.call_args[0][0]
        full_text = req.system_prompt + " ".join(m.content for m in req.messages)
        assert "Try a different genre." in full_text

    def test_returns_llm_content(self):
        from sommelier.application.response_generator import ResponseGenerator
        no_result = NoResultsResult(reason="no_matching_titles", suggestion="Relax filters.")
        rg = ResponseGenerator(_llm("Sorry, nothing found."), _dataset())
        result = rg.generate_no_results_response(no_result, "English")
        assert result == "Sorry, nothing found."


# ── 8.1/8.3: Language injection ──────────────────────────────────────────────


class TestLanguageInjection:
    def _get_system_prompt(self, llm) -> str:
        from sommelier.ports.interfaces import LLMRequest
        req: LLMRequest = llm.complete.call_args[0][0]
        return req.system_prompt

    def test_recommendations_response_injects_language(self):
        from sommelier.application.response_generator import ResponseGenerator
        llm = _llm()
        rg = ResponseGenerator(llm, _dataset())
        rg.generate_recommendations_response([_recommendation()], _session(), "Portuguese")
        assert "Portuguese" in self._get_system_prompt(llm)

    def test_clarification_injects_language(self):
        from sommelier.application.response_generator import ResponseGenerator
        llm = _llm()
        rg = ResponseGenerator(llm, _dataset())
        rg.generate_clarification("hint", _session(), "Spanish")
        assert "Spanish" in self._get_system_prompt(llm)

    def test_no_results_response_injects_language(self):
        from sommelier.application.response_generator import ResponseGenerator
        llm = _llm()
        no_result = NoResultsResult(reason="all_seen", suggestion="Broaden.")
        rg = ResponseGenerator(llm, _dataset())
        rg.generate_no_results_response(no_result, "French")
        assert "French" in self._get_system_prompt(llm)

    def test_title_detail_injects_language(self):
        from sommelier.application.response_generator import ResponseGenerator
        llm = _llm()
        rg = ResponseGenerator(llm, _dataset(_title()))
        rg.generate_title_detail_response(_title(), "question", "Japanese")
        assert "Japanese" in self._get_system_prompt(llm)
