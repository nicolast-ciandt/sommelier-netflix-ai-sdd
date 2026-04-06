"""Task 1.2 — Verify all shared domain data types are correctly defined."""

import dataclasses
from dataclasses import FrozenInstanceError

import pytest

from sommelier.domain.models import (
    DatasetLoadError,
    DurationInfo,
    ExtractionResult,
    FeedbackResult,
    LLMUnavailableError,
    Message,
    NetflixTitle,
    NoResultsResult,
    PreferenceProfile,
    PreferenceProfileDelta,
    Recommendation,
    RecommendationOutput,
    Recommendation,
    ScoredTitle,
    Session,
)


# ── DurationInfo ──────────────────────────────────────────────────────────────

class TestDurationInfo:
    def test_instantiation_with_minutes(self):
        d = DurationInfo(value=90, unit="min")
        assert d.value == 90
        assert d.unit == "min"

    def test_instantiation_with_seasons(self):
        d = DurationInfo(value=3, unit="Seasons")
        assert d.value == 3
        assert d.unit == "Seasons"

    def test_is_frozen(self):
        d = DurationInfo(value=90, unit="min")
        with pytest.raises(FrozenInstanceError):
            d.value = 120  # type: ignore[misc]

    def test_equality(self):
        assert DurationInfo(90, "min") == DurationInfo(90, "min")
        assert DurationInfo(90, "min") != DurationInfo(91, "min")


# ── NetflixTitle ──────────────────────────────────────────────────────────────

class TestNetflixTitle:
    def _make_title(self, **overrides):
        defaults = dict(
            show_id="s1",
            type="Movie",
            title="Test Film",
            director="Jane Doe",
            cast=("Actor A", "Actor B"),
            country="United States",
            release_year=2020,
            rating="PG-13",
            duration=DurationInfo(120, "min"),
            genres=("Drama", "Thriller"),
            description="A gripping drama.",
        )
        defaults.update(overrides)
        return NetflixTitle(**defaults)

    def test_instantiation_with_all_fields(self):
        t = self._make_title()
        assert t.show_id == "s1"
        assert t.title == "Test Film"
        assert t.type == "Movie"
        assert t.release_year == 2020

    def test_nullable_fields_accept_none(self):
        t = self._make_title(director=None, country=None, rating=None, duration=None)
        assert t.director is None
        assert t.country is None
        assert t.rating is None
        assert t.duration is None

    def test_genres_and_cast_are_sequences(self):
        t = self._make_title()
        assert "Drama" in t.genres
        assert "Actor A" in t.cast

    def test_is_frozen(self):
        t = self._make_title()
        with pytest.raises(FrozenInstanceError):
            t.title = "Other"  # type: ignore[misc]

    def test_equality_by_value(self):
        t1 = self._make_title()
        t2 = self._make_title()
        assert t1 == t2

    def test_type_discriminator_values(self):
        movie = self._make_title(type="Movie")
        series = self._make_title(type="TV Show")
        assert movie.type == "Movie"
        assert series.type == "TV Show"


# ── Message ───────────────────────────────────────────────────────────────────

class TestMessage:
    def test_instantiation(self):
        m = Message(role="user", content="Hello")
        assert m.role == "user"
        assert m.content == "Hello"

    def test_assistant_role(self):
        m = Message(role="assistant", content="Hi there")
        assert m.role == "assistant"

    def test_is_frozen(self):
        m = Message(role="user", content="Hello")
        with pytest.raises(FrozenInstanceError):
            m.content = "Other"  # type: ignore[misc]

    def test_equality(self):
        assert Message("user", "Hello") == Message("user", "Hello")
        assert Message("user", "Hello") != Message("user", "Hi")


# ── PreferenceProfile ─────────────────────────────────────────────────────────

class TestPreferenceProfile:
    def test_instantiation_with_defaults(self):
        p = PreferenceProfile()
        assert p.genres == []
        assert p.mood_keywords == []
        assert p.content_type is None
        assert p.year_min is None
        assert p.year_max is None
        assert p.maturity_ceiling is None
        assert p.country_filter is None
        assert p.positive_genre_signals == []

    def test_instantiation_with_values(self):
        p = PreferenceProfile(
            genres=["Drama", "Thriller"],
            mood_keywords=["dark", "suspenseful"],
            content_type="Movie",
            year_min=2000,
            year_max=2022,
            maturity_ceiling="PG-13",
            country_filter="United States",
            positive_genre_signals=["Drama"],
        )
        assert p.genres == ["Drama", "Thriller"]
        assert p.content_type == "Movie"
        assert p.year_min == 2000

    def test_default_lists_are_independent(self):
        # Each instance should get its own list, not share one
        p1 = PreferenceProfile()
        p2 = PreferenceProfile()
        p1.genres.append("Drama")
        assert p2.genres == []


# ── PreferenceProfileDelta ────────────────────────────────────────────────────

class TestPreferenceProfileDelta:
    def test_instantiation_with_defaults(self):
        delta = PreferenceProfileDelta()
        assert delta.genres == ()
        assert delta.mood_keywords == ()
        assert delta.content_type is None
        assert delta.excluded_title_ids == ()
        assert delta.positive_genre_signals == ()
        assert delta.needs_clarification is False
        assert delta.clarification_hint is None
        assert delta.has_conflict is False
        assert delta.conflict_description is None

    def test_instantiation_with_values(self):
        delta = PreferenceProfileDelta(
            genres=("Action", "Comedy"),
            mood_keywords=("fun", "light"),
            content_type="Movie",
            year_min=2010,
            year_max=2023,
            maturity_ceiling="PG",
            country_filter="UK",
            excluded_title_ids=("s1", "s2"),
            positive_genre_signals=("Action",),
            needs_clarification=True,
            clarification_hint="Did you mean animated?",
            has_conflict=False,
            conflict_description=None,
        )
        assert delta.genres == ("Action", "Comedy")
        assert delta.needs_clarification is True
        assert delta.clarification_hint == "Did you mean animated?"

    def test_is_frozen(self):
        delta = PreferenceProfileDelta()
        with pytest.raises(FrozenInstanceError):
            delta.needs_clarification = True  # type: ignore[misc]

    def test_conflicting_signals(self):
        delta = PreferenceProfileDelta(
            has_conflict=True,
            conflict_description="Action requested but also slow-paced",
        )
        assert delta.has_conflict is True
        assert delta.conflict_description == "Action requested but also slow-paced"


# ── Session ───────────────────────────────────────────────────────────────────

class TestSession:
    def test_instantiation_requires_id(self):
        s = Session(id="abc-123")
        assert s.id == "abc-123"

    def test_default_conversation_history_is_empty(self):
        s = Session(id="abc")
        assert s.conversation_history == []

    def test_default_seen_title_ids_is_empty(self):
        s = Session(id="abc")
        assert s.seen_title_ids == frozenset()

    def test_default_maturity_ceiling_locked_is_false(self):
        s = Session(id="abc")
        assert s.maturity_ceiling_locked is False

    def test_default_preference_profile_is_empty(self):
        s = Session(id="abc")
        assert s.preference_profile.genres == []

    def test_has_conversation_history_and_profile(self):
        msg = Message(role="user", content="Hi")
        profile = PreferenceProfile(genres=["Drama"])
        s = Session(
            id="xyz",
            conversation_history=[msg],
            preference_profile=profile,
            seen_title_ids=frozenset({"s1"}),
            maturity_ceiling_locked=True,
        )
        assert len(s.conversation_history) == 1
        assert s.conversation_history[0].role == "user"
        assert s.preference_profile.genres == ["Drama"]
        assert "s1" in s.seen_title_ids
        assert s.maturity_ceiling_locked is True

    def test_default_histories_are_independent(self):
        s1 = Session(id="a")
        s2 = Session(id="b")
        s1.conversation_history.append(Message("user", "test"))
        assert s2.conversation_history == []


# ── ScoredTitle ───────────────────────────────────────────────────────────────

class TestScoredTitle:
    def _make_title(self):
        return NetflixTitle(
            show_id="s1", type="Movie", title="T", director=None,
            cast=(), country=None, release_year=2020, rating=None,
            duration=None, genres=(), description="desc",
        )

    def test_instantiation(self):
        st = ScoredTitle(title=self._make_title(), similarity_score=0.85)
        assert st.similarity_score == 0.85
        assert st.title.show_id == "s1"

    def test_boundary_scores(self):
        ScoredTitle(title=self._make_title(), similarity_score=0.0)
        ScoredTitle(title=self._make_title(), similarity_score=1.0)

    def test_is_frozen(self):
        st = ScoredTitle(title=self._make_title(), similarity_score=0.5)
        with pytest.raises(FrozenInstanceError):
            st.similarity_score = 0.9  # type: ignore[misc]


# ── Recommendation ────────────────────────────────────────────────────────────

class TestRecommendation:
    def _make_title(self):
        return NetflixTitle(
            show_id="s2", type="TV Show", title="Series X", director=None,
            cast=(), country=None, release_year=2019, rating=None,
            duration=None, genres=(), description="desc",
        )

    def test_instantiation_with_empty_rationale_default(self):
        r = Recommendation(title=self._make_title(), relevance_score=0.9)
        assert r.rationale == ""
        assert r.relevance_score == 0.9

    def test_instantiation_with_rationale(self):
        r = Recommendation(
            title=self._make_title(),
            relevance_score=0.75,
            rationale="Matches your taste for suspense.",
        )
        assert r.rationale == "Matches your taste for suspense."

    def test_title_fields_are_accessible(self):
        r = Recommendation(title=self._make_title(), relevance_score=0.8)
        assert r.title.type == "TV Show"
        assert r.title.release_year == 2019


# ── NoResultsResult ───────────────────────────────────────────────────────────

class TestNoResultsResult:
    def test_no_matching_titles_reason(self):
        r = NoResultsResult(
            reason="no_matching_titles",
            suggestion="Try relaxing the genre filter.",
        )
        assert r.reason == "no_matching_titles"
        assert r.suggestion == "Try relaxing the genre filter."

    def test_all_seen_reason(self):
        r = NoResultsResult(
            reason="all_seen",
            suggestion="You've seen all matching titles.",
        )
        assert r.reason == "all_seen"

    def test_is_frozen(self):
        r = NoResultsResult(reason="all_seen", suggestion="Try broadening.")
        with pytest.raises(FrozenInstanceError):
            r.reason = "no_matching_titles"  # type: ignore[misc]


# ── Error Types ───────────────────────────────────────────────────────────────

class TestErrorTypes:
    def test_dataset_load_error_is_exception(self):
        err = DatasetLoadError("data/netflix.csv not found")
        assert isinstance(err, Exception)
        assert str(err) == "data/netflix.csv not found"

    def test_llm_unavailable_error_is_exception(self):
        err = LLMUnavailableError("API timeout after 5s")
        assert isinstance(err, Exception)
        assert str(err) == "API timeout after 5s"

    def test_dataset_load_error_can_be_raised_and_caught(self):
        with pytest.raises(DatasetLoadError, match="not found"):
            raise DatasetLoadError("file not found")

    def test_llm_unavailable_error_can_be_raised_and_caught(self):
        with pytest.raises(LLMUnavailableError, match="timeout"):
            raise LLMUnavailableError("connection timeout")


# ── Type Aliases ──────────────────────────────────────────────────────────────

class TestTypeAliases:
    def test_extraction_result_is_preference_profile_delta(self):
        delta = PreferenceProfileDelta(genres=("Action",))
        # ExtractionResult is a type alias; instances are PreferenceProfileDelta
        assert isinstance(delta, PreferenceProfileDelta)
        result: ExtractionResult = delta
        assert result.genres == ("Action",)

    def test_feedback_result_is_preference_profile_delta(self):
        delta = PreferenceProfileDelta(excluded_title_ids=("s1",))
        result: FeedbackResult = delta
        assert result.excluded_title_ids == ("s1",)

    def test_recommendation_output_can_be_list_of_recommendations(self):
        title = NetflixTitle(
            show_id="s3", type="Movie", title="Film", director=None,
            cast=(), country=None, release_year=2021, rating=None,
            duration=None, genres=(), description="desc",
        )
        recs: RecommendationOutput = [Recommendation(title=title, relevance_score=0.8)]
        assert isinstance(recs, list)

    def test_recommendation_output_can_be_no_results_result(self):
        no_results: RecommendationOutput = NoResultsResult(
            reason="no_matching_titles", suggestion="Broaden search."
        )
        assert isinstance(no_results, NoResultsResult)
