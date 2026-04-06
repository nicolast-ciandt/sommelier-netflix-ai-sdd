"""Tasks 4.1, 4.2, 4.3 — Verify SessionManager state transitions.

Tests are grouped by task:
  TestSessionCreation        — 4.1
  TestAppendMessage          — 4.1
  TestHistoryTruncation      — 4.1
  TestApplyDelta             — 4.2
  TestRegisterShownTitles    — 4.3
  TestApplyRejectedTitles    — 4.3
  TestLockMaturityCeiling    — 4.3
"""

import pytest

from sommelier.domain.models import (
    Message,
    PreferenceProfile,
    PreferenceProfileDelta,
    Session,
)


# ── 4.1: Session creation ─────────────────────────────────────────────────────


class TestSessionCreation:
    def test_create_session_returns_session(self):
        from sommelier.application.session_manager import SessionManager
        sm = SessionManager(max_history_turns=10)
        session = sm.create_session()
        assert isinstance(session, Session)

    def test_session_has_unique_uuid_id(self):
        from sommelier.application.session_manager import SessionManager
        sm = SessionManager(max_history_turns=10)
        ids = {sm.create_session().id for _ in range(5)}
        assert len(ids) == 5

    def test_session_starts_with_empty_history(self):
        from sommelier.application.session_manager import SessionManager
        sm = SessionManager(max_history_turns=10)
        assert sm.create_session().conversation_history == []

    def test_session_starts_with_empty_preference_profile(self):
        from sommelier.application.session_manager import SessionManager
        sm = SessionManager(max_history_turns=10)
        profile = sm.create_session().preference_profile
        assert isinstance(profile, PreferenceProfile)
        assert profile.genres == []
        assert profile.mood_keywords == []
        assert profile.content_type is None

    def test_session_starts_with_empty_seen_title_ids(self):
        from sommelier.application.session_manager import SessionManager
        sm = SessionManager(max_history_turns=10)
        assert sm.create_session().seen_title_ids == frozenset()

    def test_session_starts_with_maturity_ceiling_unlocked(self):
        from sommelier.application.session_manager import SessionManager
        sm = SessionManager(max_history_turns=10)
        assert sm.create_session().maturity_ceiling_locked is False


# ── 4.1: append_message ───────────────────────────────────────────────────────


class TestAppendMessage:
    @pytest.fixture
    def sm(self):
        from sommelier.application.session_manager import SessionManager
        return SessionManager(max_history_turns=10)

    def test_append_returns_new_session_instance(self, sm):
        s0 = sm.create_session()
        s1 = sm.append_message(s0, "user", "Hello")
        assert s1 is not s0

    def test_append_does_not_mutate_original(self, sm):
        s0 = sm.create_session()
        sm.append_message(s0, "user", "Hello")
        assert s0.conversation_history == []

    def test_appended_message_appears_in_history(self, sm):
        s0 = sm.create_session()
        s1 = sm.append_message(s0, "user", "Hi there")
        assert len(s1.conversation_history) == 1
        assert s1.conversation_history[0].role == "user"
        assert s1.conversation_history[0].content == "Hi there"

    def test_message_is_message_instance(self, sm):
        s0 = sm.create_session()
        s1 = sm.append_message(s0, "assistant", "Hello!")
        assert isinstance(s1.conversation_history[0], Message)

    def test_multiple_appends_accumulate_in_order(self, sm):
        s = sm.create_session()
        s = sm.append_message(s, "user", "first")
        s = sm.append_message(s, "assistant", "second")
        s = sm.append_message(s, "user", "third")
        assert len(s.conversation_history) == 3
        assert s.conversation_history[0].content == "first"
        assert s.conversation_history[2].content == "third"

    def test_append_preserves_other_session_fields(self, sm):
        s0 = sm.create_session()
        s1 = sm.append_message(s0, "user", "Hi")
        assert s1.id == s0.id
        assert s1.preference_profile == s0.preference_profile
        assert s1.seen_title_ids == s0.seen_title_ids


# ── 4.1: History truncation ───────────────────────────────────────────────────


class TestHistoryTruncation:
    def test_history_never_exceeds_max_turns(self):
        from sommelier.application.session_manager import SessionManager
        sm = SessionManager(max_history_turns=3)
        s = sm.create_session()
        for i in range(6):
            s = sm.append_message(s, "user", f"msg {i}")
        assert len(s.conversation_history) == 3

    def test_truncation_keeps_most_recent_messages(self):
        from sommelier.application.session_manager import SessionManager
        sm = SessionManager(max_history_turns=3)
        s = sm.create_session()
        for i in range(5):
            s = sm.append_message(s, "user", f"msg {i}")
        contents = [m.content for m in s.conversation_history]
        assert contents == ["msg 2", "msg 3", "msg 4"]

    def test_history_at_exact_limit_is_not_truncated(self):
        from sommelier.application.session_manager import SessionManager
        sm = SessionManager(max_history_turns=4)
        s = sm.create_session()
        for i in range(4):
            s = sm.append_message(s, "user", f"msg {i}")
        assert len(s.conversation_history) == 4

    def test_default_max_turns_read_from_env(self, monkeypatch):
        monkeypatch.setenv("MAX_HISTORY_TURNS", "2")
        from sommelier.application.session_manager import SessionManager
        sm = SessionManager()
        s = sm.create_session()
        for i in range(5):
            s = sm.append_message(s, "user", f"msg {i}")
        assert len(s.conversation_history) == 2


# ── 4.2: apply_delta ─────────────────────────────────────────────────────────


class TestApplyDelta:
    @pytest.fixture
    def sm(self):
        from sommelier.application.session_manager import SessionManager
        return SessionManager(max_history_turns=10)

    def test_apply_delta_returns_new_session(self, sm):
        s0 = sm.create_session()
        delta = PreferenceProfileDelta(genres=("Drama",))
        s1 = sm.apply_delta(s0, delta)
        assert s1 is not s0

    def test_genres_are_appended_additively(self, sm):
        s = sm.create_session()
        s = sm.apply_delta(s, PreferenceProfileDelta(genres=("Drama",)))
        s = sm.apply_delta(s, PreferenceProfileDelta(genres=("Comedy",)))
        assert "Drama" in s.preference_profile.genres
        assert "Comedy" in s.preference_profile.genres

    def test_duplicate_genres_not_added(self, sm):
        s = sm.create_session()
        s = sm.apply_delta(s, PreferenceProfileDelta(genres=("Drama",)))
        s = sm.apply_delta(s, PreferenceProfileDelta(genres=("Drama",)))
        assert s.preference_profile.genres.count("Drama") == 1

    def test_mood_keywords_are_appended_additively(self, sm):
        s = sm.create_session()
        s = sm.apply_delta(s, PreferenceProfileDelta(mood_keywords=("dark",)))
        s = sm.apply_delta(s, PreferenceProfileDelta(mood_keywords=("funny",)))
        assert "dark" in s.preference_profile.mood_keywords
        assert "funny" in s.preference_profile.mood_keywords

    def test_content_type_updated_when_provided(self, sm):
        s = sm.create_session()
        s = sm.apply_delta(s, PreferenceProfileDelta(content_type="Movie"))
        assert s.preference_profile.content_type == "Movie"

    def test_content_type_not_cleared_when_delta_is_none(self, sm):
        s = sm.create_session()
        s = sm.apply_delta(s, PreferenceProfileDelta(content_type="Movie"))
        s = sm.apply_delta(s, PreferenceProfileDelta(genres=("Drama",)))
        assert s.preference_profile.content_type == "Movie"

    def test_year_min_updated_when_provided(self, sm):
        s = sm.create_session()
        s = sm.apply_delta(s, PreferenceProfileDelta(year_min=2010))
        assert s.preference_profile.year_min == 2010

    def test_year_max_updated_when_provided(self, sm):
        s = sm.create_session()
        s = sm.apply_delta(s, PreferenceProfileDelta(year_max=2023))
        assert s.preference_profile.year_max == 2023

    def test_country_filter_updated_when_provided(self, sm):
        s = sm.create_session()
        s = sm.apply_delta(s, PreferenceProfileDelta(country_filter="Japan"))
        assert s.preference_profile.country_filter == "Japan"

    def test_positive_genre_signals_accumulated(self, sm):
        s = sm.create_session()
        s = sm.apply_delta(s, PreferenceProfileDelta(positive_genre_signals=("Action",)))
        s = sm.apply_delta(s, PreferenceProfileDelta(positive_genre_signals=("Drama",)))
        assert "Action" in s.preference_profile.positive_genre_signals
        assert "Drama" in s.preference_profile.positive_genre_signals

    def test_apply_delta_does_not_mutate_original(self, sm):
        s0 = sm.create_session()
        sm.apply_delta(s0, PreferenceProfileDelta(genres=("Drama",)))
        assert s0.preference_profile.genres == []

    def test_empty_delta_changes_nothing(self, sm):
        s0 = sm.create_session()
        s1 = sm.apply_delta(s0, PreferenceProfileDelta())
        assert s1.preference_profile.genres == []
        assert s1.preference_profile.content_type is None


# ── 4.3: register_shown_titles ────────────────────────────────────────────────


class TestRegisterShownTitles:
    @pytest.fixture
    def sm(self):
        from sommelier.application.session_manager import SessionManager
        return SessionManager(max_history_turns=10)

    def test_register_returns_new_session(self, sm):
        s0 = sm.create_session()
        s1 = sm.register_shown_titles(s0, ["s1"])
        assert s1 is not s0

    def test_registered_ids_appear_in_seen_set(self, sm):
        s = sm.create_session()
        s = sm.register_shown_titles(s, ["s1", "s2"])
        assert "s1" in s.seen_title_ids
        assert "s2" in s.seen_title_ids

    def test_ids_accumulate_across_rounds(self, sm):
        s = sm.create_session()
        s = sm.register_shown_titles(s, ["s1"])
        s = sm.register_shown_titles(s, ["s2", "s3"])
        assert s.seen_title_ids == frozenset({"s1", "s2", "s3"})

    def test_register_does_not_mutate_original(self, sm):
        s0 = sm.create_session()
        sm.register_shown_titles(s0, ["s1"])
        assert s0.seen_title_ids == frozenset()

    def test_empty_list_changes_nothing(self, sm):
        s0 = sm.create_session()
        s1 = sm.register_shown_titles(s0, [])
        assert s1.seen_title_ids == frozenset()


# ── 4.3: apply_rejected_titles ───────────────────────────────────────────────


class TestApplyRejectedTitles:
    @pytest.fixture
    def sm(self):
        from sommelier.application.session_manager import SessionManager
        return SessionManager(max_history_turns=10)

    def test_rejected_ids_added_to_seen_set(self, sm):
        s = sm.create_session()
        s = sm.apply_rejected_titles(s, ["s5"])
        assert "s5" in s.seen_title_ids

    def test_rejected_ids_accumulate_with_shown_ids(self, sm):
        s = sm.create_session()
        s = sm.register_shown_titles(s, ["s1"])
        s = sm.apply_rejected_titles(s, ["s2"])
        assert s.seen_title_ids == frozenset({"s1", "s2"})

    def test_apply_rejected_returns_new_session(self, sm):
        s0 = sm.create_session()
        s1 = sm.apply_rejected_titles(s0, ["s3"])
        assert s1 is not s0


# ── 4.3: lock_maturity_ceiling ────────────────────────────────────────────────


class TestLockMaturityCeiling:
    @pytest.fixture
    def sm(self):
        from sommelier.application.session_manager import SessionManager
        return SessionManager(max_history_turns=10)

    def test_lock_sets_ceiling_on_profile(self, sm):
        s = sm.create_session()
        s = sm.lock_maturity_ceiling(s, "PG-13")
        assert s.preference_profile.maturity_ceiling == "PG-13"

    def test_lock_sets_locked_flag(self, sm):
        s = sm.create_session()
        s = sm.lock_maturity_ceiling(s, "PG")
        assert s.maturity_ceiling_locked is True

    def test_locked_ceiling_cannot_be_raised(self, sm):
        s = sm.create_session()
        s = sm.lock_maturity_ceiling(s, "PG")
        s = sm.lock_maturity_ceiling(s, "TV-MA")  # attempt to raise — must be ignored
        assert s.preference_profile.maturity_ceiling == "PG"

    def test_locked_ceiling_cannot_be_replaced_at_all(self, sm):
        s = sm.create_session()
        s = sm.lock_maturity_ceiling(s, "G")
        s = sm.lock_maturity_ceiling(s, "G")  # same value — still ignored
        assert s.preference_profile.maturity_ceiling == "G"
        assert s.maturity_ceiling_locked is True

    def test_lock_returns_new_session(self, sm):
        s0 = sm.create_session()
        s1 = sm.lock_maturity_ceiling(s0, "PG")
        assert s1 is not s0

    def test_lock_does_not_mutate_original(self, sm):
        s0 = sm.create_session()
        sm.lock_maturity_ceiling(s0, "PG")
        assert s0.maturity_ceiling_locked is False
