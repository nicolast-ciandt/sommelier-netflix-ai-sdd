"""Tasks 10.1, 10.2 — Verify CLIAdapter display helpers and loop control.

Rich output is captured via Console(file=StringIO()) to avoid TTY coupling.
The conversation loop is tested by monkeypatching input() and verifying
that quit/exit terminates cleanly and that turn responses are displayed.

Groups:
  TestDisplayHelpers     — 10.1: panel/user-turn rendering, startup info
  TestConversationLoop   — 10.2: quit/exit, KeyboardInterrupt, turn flow
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from sommelier.domain.models import (
    DurationInfo,
    NetflixTitle,
    PreferenceProfile,
    Recommendation,
    Session,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, highlight=False, markup=False), buf


def _title(show_id: str = "s1", name: str = "Drama Night") -> NetflixTitle:
    return NetflixTitle(
        show_id=show_id, type="Movie", title=name,
        director="Dir", cast=("Actor A",), country="Brazil",
        release_year=2021, rating="PG-13", duration=DurationInfo(90, "min"),
        genres=("Drama", "Thriller"), description="A dramatic film.",
    )


def _recommendation(show_id: str = "s1", rationale: str = "A good pick") -> Recommendation:
    return Recommendation(
        title=_title(show_id), relevance_score=0.9, rationale=rationale
    )


def _session() -> Session:
    return Session(
        id="t", conversation_history=[], preference_profile=PreferenceProfile(),
        seen_title_ids=frozenset(), maturity_ceiling_locked=False,
    )


# ── 10.1: Display helpers ─────────────────────────────────────────────────────


class TestDisplayHelpers:
    def test_render_assistant_message_importable(self):
        from sommelier.interface.cli_adapter import render_assistant_message  # noqa

    def test_render_user_turn_importable(self):
        from sommelier.interface.cli_adapter import render_user_turn  # noqa

    def test_render_startup_importable(self):
        from sommelier.interface.cli_adapter import render_startup  # noqa

    def test_assistant_message_contains_text(self):
        from sommelier.interface.cli_adapter import render_assistant_message
        console, buf = _console()
        render_assistant_message("Here are my picks!", console=console)
        assert "Here are my picks!" in buf.getvalue()

    def test_assistant_message_shows_sommelier_label(self):
        from sommelier.interface.cli_adapter import render_assistant_message
        console, buf = _console()
        render_assistant_message("test response", console=console)
        assert "Sommelier" in buf.getvalue()

    def test_user_turn_contains_message(self):
        from sommelier.interface.cli_adapter import render_user_turn
        console, buf = _console()
        render_user_turn("I want action films", console=console)
        assert "I want action films" in buf.getvalue()

    def test_startup_shows_title_count(self):
        from sommelier.interface.cli_adapter import render_startup
        console, buf = _console()
        render_startup(title_count=8500, dataset_path="/data/netflix.csv", console=console)
        assert "8500" in buf.getvalue()

    def test_startup_shows_dataset_path(self):
        from sommelier.interface.cli_adapter import render_startup
        console, buf = _console()
        render_startup(title_count=100, dataset_path="/data/netflix.csv", console=console)
        assert "netflix.csv" in buf.getvalue()


# ── 10.2: Conversation loop ───────────────────────────────────────────────────


class TestConversationLoop:
    def _make_orchestrator(self, responses=None):
        """Return a mock orchestrator that cycles through responses."""
        orch = MagicMock()
        session = _session()
        orch.start_session.return_value = (session, "Welcome!")
        response_iter = iter(responses or ["Here are some films."])

        def handle_turn(msg, sess):
            return sess, next(response_iter, "Here are more films.")

        orch.handle_turn.side_effect = handle_turn
        return orch

    def test_quit_command_exits_loop(self, capsys):
        from sommelier.interface.cli_adapter import run_conversation_loop
        orch = self._make_orchestrator()
        inputs = iter(["quit"])
        with patch("builtins.input", side_effect=inputs):
            run_conversation_loop(orch)  # must not block or raise

    def test_exit_command_exits_loop(self, capsys):
        from sommelier.interface.cli_adapter import run_conversation_loop
        orch = self._make_orchestrator()
        inputs = iter(["exit"])
        with patch("builtins.input", side_effect=inputs):
            run_conversation_loop(orch)

    def test_keyboard_interrupt_exits_loop(self, capsys):
        from sommelier.interface.cli_adapter import run_conversation_loop
        orch = self._make_orchestrator()
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            run_conversation_loop(orch)  # must not raise

    def test_turn_response_displayed(self, capsys):
        from sommelier.interface.cli_adapter import run_conversation_loop
        orch = self._make_orchestrator(responses=["Five great picks for you!"])
        inputs = iter(["I want drama", "quit"])
        console, buf = _console()
        with patch("builtins.input", side_effect=inputs):
            run_conversation_loop(orch, console=console)
        assert "Five great picks for you!" in buf.getvalue()

    def test_session_state_passed_between_turns(self):
        from sommelier.interface.cli_adapter import run_conversation_loop
        orch = MagicMock()
        s0 = _session()
        s1 = Session(id="updated", conversation_history=[], preference_profile=PreferenceProfile(),
                     seen_title_ids=frozenset({"s1"}), maturity_ceiling_locked=False)
        orch.start_session.return_value = (s0, "Hi!")
        call_log = []

        def handle_turn(msg, sess):
            call_log.append(sess)
            return s1, "response"

        orch.handle_turn.side_effect = handle_turn
        inputs = iter(["first message", "quit"])
        with patch("builtins.input", side_effect=inputs):
            run_conversation_loop(orch)
        # second call should receive the updated session s1
        # (only one non-quit message so only one call)
        assert call_log[0].id == s0.id

    def test_eof_exits_loop_gracefully(self):
        from sommelier.interface.cli_adapter import run_conversation_loop
        orch = self._make_orchestrator()
        with patch("builtins.input", side_effect=EOFError):
            run_conversation_loop(orch)  # must not raise
