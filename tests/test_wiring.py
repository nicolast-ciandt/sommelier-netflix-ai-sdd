"""Task 11.1 — Verify application wiring and entry point assembly."""

from unittest.mock import MagicMock, patch

import pytest


class TestBuildApp:
    def test_build_app_importable(self):
        from sommelier.main import build_app  # noqa

    def test_build_app_returns_orchestrator(self, monkeypatch):
        from sommelier.main import build_app
        from sommelier.application.conversation_orchestrator import ConversationOrchestrator

        monkeypatch.setenv("DATABASE_URL", "postgresql://fake/db")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: s
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            {"show_id": "s1", "type": "Movie", "title": "Test Film", "director": "Dir",
             "cast": "Actor", "country": "USA", "release_year": "2020", "rating": "PG",
             "duration": "90 min", "listed_in": "Drama", "description": "A film."},
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch("psycopg2.connect", return_value=mock_conn), \
             patch("sommelier.infrastructure.claude_adapter.anthropic.Anthropic"):
            orch = build_app()

        assert isinstance(orch, ConversationOrchestrator)

    def test_missing_database_url_raises_on_build(self, monkeypatch):
        from sommelier.main import build_app
        from sommelier.domain.models import DatasetLoadError

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        monkeypatch.delenv("DATABASE_URL", raising=False)

        with patch("psycopg2.connect", side_effect=Exception("no host")), \
             patch("sommelier.infrastructure.claude_adapter.anthropic.Anthropic"):
            with pytest.raises(DatasetLoadError):
                build_app()
