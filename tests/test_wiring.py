"""Task 11.1 — Verify application wiring and entry point assembly.

Tests that build_app() wires all components correctly without a real
API key or dataset file — we mock the filesystem and env vars.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestBuildApp:
    def test_build_app_importable(self):
        from sommelier.main import build_app  # noqa

    def test_build_app_returns_orchestrator_and_cli(self, tmp_path, monkeypatch):
        from sommelier.main import build_app
        from sommelier.application.conversation_orchestrator import ConversationOrchestrator
        from sommelier.interface.cli_adapter import run_conversation_loop

        # Create a minimal valid CSV so DatasetStore can load it
        csv = tmp_path / "netflix.csv"
        csv.write_text(
            "show_id,type,title,director,cast,country,date_added,"
            "release_year,rating,duration,listed_in,description\n"
            "s1,Movie,Test Film,Dir,Actor,USA,Jan 1 2020,2020,PG,90 min,Drama,A film.\n"
        )
        monkeypatch.setenv("DATASET_PATH", str(csv))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

        with patch("sommelier.infrastructure.claude_adapter.anthropic.Anthropic"):
            orch = build_app()

        assert isinstance(orch, ConversationOrchestrator)

    def test_missing_dataset_raises_on_build(self, tmp_path, monkeypatch):
        from sommelier.main import build_app
        from sommelier.domain.models import DatasetLoadError

        monkeypatch.setenv("DATASET_PATH", str(tmp_path / "nonexistent.csv"))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

        with patch("sommelier.infrastructure.claude_adapter.anthropic.Anthropic"):
            with pytest.raises(DatasetLoadError):
                build_app()
