"""Entry point — boots the app and hands control to the CLI adapter.

Run with: python -m sommelier
"""

from sommelier.main import build_app
from sommelier.interface.cli_adapter import render_startup, run_conversation_loop
import os


def main() -> None:
    orchestrator = build_app()
    dataset_path = os.environ.get("DATASET_PATH", "netflix_titles.csv")
    title_count = orchestrator._dataset.title_count()
    render_startup(title_count=title_count, dataset_path=dataset_path)
    run_conversation_loop(orchestrator)


if __name__ == "__main__":
    main()
