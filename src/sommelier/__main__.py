"""Entry point — boots the app and hands control to the CLI adapter.

Run with: python -m sommelier
"""

import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

from sommelier import debug
from sommelier.main import build_app
from sommelier.interface.cli_adapter import render_startup, run_conversation_loop


def main() -> None:
    if debug.is_enabled():
        debug.log("startup", "Debug mode enabled")
    orchestrator = build_app()
    database_url = os.environ.get("DATABASE_URL", "")
    title_count = orchestrator._dataset.title_count()
    render_startup(title_count=title_count, database_url=database_url)
    run_conversation_loop(orchestrator)


if __name__ == "__main__":
    main()
