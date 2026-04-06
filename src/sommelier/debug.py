"""Debug / verbose logging utility.

Reads DEBUG from the environment (set via .env before import).
All debug output goes to stderr so it doesn't pollute stdout/Rich output.
"""

from __future__ import annotations

import os
import sys
import traceback

_enabled: bool = os.environ.get("DEBUG", "").strip().lower() in {"1", "true", "yes"}


def is_enabled() -> bool:
    return _enabled


def log(tag: str, message: str) -> None:
    """Print a debug line to stderr when DEBUG is enabled."""
    if _enabled:
        print(f"[DEBUG:{tag}] {message}", file=sys.stderr)


def log_exception(tag: str, exc: BaseException) -> None:
    """Print full traceback to stderr when DEBUG is enabled."""
    if _enabled:
        print(f"[DEBUG:{tag}] Exception: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
