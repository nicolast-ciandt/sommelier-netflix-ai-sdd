"""CLIAdapter — Interface adapter for the Rich-formatted terminal conversation.

Tasks covered:
  10.1 — render_assistant_message(), render_user_turn(), render_startup()
  10.2 — run_conversation_loop(): input loop, quit/exit, KeyboardInterrupt/EOFError
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

_FAREWELL = "Thanks for using Sommelier. Enjoy your watching! 🎬"

_default_console = Console()


# ── 10.1: Display helpers ─────────────────────────────────────────────────────


def render_assistant_message(message: str, *, console: Console | None = None) -> None:
    """Render the assistant response inside a Rich panel labelled 'Sommelier'."""
    con = console or _default_console
    con.print(Panel(message, title="[bold cyan]Sommelier[/bold cyan]", border_style="cyan"))


def render_user_turn(message: str, *, console: Console | None = None) -> None:
    """Render the user's message with a contrasting style."""
    con = console or _default_console
    con.print(Text(f"You: {message}", style="bold white"))


def render_startup(
    title_count: int,
    database_url: str,
    *,
    console: Console | None = None,
) -> None:
    """Display startup information: database source and loaded title count."""
    con = console or _default_console
    # Redact credentials from the URL for display
    import re
    safe_url = re.sub(r"://[^@]+@", "://<credentials>@", database_url)
    con.print(
        Panel(
            f"Database: [bold]{safe_url}[/bold]\n"
            f"Titles loaded: [bold green]{title_count}[/bold green]",
            title="[bold cyan]Sommelier[/bold cyan] — Starting up",
            border_style="cyan",
        )
    )


# ── 10.2: Conversation loop ───────────────────────────────────────────────────


def run_conversation_loop(
    orchestrator,
    *,
    console: Console | None = None,
) -> None:
    """Run the main REPL: read user input, pass to orchestrator, display response.

    Exits cleanly on:
      - user typing 'quit' or 'exit'
      - KeyboardInterrupt (Ctrl-C)
      - EOFError (piped input exhausted)
    """
    con = console or _default_console

    session, greeting = orchestrator.start_session()
    render_assistant_message(greeting, console=con)

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            render_assistant_message(_FAREWELL, console=con)
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit"}:
            render_assistant_message(_FAREWELL, console=con)
            break

        render_user_turn(user_input, console=con)
        session, response = orchestrator.handle_turn(user_input, session)
        render_assistant_message(response, console=con)
