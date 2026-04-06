"""Sommelier MCP Server — exposes the recommendation engine as remote MCP tools.

Run locally:
    python -m sommelier.mcp_server

Clients connect via HTTP to: http://localhost:8000/mcp
On Render: https://sommelier-mcp.onrender.com/mcp
"""

from __future__ import annotations

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

from mcp.server.fastmcp import FastMCP

from sommelier.main import build_app

import os as _os
mcp = FastMCP(
    "sommelier",
    host=_os.environ.get("HOST", "0.0.0.0"),
    port=int(_os.environ.get("PORT", 8000)),
)

# Load dataset and wire all components once at startup
_app = build_app()


@mcp.tool()
def recommend(query: str) -> str:
    """Recommend Netflix titles based on a natural language preference.

    Args:
        query: Natural language description of what to watch.
               Example: "dark psychological thriller from the 90s"
               Example: "feel-good romantic comedy for a Friday night"
    """
    session, _ = _app.start_session()
    _, response = _app.handle_turn(query, session)
    return response


@mcp.tool()
def get_title_details(title_name: str) -> str:
    """Get details about a specific Netflix title by name.

    Args:
        title_name: The name of the Netflix title to look up.
                    Example: "Stranger Things"
    """
    titles = _app._dataset._titles_list
    name_lower = title_name.lower()

    # Exact match first, then partial
    match = next((t for t in titles if t.title.lower() == name_lower), None)
    if not match:
        match = next((t for t in titles if name_lower in t.title.lower()), None)
    if not match:
        return f"Title '{title_name}' not found in the Netflix catalog."

    t = match
    cast = ", ".join(t.cast[:5]) if t.cast else "N/A"
    duration = f"{t.duration.value} {t.duration.unit}" if t.duration else "N/A"
    return (
        f"**{t.title}** ({t.release_year})\n"
        f"Type: {t.type} | Rating: {t.rating or 'N/A'} | Duration: {duration}\n"
        f"Genres: {', '.join(t.genres)}\n"
        f"Director: {t.director or 'N/A'}\n"
        f"Cast: {cast}\n"
        f"Country: {t.country or 'N/A'}\n\n"
        f"{t.description}"
    )


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
