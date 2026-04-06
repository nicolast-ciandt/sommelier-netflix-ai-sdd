"""ResponseGenerator — Application service for LLM-backed natural-language responses.

All responses use the Sonnet (generation) model and inject the user's
detected language into every system prompt.

Tasks covered:
  8.1 — generate_recommendations_response(): per-title rationale
  8.2 — generate_title_detail_response(), generate_catalog_miss_response()
  8.3 — generate_clarification(), generate_no_results_response()
"""

from __future__ import annotations

from sommelier.domain.models import (
    Message,
    NetflixTitle,
    NoResultsResult,
    Recommendation,
    Session,
)
from sommelier.ports.interfaces import DatasetPort, LLMPort, LLMRequest


class ResponseGenerator:
    """Generates all assistant-facing natural-language strings via the LLM."""

    def __init__(self, llm: LLMPort, dataset: DatasetPort) -> None:
        self._llm = llm
        self._dataset = dataset

    # ── 8.1: Recommendation response ─────────────────────────────────────────

    def generate_recommendations_response(
        self,
        recommendations: list[Recommendation],
        session: Session,
        user_language: str,
    ) -> str:
        """Generate a formatted recommendation list with per-title rationale."""
        titles_context = _format_recommendations(recommendations)
        system_prompt = (
            f"You are a friendly Netflix recommendation assistant. "
            f"Respond in {user_language}. "
            f"Generate an engaging, conversational response that presents each "
            f"recommended title with a brief personalized rationale. "
            f"Include the title name, type, year, and genres naturally in your response."
        )
        user_content = (
            f"Please present these recommendations to the user:\n\n{titles_context}"
        )
        return self._complete(system_prompt, user_content)

    # ── 8.2: Title detail and catalog-miss ───────────────────────────────────

    def generate_title_detail_response(
        self,
        title: NetflixTitle,
        user_question: str,
        user_language: str,
    ) -> str:
        """Answer a specific question about a known title."""
        title_context = _format_title_detail(title)
        system_prompt = (
            f"You are a knowledgeable Netflix assistant. "
            f"Respond in {user_language}. "
            f"Answer the user's question concisely using only the provided metadata. "
            f"Do not invent details not present in the metadata."
        )
        user_content = (
            f"Title metadata:\n{title_context}\n\n"
            f"User question: {user_question}"
        )
        return self._complete(system_prompt, user_content)

    def generate_catalog_miss_response(
        self,
        title_name: str,
        user_language: str,
    ) -> str:
        """Return a canned message when the title is not in the catalog."""
        return (
            f'"{title_name}" is not available in the current catalog. '
            f"Try asking about a different title or request new recommendations."
        )

    # ── 8.3: Clarification and no-results ────────────────────────────────────

    def generate_clarification(
        self,
        hint: str,
        session: Session,
        user_language: str,
    ) -> str:
        """Turn a clarification hint into a natural-language question."""
        system_prompt = (
            f"You are a helpful Netflix recommendation assistant. "
            f"Respond in {user_language}. "
            f"Ask the user a short, friendly clarifying question based on the hint provided."
        )
        user_content = f"Clarification needed: {hint}"
        return self._complete(system_prompt, user_content)

    def generate_no_results_response(
        self,
        result: NoResultsResult,
        user_language: str,
    ) -> str:
        """Explain why no results were found and suggest how to proceed."""
        system_prompt = (
            f"You are a helpful Netflix recommendation assistant. "
            f"Respond in {user_language}. "
            f"Explain sympathetically that no titles were found and relay the suggestion."
        )
        user_content = (
            f"No results reason: {result.reason}\n"
            f"Suggestion for the user: {result.suggestion}"
        )
        return self._complete(system_prompt, user_content)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _complete(self, system_prompt: str, user_content: str) -> str:
        request = LLMRequest(
            system_prompt=system_prompt,
            messages=[Message(role="user", content=user_content)],
            model="generation",
            max_tokens=1024,
            temperature=0.7,
        )
        return self._llm.complete(request).content


# ── Formatting helpers ────────────────────────────────────────────────────────


def _format_recommendations(recommendations: list[Recommendation]) -> str:
    if not recommendations:
        return "(no titles to present)"
    lines = []
    for i, rec in enumerate(recommendations, 1):
        t = rec.title
        genres = ", ".join(t.genres) if t.genres else "N/A"
        lines.append(
            f"{i}. {t.title} ({t.type}, {t.release_year}) — Genres: {genres}"
        )
    return "\n".join(lines)


def _format_title_detail(title: NetflixTitle) -> str:
    genres = ", ".join(title.genres) if title.genres else "N/A"
    cast = ", ".join(title.cast) if title.cast else "N/A"
    return (
        f"Title: {title.title}\n"
        f"Type: {title.type}\n"
        f"Year: {title.release_year}\n"
        f"Director: {title.director or 'N/A'}\n"
        f"Cast: {cast}\n"
        f"Country: {title.country or 'N/A'}\n"
        f"Rating: {title.rating or 'N/A'}\n"
        f"Genres: {genres}\n"
        f"Description: {title.description}"
    )
