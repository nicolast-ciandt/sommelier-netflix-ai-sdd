"""PreferenceExtractor — Domain service for structured preference extraction.

Calls the LLM (Haiku model) with a strict JSON-output system prompt and
parses the response into a PreferenceProfileDelta.  Never raises on failure.

Tasks covered:
  5.1 — extract(), system prompt, JSON parsing, fallback
  5.2 — ambiguity/conflict detection, maturity_ceiling validation
  5.3 — mode="feedback" variant
"""

from __future__ import annotations

import json
from typing import Literal

from sommelier.domain.models import (
    NETFLIX_RATINGS_ORDERED,
    LLMUnavailableError,
    Message,
    PreferenceProfileDelta,
    Session,
)
from sommelier.ports.interfaces import LLMRequest, LLMPort

_RATINGS_SET: frozenset[str] = frozenset(NETFLIX_RATINGS_ORDERED)

# ── System prompts ────────────────────────────────────────────────────────────

_SCHEMA = """\
{
  "genres": ["string"],
  "mood_keywords": ["string"],
  "content_type": "Movie" | "TV Show" | null,
  "year_min": integer | null,
  "year_max": integer | null,
  "maturity_ceiling": "G"|"TV-Y"|"TV-Y7"|"TV-G"|"PG"|"TV-PG"|"PG-13"|"TV-14"|"R"|"TV-MA"|"NC-17"|"NR" | null,
  "country_filter": "string" | null,
  "excluded_title_ids": ["string"],
  "positive_genre_signals": ["string"],
  "needs_clarification": boolean,
  "clarification_hint": "string" | null,
  "has_conflict": boolean,
  "conflict_description": "string" | null
}"""

_PREFERENCE_SYSTEM_PROMPT = f"""\
You are a preference extraction engine for a Netflix recommendation assistant.

Given a user message, extract their viewing preferences and return ONLY a \
JSON object matching this schema:

{_SCHEMA}

Rules:
- genres: list of Netflix genre names mentioned or implied (e.g. "Drama", "Comedy")
- mood_keywords: descriptive mood/tone words (e.g. "dark", "uplifting", "suspenseful")
- content_type: "Movie" or "TV Show" if specified, otherwise null
- year_min / year_max: release year bounds if mentioned (e.g. "90s" → year_min=1990, year_max=1999)
- maturity_ceiling: the most permissive rating acceptable; must be one of the enum values or null
- country_filter: country of origin if mentioned, otherwise null
- excluded_title_ids: always empty for preference extraction (use feedback mode for rejections)
- positive_genre_signals: always empty for preference extraction
- needs_clarification: true if the message is too vague to extract any signals
- clarification_hint: a short question to ask if needs_clarification is true, otherwise null
- has_conflict: true if the message contains contradictory signals
- conflict_description: brief description of the conflict if has_conflict is true, otherwise null

Example input: "I want a dark thriller from the 90s, nothing too violent"
Example output:
{{
  "genres": ["Thriller"],
  "mood_keywords": ["dark"],
  "content_type": null,
  "year_min": 1990,
  "year_max": 1999,
  "maturity_ceiling": "R",
  "country_filter": null,
  "excluded_title_ids": [],
  "positive_genre_signals": [],
  "needs_clarification": false,
  "clarification_hint": null,
  "has_conflict": false,
  "conflict_description": null
}}

Return ONLY the JSON object, no markdown, no explanation."""

_FEEDBACK_SYSTEM_PROMPT = f"""\
You are a feedback extraction engine for a Netflix recommendation assistant.

Given a user feedback message, extract rejection signals and positive \
reinforcement signals, and return ONLY a JSON object matching this schema:

{_SCHEMA}

Rules:
- excluded_title_ids: show_ids explicitly rejected by the user (e.g. "not that one" → extract the ID if mentioned)
- positive_genre_signals: genres the user positively responded to (e.g. "I liked that action one")
- genres / mood_keywords: any new preference constraints mentioned alongside feedback
- content_type / year_min / year_max / maturity_ceiling / country_filter: update if the user adds a new constraint
- needs_clarification: true if the feedback is too vague to parse
- has_conflict: true if the feedback contradicts prior signals

Return ONLY the JSON object, no markdown, no explanation."""


class PreferenceExtractor:
    """Domain service: parse user messages into PreferenceProfileDelta via LLM."""

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    def extract(
        self,
        user_message: str,
        session: Session,
        mode: Literal["preference", "feedback"] = "preference",
    ) -> PreferenceProfileDelta:
        """Extract a PreferenceProfileDelta from *user_message*.

        Never raises — returns a fallback delta with needs_clarification=True
        on any LLM failure or parse error.
        """
        system_prompt = (
            _PREFERENCE_SYSTEM_PROMPT if mode == "preference" else _FEEDBACK_SYSTEM_PROMPT
        )

        messages = [Message(role="user", content=user_message)]
        request = LLMRequest(
            system_prompt=system_prompt,
            messages=messages,
            model="extraction",
            max_tokens=512,
            temperature=0.1,
        )

        try:
            response = self._llm.complete(request)
            raw = response.content.strip()
        except LLMUnavailableError:
            return _fallback_delta(user_message)

        return _parse_delta(raw, user_message)


# ── Parsing helpers ───────────────────────────────────────────────────────────


def _parse_delta(raw: str, user_message: str) -> PreferenceProfileDelta:
    """Parse raw JSON string into PreferenceProfileDelta; fallback on error."""
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return _fallback_delta(user_message)
        return _build_delta(data)
    except (json.JSONDecodeError, Exception):
        return _fallback_delta(user_message)


def _build_delta(data: dict) -> PreferenceProfileDelta:
    """Build a validated PreferenceProfileDelta from a parsed dict."""
    maturity = data.get("maturity_ceiling")
    if maturity is not None and maturity not in _RATINGS_SET:
        maturity = None

    return PreferenceProfileDelta(
        genres=tuple(str(g) for g in data.get("genres") or []),
        mood_keywords=tuple(str(k) for k in data.get("mood_keywords") or []),
        content_type=data.get("content_type"),
        year_min=_int_or_none(data.get("year_min")),
        year_max=_int_or_none(data.get("year_max")),
        maturity_ceiling=maturity,
        country_filter=data.get("country_filter"),
        excluded_title_ids=tuple(str(i) for i in data.get("excluded_title_ids") or []),
        positive_genre_signals=tuple(str(g) for g in data.get("positive_genre_signals") or []),
        needs_clarification=bool(data.get("needs_clarification", False)),
        clarification_hint=data.get("clarification_hint"),
        has_conflict=bool(data.get("has_conflict", False)),
        conflict_description=data.get("conflict_description"),
    )


def _fallback_delta(user_message: str) -> PreferenceProfileDelta:
    return PreferenceProfileDelta(
        mood_keywords=(user_message,) if user_message.strip() else (),
        needs_clarification=True,
    )


def _int_or_none(val: object) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None
