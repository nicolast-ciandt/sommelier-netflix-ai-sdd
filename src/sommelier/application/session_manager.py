"""SessionManager — Application service owning all mutable session state.

Sole writer of Session objects. Every method returns a new Session
(immutable update pattern) and never mutates its input.

Tasks covered:
  4.1 — create_session(), append_message(), history truncation
  4.2 — apply_delta() (additive preference profile merging)
  4.3 — register_shown_titles(), apply_rejected_titles(), lock_maturity_ceiling()
"""

from __future__ import annotations

import dataclasses
import os
import uuid

from sommelier.domain.models import (
    Message,
    PreferenceProfile,
    PreferenceProfileDelta,
    Session,
)

_DEFAULT_MAX_HISTORY_TURNS = 20


class SessionManager:
    """Owns and mutates session state via immutable update pattern."""

    def __init__(self, max_history_turns: int | None = None) -> None:
        if max_history_turns is not None:
            self._max_turns = max_history_turns
        else:
            self._max_turns = int(
                os.environ.get("MAX_HISTORY_TURNS", _DEFAULT_MAX_HISTORY_TURNS)
            )

    # ── 4.1: Session lifecycle ────────────────────────────────────────────────

    def create_session(self) -> Session:
        """Return a new blank session with a unique UUID."""
        return Session(
            id=str(uuid.uuid4()),
            conversation_history=[],
            preference_profile=PreferenceProfile(),
            seen_title_ids=frozenset(),
            maturity_ceiling_locked=False,
        )

    def append_message(
        self, session: Session, role: str, content: str
    ) -> Session:
        """Return a new Session with *message* appended, truncated to max_turns."""
        new_history = list(session.conversation_history) + [
            Message(role=role, content=content)  # type: ignore[arg-type]
        ]
        if len(new_history) > self._max_turns:
            new_history = new_history[-self._max_turns :]
        return dataclasses.replace(session, conversation_history=new_history)

    # ── 4.2: Preference profile accumulation ─────────────────────────────────

    def apply_delta(self, session: Session, delta: PreferenceProfileDelta) -> Session:
        """Merge *delta* into the session's PreferenceProfile additively.

        List fields are appended (deduped); scalar fields updated only when
        the delta provides a non-None value.
        """
        p = session.preference_profile

        genres = _merge_list(p.genres, delta.genres)
        mood_keywords = _merge_list(p.mood_keywords, delta.mood_keywords)
        positive_genre_signals = _merge_list(
            p.positive_genre_signals, delta.positive_genre_signals
        )

        # Apply maturity ceiling only if not locked; once locked, ceiling can't rise
        if delta.maturity_ceiling is not None and not session.maturity_ceiling_locked:
            new_ceiling = delta.maturity_ceiling
            ceiling_locked = True
        else:
            new_ceiling = p.maturity_ceiling
            ceiling_locked = session.maturity_ceiling_locked

        new_profile = dataclasses.replace(
            p,
            genres=genres,
            mood_keywords=mood_keywords,
            positive_genre_signals=positive_genre_signals,
            content_type=delta.content_type if delta.content_type is not None else p.content_type,
            year_min=delta.year_min if delta.year_min is not None else p.year_min,
            year_max=delta.year_max if delta.year_max is not None else p.year_max,
            country_filter=delta.country_filter if delta.country_filter is not None else p.country_filter,
            maturity_ceiling=new_ceiling,
        )
        return dataclasses.replace(
            session, preference_profile=new_profile, maturity_ceiling_locked=ceiling_locked
        )

    # ── 4.3: Seen-title registry and maturity ceiling ─────────────────────────

    def register_shown_titles(
        self, session: Session, title_ids: list[str]
    ) -> Session:
        """Return a new Session with *title_ids* added to seen_title_ids."""
        return dataclasses.replace(
            session,
            seen_title_ids=session.seen_title_ids | frozenset(title_ids),
        )

    def apply_rejected_titles(
        self, session: Session, rejected_ids: list[str]
    ) -> Session:
        """Add rejected title IDs to seen_title_ids so they won't reappear."""
        return self.register_shown_titles(session, rejected_ids)

    def lock_maturity_ceiling(self, session: Session, ceiling: str) -> Session:
        """Set the maturity ceiling and lock it; subsequent calls are ignored."""
        if session.maturity_ceiling_locked:
            return session
        new_profile = dataclasses.replace(
            session.preference_profile, maturity_ceiling=ceiling
        )
        return dataclasses.replace(
            session,
            preference_profile=new_profile,
            maturity_ceiling_locked=True,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _merge_list(existing: list[str], additions: tuple[str, ...]) -> list[str]:
    """Append *additions* to *existing*, skipping duplicates."""
    result = list(existing)
    seen = set(existing)
    for item in additions:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result
