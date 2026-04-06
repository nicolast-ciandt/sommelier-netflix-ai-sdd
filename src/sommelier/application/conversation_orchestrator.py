"""ConversationOrchestrator — Application entry-point coordinating all components.

Routes each user turn through the correct pipeline, guards against LLM
failures and timeouts, and keeps session state consistent.

Tasks covered:
  9.1 — start_session(), handle_turn(), intent detection
  9.2 — recommendation, clarification, feedback, and title-detail pipelines
  9.3 — LLMUnavailableError recovery, unexpected exception recovery
"""

from __future__ import annotations

import sys

from sommelier import debug
from sommelier.application.recommendation_engine import RecommendationEngine
from sommelier.application.response_generator import ResponseGenerator
from sommelier.application.session_manager import SessionManager
from sommelier.domain.models import (
    LLMUnavailableError,
    NoResultsResult,
    Recommendation,
    Session,
)
from sommelier.domain.preference_extractor import PreferenceExtractor
from sommelier.ports.interfaces import DatasetPort

_RETRY_MSG = (
    "I'm having trouble reaching my recommendations engine right now. "
    "Please try again in a moment."
)
_ERROR_MSG = (
    "Something went wrong on my end. "
    "Your session is still active — please try again."
)
_GREETING = (
    "Welcome to Sommelier! I'm your personal Netflix guide.\n"
    "Tell me what you're in the mood to watch and I'll find the perfect titles for you."
)

# Simple heuristics for intent detection
_FEEDBACK_KEYWORDS = (
    "not that", "don't want", "dislike", "hated", "not interested",
    "too", "instead", "rather", "exclude", "remove", "liked", "loved",
    "enjoyed", "more like", "similar to", "that was great",
)
_DETAIL_PREFIXES = ("tell me about", "what is", "describe", "who directed",
                    "who stars", "when was", "what year", "is it")


class ConversationOrchestrator:
    """Coordinates all application components for each conversation turn."""

    def __init__(
        self,
        session_manager: SessionManager,
        preference_extractor: PreferenceExtractor,
        recommendation_engine: RecommendationEngine,
        response_generator: ResponseGenerator,
        dataset: DatasetPort,
    ) -> None:
        self._sm = session_manager
        self._pe = preference_extractor
        self._re = recommendation_engine
        self._rg = response_generator
        self._dataset = dataset

    # ── 9.1: Session lifecycle ────────────────────────────────────────────────

    def start_session(self) -> tuple[Session, str]:
        """Create a fresh session and return (session, greeting_message)."""
        session = self._sm.create_session()
        return session, _GREETING

    # ── 9.1 / 9.2: Turn routing ───────────────────────────────────────────────

    def handle_turn(
        self,
        user_message: str,
        session: Session,
    ) -> tuple[Session, str]:
        """Process one user turn and return (updated_session, assistant_response).

        Never raises — on any error, returns a recovery message with the
        session state preserved from before the failing call.
        """
        # Always record the user message first
        session = self._sm.append_message(session, "user", user_message)

        try:
            session, response = self._route(user_message, session)
        except LLMUnavailableError as exc:
            debug.log_exception("orchestrator", exc)
            response = _RETRY_MSG
        except Exception as exc:
            print(f"[Orchestrator] Unexpected error: {exc}", file=sys.stderr)
            debug.log_exception("orchestrator", exc)
            response = _ERROR_MSG

        session = self._sm.append_message(session, "assistant", response)
        return session, response

    # ── Private routing ───────────────────────────────────────────────────────

    def _route(
        self, user_message: str, session: Session
    ) -> tuple[Session, str]:
        lower = user_message.lower()

        # Feedback / refinement turn
        if session.seen_title_ids and any(kw in lower for kw in _FEEDBACK_KEYWORDS):
            debug.log("route", "intent=feedback")
            return self._feedback_turn(user_message, session)

        # Title detail question
        if any(lower.startswith(prefix) for prefix in _DETAIL_PREFIXES):
            debug.log("route", "intent=title_detail")
            return session, self._title_detail_turn(user_message, session)

        # Default: preference extraction + recommendation
        debug.log("route", "intent=recommendation")
        return self._recommendation_turn(user_message, session)

    def _recommendation_turn(
        self, user_message: str, session: Session
    ) -> tuple[Session, str]:
        delta = self._pe.extract(user_message, session, mode="preference")
        session = self._sm.apply_delta(session, delta)

        if delta.needs_clarification:
            hint = delta.clarification_hint or "Could you tell me more about what you'd like to watch?"
            response = self._rg.generate_clarification(hint, session, user_language="English")
            return session, response

        output = self._re.recommend(session.preference_profile, session)

        if isinstance(output, NoResultsResult):
            response = self._rg.generate_no_results_response(output, user_language="English")
            return session, response

        recommendations: list[Recommendation] = output
        response = self._rg.generate_recommendations_response(
            recommendations, session, user_language="English"
        )
        shown_ids = [r.title.show_id for r in recommendations]
        session = self._sm.register_shown_titles(session, shown_ids)
        return session, response

    def _feedback_turn(
        self, user_message: str, session: Session
    ) -> tuple[Session, str]:
        delta = self._pe.extract(user_message, session, mode="feedback")
        session = self._sm.apply_delta(session, delta)
        if delta.excluded_title_ids:
            session = self._sm.apply_rejected_titles(
                session, list(delta.excluded_title_ids)
            )

        output = self._re.recommend(session.preference_profile, session)

        if isinstance(output, NoResultsResult):
            response = self._rg.generate_no_results_response(output, user_language="English")
            return session, response

        recommendations: list[Recommendation] = output
        response = self._rg.generate_recommendations_response(
            recommendations, session, user_language="English"
        )
        shown_ids = [r.title.show_id for r in recommendations]
        session = self._sm.register_shown_titles(session, shown_ids)
        return session, response

    def _title_detail_turn(
        self, user_message: str, session: Session
    ) -> str:
        # Attempt to find a show_id mentioned in the message
        for word in user_message.split():
            title = self._dataset.get_by_id(word.strip(",.?!"))
            if title:
                return self._rg.generate_title_detail_response(
                    title, user_message, user_language="English"
                )
        # Catalog miss — extract the likely title name (words after prefix)
        title_name = user_message
        return self._rg.generate_catalog_miss_response(title_name, user_language="English")
