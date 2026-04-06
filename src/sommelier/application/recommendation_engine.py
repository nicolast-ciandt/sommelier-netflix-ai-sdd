"""RecommendationEngine — Application service orchestrating the retrieval pipeline.

Calls CandidateRetriever, enforces the 3–10 result-count rule, wraps
ScoredTitle objects into Recommendation objects, and returns structured
NoResultsResult when candidates are exhausted or filters match nothing.

Tasks covered:
  7.1 — recommend(), result-count enforcement, Recommendation wrapping
  7.2 — NoResultsResult with reason discrimination and suggestion text
"""

from __future__ import annotations

from sommelier import debug
from sommelier.domain.models import (
    NoResultsResult,
    PreferenceProfile,
    Recommendation,
    RecommendationOutput,
    Session,
)
from sommelier.domain.candidate_retriever import CandidateRetriever

_MIN_RESULTS = 3
_MAX_RESULTS = 10

_SUGGESTION_NO_MATCH = (
    "Try relaxing the genre or year filter — "
    "no titles matched your current preferences."
)
_SUGGESTION_ALL_SEEN = (
    "You've seen all matching titles — "
    "try broadening your search or clearing your history."
)


class RecommendationEngine:
    """Orchestrates retrieval and enforces business rules on result count."""

    def __init__(self, retriever: CandidateRetriever) -> None:
        self._retriever = retriever

    def recommend(
        self,
        profile: PreferenceProfile,
        session: Session,
    ) -> RecommendationOutput:
        """Return 3–10 Recommendation objects, or a NoResultsResult.

        Distinguishes between:
          - "no_matching_titles": the dataset filter returned nothing
          - "all_seen": titles matched but all are in seen_title_ids
        """
        # Fetch more than _MAX_RESULTS so we have room after the count check
        candidates = self._retriever.retrieve(
            profile,
            session.seen_title_ids,
            max_candidates=_MAX_RESULTS,
        )

        debug.log("engine", f"candidates={len(candidates)} seen={len(session.seen_title_ids)} profile={profile}")

        if len(candidates) < _MIN_RESULTS:
            reason = _detect_reason(profile, session, self._retriever)
            suggestion = (
                _SUGGESTION_ALL_SEEN
                if reason == "all_seen"
                else _SUGGESTION_NO_MATCH
            )
            debug.log("engine", f"no_results reason={reason}")
            return NoResultsResult(reason=reason, suggestion=suggestion)

        return [
            Recommendation(
                title=scored.title,
                relevance_score=scored.similarity_score,
                rationale="",
            )
            for scored in candidates[:_MAX_RESULTS]
        ]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _detect_reason(
    profile: PreferenceProfile,
    session: Session,
    retriever: CandidateRetriever,
) -> str:
    """Determine whether empty results are due to filtering or all being seen."""
    # Re-run retrieval without exclusions to check if anything matches the filter
    unexcluded = retriever.retrieve(profile, frozenset(), max_candidates=_MIN_RESULTS)
    if not unexcluded:
        return "no_matching_titles"
    return "all_seen"
