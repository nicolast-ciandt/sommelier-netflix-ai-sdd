"""CandidateRetriever — Domain service for filter-and-rank candidate retrieval.

Translates a PreferenceProfile into a DatasetFilter, applies it, scores
the filtered pool via TF-IDF cosine similarity, applies session exclusions,
and returns a capped ranked list of ScoredTitle objects.

Tasks covered:
  6.1 — retrieve(): filter translation, TF-IDF ranking, no-keyword fallback
  6.2 — session-scoped exclusions applied after scoring
"""

from __future__ import annotations

import random

from sommelier.domain.models import PreferenceProfile, ScoredTitle
from sommelier.ports.interfaces import DatasetFilter, DatasetPort


class CandidateRetriever:
    """Domain service: produce a ranked candidate list from a PreferenceProfile."""

    def __init__(self, dataset: DatasetPort) -> None:
        self._dataset = dataset

    def retrieve(
        self,
        profile: PreferenceProfile,
        excluded_ids: frozenset[str],
        max_candidates: int = 20,
    ) -> list[ScoredTitle]:
        """Return up to *max_candidates* ScoredTitle objects ranked by relevance.

        Steps:
          1. Translate profile → DatasetFilter and call dataset.filter()
          2. Score filtered pool with tfidf_similarity() using genres+keywords query
          3. Remove any title whose show_id is in excluded_ids
          4. Return top-max_candidates results
        """
        # ── Step 1: filter ────────────────────────────────────────────────────
        criteria = DatasetFilter(
            content_type=profile.content_type,  # type: ignore[arg-type]
            genres=profile.genres if profile.genres else None,
            year_min=profile.year_min,
            year_max=profile.year_max,
            maturity_ceiling=profile.maturity_ceiling,
            country=profile.country_filter,
        )
        candidates = self._dataset.filter(criteria)

        if not candidates:
            return []

        # ── Step 2: score ─────────────────────────────────────────────────────
        query_parts = list(profile.genres) + list(profile.mood_keywords)
        query = " ".join(query_parts).strip()

        if query:
            scored = self._dataset.tfidf_similarity(query, candidates)
        else:
            scored = [ScoredTitle(title=t, similarity_score=0.0) for t in candidates]

        # ── Step 3: exclude seen titles ───────────────────────────────────────
        scored = [s for s in scored if s.title.show_id not in excluded_ids]

        if not query:
            # Random shuffle for zero-score results to avoid position bias
            random.shuffle(scored)

        # ── Step 4: cap ───────────────────────────────────────────────────────
        return scored[:max_candidates]
