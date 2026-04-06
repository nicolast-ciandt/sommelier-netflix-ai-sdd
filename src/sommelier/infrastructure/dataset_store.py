"""DatasetStore — Infrastructure adapter implementing DatasetPort.

Loads the Netflix CSV at startup, normalizes all fields into typed
domain objects, and exposes filtering and TF-IDF similarity operations.

Tasks covered:
  2.1 — load_and_index(), title_count(), get_by_id()   ← this file
  2.2 — filter()                                        ← added in task 2.2
  2.3 — tfidf_similarity()                              ← added in task 2.3
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sommelier.domain.models import (
    NETFLIX_RATINGS_ORDERED,
    DatasetLoadError,
    DurationInfo,
    NetflixTitle,
    ScoredTitle,
)
from sommelier.ports.interfaces import DatasetFilter

if TYPE_CHECKING:
    pass


def _nullable_str(val: object) -> str | None:
    """Return *val* as a stripped string, or ``None`` if it is blank/NaN."""
    if val is None:
        return None
    if isinstance(val, float):
        # pandas represents missing values as float NaN
        import math
        if math.isnan(val):
            return None
    s = str(val).strip()
    return s if s else None


def _split_comma(val: object) -> tuple[str, ...]:
    """Split a comma-separated string into a tuple of stripped non-empty items."""
    s = _nullable_str(val)
    if s is None:
        return ()
    return tuple(item.strip() for item in s.split(",") if item.strip())


def _parse_duration(val: object) -> DurationInfo | None:
    """Parse Netflix 'duration' strings such as '90 min' or '3 Seasons'."""
    s = _nullable_str(val)
    if s is None:
        return None
    if s.endswith(" min"):
        try:
            return DurationInfo(value=int(s[:-4].strip()), unit="min")
        except ValueError:
            return None
    if "Season" in s:
        try:
            numeric_part = s.split()[0]
            return DurationInfo(value=int(numeric_part), unit="Seasons")
        except (ValueError, IndexError):
            return None
    return None


def _normalize_row(row: pd.Series) -> NetflixTitle:
    """Convert a raw pandas row into a typed ``NetflixTitle`` value object."""
    release_year_raw = row.get("release_year")
    try:
        release_year = int(float(str(release_year_raw))) if release_year_raw is not None else 0
    except (ValueError, TypeError):
        release_year = 0

    return NetflixTitle(
        show_id=str(row["show_id"]).strip(),
        type=str(row["type"]).strip(),  # type: ignore[arg-type]
        title=str(row["title"]).strip(),
        director=_nullable_str(row.get("director")),
        cast=_split_comma(row.get("cast")),
        country=_nullable_str(row.get("country")),
        release_year=release_year,
        rating=_nullable_str(row.get("rating")),
        duration=_parse_duration(row.get("duration")),
        genres=_split_comma(row.get("listed_in")),
        description=_nullable_str(row.get("description")) or "",
    )


class DatasetStore:
    """In-memory Netflix catalog store.

    Call ``load_and_index(path)`` once at startup before any other method.
    All subsequent operations are purely in-memory.
    """

    def __init__(self) -> None:
        self._titles_by_id: dict[str, NetflixTitle] = {}
        self._titles_list: list[NetflixTitle] = []
        self._tfidf_vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None  # scipy sparse matrix

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load_and_index(self, path: str | Path) -> None:
        """Load and normalize the Netflix CSV at *path*.

        Raises ``DatasetLoadError`` if the file is missing, empty, or
        structurally malformed.  Logs a summary line to ``stderr`` on success.
        """
        path = Path(path)

        if not path.exists():
            raise DatasetLoadError(f"Dataset file not found: {path}")

        try:
            df = pd.read_csv(path, dtype=object, keep_default_na=True)
        except pd.errors.EmptyDataError as exc:
            raise DatasetLoadError(f"Dataset file is empty: {path}") from exc
        except Exception as exc:
            raise DatasetLoadError(
                f"Failed to read dataset at {path}: {exc}"
            ) from exc

        if df.empty:
            raise DatasetLoadError(f"Dataset file is empty: {path}")

        titles = [_normalize_row(row) for _, row in df.iterrows()]

        self._titles_by_id = {t.show_id: t for t in titles}
        self._titles_list = titles

        corpus = [f"{t.title} {t.description}" for t in titles]
        vectorizer = TfidfVectorizer(strip_accents="unicode", lowercase=True)
        self._tfidf_matrix = vectorizer.fit_transform(corpus)
        self._tfidf_vectorizer = vectorizer

        print(
            f"[DatasetStore] Loaded {len(titles)} titles from {path}",
            file=sys.stderr,
        )

    # ── DatasetPort — query interface ─────────────────────────────────────────

    def title_count(self) -> int:
        """Return the number of titles currently loaded."""
        return len(self._titles_list)

    def get_by_id(self, show_id: str) -> NetflixTitle | None:
        """Return the title matching *show_id*, or ``None`` if not found."""
        return self._titles_by_id.get(show_id)

    def filter(self, criteria: DatasetFilter) -> list[NetflixTitle]:
        """Return titles matching *criteria*.

        Each non-None field in *criteria* acts as an AND condition.
        Returns an empty list when no titles match; never raises.
        """
        ceiling_index: int | None = None
        if criteria.maturity_ceiling is not None:
            try:
                ceiling_index = NETFLIX_RATINGS_ORDERED.index(criteria.maturity_ceiling)
            except ValueError:
                return []

        results: list[NetflixTitle] = []
        for title in self._titles_list:
            if criteria.content_type is not None:
                if title.type != criteria.content_type:
                    continue

            if criteria.genres is not None:
                lowered_criteria = {g.lower() for g in criteria.genres}
                lowered_title = {g.lower() for g in title.genres}
                if not lowered_criteria.intersection(lowered_title):
                    continue

            if criteria.year_min is not None:
                if title.release_year < criteria.year_min:
                    continue

            if criteria.year_max is not None:
                if title.release_year > criteria.year_max:
                    continue

            if ceiling_index is not None:
                if title.rating is None:
                    continue
                try:
                    title_index = NETFLIX_RATINGS_ORDERED.index(title.rating)
                except ValueError:
                    continue
                if title_index > ceiling_index:
                    continue

            if criteria.country is not None:
                if title.country is None:
                    continue
                if criteria.country.lower() not in title.country.lower():
                    continue

            results.append(title)

        return results

    def tfidf_similarity(
        self, query: str, candidates: list[NetflixTitle]
    ) -> list[ScoredTitle]:
        """Score *candidates* by TF-IDF cosine similarity against *query*.

        Returns candidates sorted descending by similarity score.
        Empty or whitespace-only queries return all candidates with score 0.0.
        Empty candidate list returns [].
        """
        if not candidates:
            return []

        query = query.strip()
        if not query or self._tfidf_vectorizer is None or self._tfidf_matrix is None:
            return [ScoredTitle(title=t, similarity_score=0.0) for t in candidates]

        # Build index mapping show_id → row position in the full matrix
        id_to_row: dict[str, int] = {
            t.show_id: i for i, t in enumerate(self._titles_list)
        }

        # Rows in the full matrix corresponding to the candidate titles
        candidate_rows = [id_to_row[t.show_id] for t in candidates]
        candidate_matrix = self._tfidf_matrix[candidate_rows]

        query_vec = self._tfidf_vectorizer.transform([query])
        scores: np.ndarray = cosine_similarity(query_vec, candidate_matrix).flatten()

        scored = [
            ScoredTitle(title=t, similarity_score=float(scores[i]))
            for i, t in enumerate(candidates)
        ]
        scored.sort(key=lambda s: s.similarity_score, reverse=True)
        return scored
