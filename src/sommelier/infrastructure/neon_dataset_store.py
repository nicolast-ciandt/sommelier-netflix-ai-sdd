"""NeonDatasetStore — Infrastructure adapter implementing DatasetPort via Neon PostgreSQL.

Loads the Netflix catalog from the ``netflix_shows`` table in a Neon DB
instance at startup, normalizes rows into typed domain objects, and exposes
the same filtering and TF-IDF similarity operations as ``DatasetStore``.

The table is expected to have the standard Kaggle Netflix dataset columns:
    show_id, type, title, director, cast, country, date_added,
    release_year, rating, duration, listed_in, description
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
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
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


def _split_comma(val: object) -> tuple[str, ...]:
    s = _nullable_str(val)
    if s is None:
        return ()
    return tuple(item.strip() for item in s.split(",") if item.strip())


def _parse_duration(val: object) -> DurationInfo | None:
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
            return DurationInfo(value=int(s.split()[0]), unit="Seasons")
        except (ValueError, IndexError):
            return None
    return None


def _normalize_row(row: dict) -> NetflixTitle:
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


class NeonDatasetStore:
    """Netflix catalog store backed by a Neon PostgreSQL database.

    Call ``load_and_index(connection_string)`` once at startup before any
    other method.  All subsequent operations are purely in-memory.
    """

    def __init__(self) -> None:
        self._titles_by_id: dict[str, NetflixTitle] = {}
        self._titles_list: list[NetflixTitle] = []
        self._tfidf_vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None  # scipy sparse matrix

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load_and_index(self, connection_string: str) -> None:
        """Load and normalize the ``netflix_shows`` table from Neon.

        Raises ``DatasetLoadError`` on connection failure or empty result.
        Logs a summary line to ``stderr`` on success.
        """
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError as exc:
            raise DatasetLoadError(
                "psycopg2-binary is required for Neon DB support. "
                "Run: pip install psycopg2-binary"
            ) from exc

        try:
            conn = psycopg2.connect(connection_string)
        except Exception as exc:
            raise DatasetLoadError(f"Failed to connect to Neon DB: {exc}") from exc

        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT show_id, type, title, director, cast, country, "
                    "release_year, rating, duration, listed_in, description "
                    "FROM netflix_shows"
                )
                rows = cur.fetchall()
        except Exception as exc:
            conn.close()
            raise DatasetLoadError(
                f"Failed to query netflix_shows table: {exc}"
            ) from exc
        finally:
            conn.close()

        if not rows:
            raise DatasetLoadError("netflix_shows table is empty or returned no rows")

        titles = [_normalize_row(dict(row)) for row in rows]

        self._titles_by_id = {t.show_id: t for t in titles}
        self._titles_list = titles

        corpus = [f"{t.title} {t.description}" for t in titles]
        vectorizer = TfidfVectorizer(strip_accents="unicode", lowercase=True)
        self._tfidf_matrix = vectorizer.fit_transform(corpus)
        self._tfidf_vectorizer = vectorizer

        print(
            f"[NeonDatasetStore] Loaded {len(titles)} titles from Neon DB",
            file=sys.stderr,
        )

    # ── DatasetPort — query interface ─────────────────────────────────────────

    def title_count(self) -> int:
        return len(self._titles_list)

    def get_by_id(self, show_id: str) -> NetflixTitle | None:
        return self._titles_by_id.get(show_id)

    def filter(self, criteria: DatasetFilter) -> list[NetflixTitle]:
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
        if not candidates:
            return []

        query = query.strip()
        if not query or self._tfidf_vectorizer is None or self._tfidf_matrix is None:
            return [ScoredTitle(title=t, similarity_score=0.0) for t in candidates]

        id_to_row: dict[str, int] = {
            t.show_id: i for i, t in enumerate(self._titles_list)
        }

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
