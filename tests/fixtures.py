"""Shared fixture generators for integration and performance tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from sommelier.infrastructure.dataset_store import DatasetStore


_GENRES = [
    "Drama", "Comedy", "Action", "Thriller", "Romance", "Documentary",
    "Horror", "Sci-Fi & Fantasy", "Animation", "Crime",
]
_COUNTRIES = ["United States", "United Kingdom", "France", "Japan", "Brazil"]
_RATINGS = ["G", "PG", "PG-13", "TV-14", "TV-MA", "R"]
_TYPES = ["Movie", "TV Show"]


def make_fixture_rows(n: int = 50) -> list[dict]:
    """Return n Netflix-style row dicts."""
    rows = []
    for i in range(1, n + 1):
        genre = _GENRES[(i - 1) % len(_GENRES)]
        genre2 = _GENRES[i % len(_GENRES)]
        content_type = _TYPES[(i - 1) % 2]
        rating = _RATINGS[(i - 1) % len(_RATINGS)]
        country = _COUNTRIES[(i - 1) % len(_COUNTRIES)]
        year = 1990 + (i % 35)
        duration = f"{80 + i} min" if content_type == "Movie" else f"{1 + (i % 5)} Seasons"
        rows.append({
            "show_id": f"s{i}",
            "type": content_type,
            "title": f"Title {i}: A {genre} {content_type}",
            "director": f"Director {i}",
            "cast": f"Actor {i}A, Actor {i}B",
            "country": country,
            "release_year": str(year),
            "rating": rating,
            "duration": duration,
            "listed_in": f"{genre}, {genre2}",
            "description": f"A compelling {genre.lower()} story featuring adventure and intrigue. Film number {i}.",
        })
    return rows


def make_store(n: int = 50) -> DatasetStore:
    """Return a DatasetStore loaded with n generated rows via a mocked psycopg2."""
    rows = make_fixture_rows(n)

    mock_cursor = MagicMock()
    mock_cursor.__enter__ = lambda s: s
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_cursor.fetchall.return_value = rows

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch("psycopg2.connect", return_value=mock_conn):
        ds = DatasetStore()
        ds.load_and_index("postgresql://fake/db")
    return ds
