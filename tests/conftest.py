"""Shared pytest fixtures for the Netflix Content Recommender test suite."""

from unittest.mock import MagicMock, patch

import pytest

from sommelier.infrastructure.dataset_store import DatasetStore


# ---------------------------------------------------------------------------
# Raw fixture rows (same 10-row inventory as before, now as dicts)
# ---------------------------------------------------------------------------

FIXTURE_ROWS = [
    {"show_id": "s1",  "type": "Movie",   "title": "Thriller Night",   "director": "Jane Doe",      "cast": "Actor A, Actor B",          "country": "United States",  "release_year": "2019", "rating": "PG-13", "duration": "90 min",     "listed_in": "Drama, Thriller",                 "description": "A gripping thriller."},
    {"show_id": "s2",  "type": "TV Show", "title": "Epic Series",      "director": "John Smith",    "cast": "Actor C, Actor D",          "country": "United Kingdom", "release_year": "2021", "rating": "TV-MA", "duration": "3 Seasons",  "listed_in": "Action, Adventure",               "description": "An epic adventure series."},
    {"show_id": "s3",  "type": "TV Show", "title": "One Season Show",  "director": "Solo Dir",      "cast": "Actor E",                   "country": "Canada",         "release_year": "2022", "rating": "TV-14", "duration": "1 Season",   "listed_in": "Drama",                           "description": "A limited series."},
    {"show_id": "s4",  "type": "Movie",   "title": "No Director Film", "director": "",              "cast": "Actor F",                   "country": "France",         "release_year": "2018", "rating": "R",     "duration": "105 min",    "listed_in": "Documentary",                     "description": "A documentary."},
    {"show_id": "s5",  "type": "Movie",   "title": "No Cast Film",     "director": "Some Director", "cast": "",                          "country": "Germany",        "release_year": "2020", "rating": "PG",    "duration": "80 min",     "listed_in": "Comedy",                          "description": "A comedy."},
    {"show_id": "s6",  "type": "Movie",   "title": "No Country Film",  "director": "Director G",    "cast": "Actor G",                   "country": "",               "release_year": "2019", "rating": "G",     "duration": "70 min",     "listed_in": "Family",                          "description": "A family film."},
    {"show_id": "s7",  "type": "Movie",   "title": "No Rating Film",   "director": "Director H",    "cast": "Actor H",                   "country": "Australia",      "release_year": "2017", "rating": "",      "duration": "95 min",     "listed_in": "Romance",                         "description": "A romantic film."},
    {"show_id": "s8",  "type": "Movie",   "title": "No Duration Film", "director": "Director I",    "cast": "Actor I",                   "country": "Japan",          "release_year": "2016", "rating": "NR",    "duration": "",           "listed_in": "Anime Features",                  "description": "An anime film."},
    {"show_id": "s9",  "type": "Movie",   "title": "Multi-genre Film", "director": "Director J",    "cast": "Actor J, Actor K, Actor L", "country": "Brazil",         "release_year": "2020", "rating": "TV-PG", "duration": "120 min",    "listed_in": "Comedy, Drama, Romance",          "description": "A multi-genre film."},
    {"show_id": "s10", "type": "TV Show", "title": "Long Series",      "director": "Director K",    "cast": "Actor M",                   "country": "South Korea",    "release_year": "2017", "rating": "TV-Y7", "duration": "10 Seasons", "listed_in": "Anime Series, Action & Adventure","description": "A long-running series."},
]


def _make_store(rows: list[dict]) -> DatasetStore:
    """Return a DatasetStore loaded with the given rows via a mocked psycopg2."""
    mock_rows = [dict(r) for r in rows]

    mock_cursor = MagicMock()
    mock_cursor.__enter__ = lambda s: s
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_cursor.fetchall.return_value = mock_rows

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch("psycopg2.connect", return_value=mock_conn):
        ds = DatasetStore()
        ds.load_and_index("postgresql://fake/db")
    return ds


@pytest.fixture()
def store() -> DatasetStore:
    """An empty (unloaded) DatasetStore instance."""
    return DatasetStore()


@pytest.fixture()
def loaded_store() -> DatasetStore:
    """A DatasetStore loaded with the 10-row sample fixture."""
    return _make_store(FIXTURE_ROWS)
