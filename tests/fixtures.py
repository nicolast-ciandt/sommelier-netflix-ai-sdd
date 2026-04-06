"""Shared fixture generators for integration tests."""

from __future__ import annotations

import csv
from pathlib import Path


_GENRES = [
    "Drama", "Comedy", "Action", "Thriller", "Romance", "Documentary",
    "Horror", "Sci-Fi & Fantasy", "Animation", "Crime",
]
_COUNTRIES = ["United States", "United Kingdom", "France", "Japan", "Brazil"]
_RATINGS = ["G", "PG", "PG-13", "TV-14", "TV-MA", "R"]
_TYPES = ["Movie", "TV Show"]

_HEADER = [
    "show_id", "type", "title", "director", "cast", "country",
    "date_added", "release_year", "rating", "duration", "listed_in",
    "description",
]


def write_fixture_csv(path: Path, n: int = 50) -> Path:
    """Write an n-row Netflix-style CSV to *path* and return it."""
    rows = []
    for i in range(1, n + 1):
        genre = _GENRES[(i - 1) % len(_GENRES)]
        genre2 = _GENRES[i % len(_GENRES)]
        content_type = _TYPES[(i - 1) % 2]
        rating = _RATINGS[(i - 1) % len(_RATINGS)]
        country = _COUNTRIES[(i - 1) % len(_COUNTRIES)]
        year = 1990 + (i % 35)  # spans 1990-2024
        duration = f"{80 + i} min" if content_type == "Movie" else f"{1 + (i % 5)} Seasons"
        rows.append([
            f"s{i}",
            content_type,
            f"Title {i}: A {genre} {content_type}",
            f"Director {i}",
            f"Actor {i}A, Actor {i}B",
            country,
            f"January {1 + (i % 28)} 2022",
            str(year),
            rating,
            duration,
            f"{genre}, {genre2}",
            f"A compelling {genre.lower()} story featuring adventure and intrigue. Film number {i}.",
        ])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_HEADER)
        writer.writerows(rows)
    return path
