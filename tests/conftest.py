"""Shared pytest fixtures for the Netflix Content Recommender test suite."""

import csv
from pathlib import Path

import pytest

from sommelier.infrastructure.dataset_store import DatasetStore


@pytest.fixture()
def store() -> DatasetStore:
    return DatasetStore()


@pytest.fixture()
def sample_csv(tmp_path: Path) -> Path:
    """10-row fixture covering movies, TV shows, and various null/edge cases.

    Row inventory:
      s1  Movie    PG-13  2019  United States   Drama, Thriller
      s2  TV Show  TV-MA  2021  United Kingdom  Action, Adventure
      s3  TV Show  TV-14  2022  Canada          Drama
      s4  Movie    R      2018  France          Documentary
      s5  Movie    PG     2020  Germany         Comedy
      s6  Movie    G      2019  None            Family
      s7  Movie    None   2017  Australia       Romance
      s8  Movie    NR     2016  Japan           Anime Features
      s9  Movie    TV-PG  2020  Brazil          Comedy, Drama, Romance
      s10 TV Show  TV-Y7  2017  South Korea     Anime Series, Action & Adventure
    """
    rows = [
        # fmt: off
        ["s1",  "Movie",   "Thriller Night",   "Jane Doe",       "Actor A, Actor B",          "United States", "January 1 2020",   "2019", "PG-13", "90 min",    "Drama, Thriller",                "A gripping thriller."],
        ["s2",  "TV Show", "Epic Series",      "John Smith",     "Actor C, Actor D",          "United Kingdom","February 5 2021",  "2021", "TV-MA", "3 Seasons", "Action, Adventure",              "An epic adventure series."],
        ["s3",  "TV Show", "One Season Show",  "Solo Dir",       "Actor E",                   "Canada",        "March 10 2022",    "2022", "TV-14", "1 Season",  "Drama",                          "A limited series."],
        ["s4",  "Movie",   "No Director Film", "",               "Actor F",                   "France",        "April 15 2019",    "2018", "R",     "105 min",   "Documentary",                    "A documentary."],
        ["s5",  "Movie",   "No Cast Film",     "Some Director",  "",                          "Germany",       "May 20 2020",      "2020", "PG",    "80 min",    "Comedy",                         "A comedy."],
        ["s6",  "Movie",   "No Country Film",  "Director G",     "Actor G",                   "",              "June 1 2021",      "2019", "G",     "70 min",    "Family",                         "A family film."],
        ["s7",  "Movie",   "No Rating Film",   "Director H",     "Actor H",                   "Australia",     "July 7 2018",      "2017", "",      "95 min",    "Romance",                        "A romantic film."],
        ["s8",  "Movie",   "No Duration Film", "Director I",     "Actor I",                   "Japan",         "August 8 2019",    "2016", "NR",    "",          "Anime Features",                 "An anime film."],
        ["s9",  "Movie",   "Multi-genre Film", "Director J",     "Actor J, Actor K, Actor L", "Brazil",        "September 9 2020", "2020", "TV-PG", "120 min",   "Comedy, Drama, Romance",         "A multi-genre film."],
        ["s10", "TV Show", "Long Series",      "Director K",     "Actor M",                   "South Korea",   "October 10 2017",  "2017", "TV-Y7", "10 Seasons","Anime Series, Action & Adventure","A long-running series."],
        # fmt: on
    ]
    path = tmp_path / "netflix.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "show_id", "type", "title", "director", "cast", "country",
            "date_added", "release_year", "rating", "duration", "listed_in",
            "description",
        ])
        writer.writerows(rows)
    return path


@pytest.fixture()
def loaded_store(store: DatasetStore, sample_csv: Path) -> DatasetStore:
    """A DatasetStore with the 10-row sample fixture already loaded."""
    store.load_and_index(sample_csv)
    return store
