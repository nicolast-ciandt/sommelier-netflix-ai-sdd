"""Task 2.1 — Verify DatasetStore CSV loading and field normalization.

Fixtures (store, sample_csv, loaded_store) are provided by conftest.py.
"""

from pathlib import Path

import pytest

from sommelier.domain.models import DatasetLoadError, DurationInfo, NetflixTitle
from sommelier.infrastructure.dataset_store import DatasetStore


# ── Successful load ───────────────────────────────────────────────────────────


class TestSuccessfulLoad:
    def test_load_returns_correct_title_count(self, store, sample_csv):
        store.load_and_index(sample_csv)
        assert store.title_count() == 10

    def test_load_logs_count_and_path_to_stderr(self, store, sample_csv, capsys):
        store.load_and_index(sample_csv)
        captured = capsys.readouterr()
        assert str(sample_csv) in captured.err
        assert "10" in captured.err

    def test_loaded_titles_are_accessible_by_id(self, store, sample_csv):
        store.load_and_index(sample_csv)
        title = store.get_by_id("s1")
        assert title is not None
        assert title.show_id == "s1"
        assert title.title == "Thriller Night"

    def test_get_by_id_returns_none_for_unknown_id(self, store, sample_csv):
        store.load_and_index(sample_csv)
        assert store.get_by_id("s999") is None

    def test_titles_are_netflix_title_instances(self, store, sample_csv):
        store.load_and_index(sample_csv)
        title = store.get_by_id("s1")
        assert isinstance(title, NetflixTitle)


# ── Field normalization: type and basic strings ───────────────────────────────


class TestBasicFieldNormalization:
    def test_movie_type_preserved(self, store, sample_csv):
        store.load_and_index(sample_csv)
        assert store.get_by_id("s1").type == "Movie"

    def test_tv_show_type_preserved(self, store, sample_csv):
        store.load_and_index(sample_csv)
        assert store.get_by_id("s2").type == "TV Show"

    def test_release_year_is_integer(self, store, sample_csv):
        store.load_and_index(sample_csv)
        title = store.get_by_id("s1")
        assert isinstance(title.release_year, int)
        assert title.release_year == 2019

    def test_description_is_string(self, store, sample_csv):
        store.load_and_index(sample_csv)
        assert store.get_by_id("s1").description == "A gripping thriller."


# ── Genres normalization (listed_in → tuple[str, ...]) ───────────────────────


class TestGenresNormalization:
    def test_multi_genre_is_split_correctly(self, store, sample_csv):
        store.load_and_index(sample_csv)
        title = store.get_by_id("s1")
        assert "Drama" in title.genres
        assert "Thriller" in title.genres
        assert len(title.genres) == 2

    def test_single_genre_produces_one_element(self, store, sample_csv):
        store.load_and_index(sample_csv)
        title = store.get_by_id("s3")
        assert title.genres == ("Drama",)

    def test_genres_with_ampersand_preserved(self, store, sample_csv):
        store.load_and_index(sample_csv)
        title = store.get_by_id("s10")
        assert "Action & Adventure" in title.genres

    def test_genres_stripped_of_whitespace(self, store, sample_csv):
        store.load_and_index(sample_csv)
        for genre in store.get_by_id("s9").genres:
            assert genre == genre.strip()

    def test_three_genres_parsed_correctly(self, store, sample_csv):
        store.load_and_index(sample_csv)
        title = store.get_by_id("s9")
        assert len(title.genres) == 3
        assert "Comedy" in title.genres
        assert "Drama" in title.genres
        assert "Romance" in title.genres


# ── Cast normalization (cast → tuple[str, ...]) ──────────────────────────────


class TestCastNormalization:
    def test_multi_cast_is_split(self, store, sample_csv):
        store.load_and_index(sample_csv)
        title = store.get_by_id("s1")
        assert "Actor A" in title.cast
        assert "Actor B" in title.cast
        assert len(title.cast) == 2

    def test_three_cast_members_parsed(self, store, sample_csv):
        store.load_and_index(sample_csv)
        title = store.get_by_id("s9")
        assert len(title.cast) == 3

    def test_empty_cast_becomes_empty_tuple(self, store, sample_csv):
        store.load_and_index(sample_csv)
        title = store.get_by_id("s5")
        assert title.cast == ()

    def test_cast_members_stripped(self, store, sample_csv):
        store.load_and_index(sample_csv)
        for member in store.get_by_id("s9").cast:
            assert member == member.strip()


# ── Duration normalization ────────────────────────────────────────────────────


class TestDurationNormalization:
    def test_minutes_parsed_to_duration_info(self, store, sample_csv):
        store.load_and_index(sample_csv)
        title = store.get_by_id("s1")
        assert title.duration == DurationInfo(value=90, unit="min")

    def test_multiple_seasons_parsed(self, store, sample_csv):
        store.load_and_index(sample_csv)
        title = store.get_by_id("s2")
        assert title.duration == DurationInfo(value=3, unit="Seasons")

    def test_single_season_normalized_to_seasons_unit(self, store, sample_csv):
        store.load_and_index(sample_csv)
        title = store.get_by_id("s3")
        assert title.duration == DurationInfo(value=1, unit="Seasons")

    def test_ten_seasons_parsed(self, store, sample_csv):
        store.load_and_index(sample_csv)
        title = store.get_by_id("s10")
        assert title.duration == DurationInfo(value=10, unit="Seasons")

    def test_missing_duration_becomes_none(self, store, sample_csv):
        store.load_and_index(sample_csv)
        title = store.get_by_id("s8")
        assert title.duration is None


# ── Nullable field handling ───────────────────────────────────────────────────


class TestNullableFields:
    def test_empty_director_becomes_none(self, store, sample_csv):
        store.load_and_index(sample_csv)
        title = store.get_by_id("s4")
        assert title.director is None

    def test_empty_country_becomes_none(self, store, sample_csv):
        store.load_and_index(sample_csv)
        title = store.get_by_id("s6")
        assert title.country is None

    def test_empty_rating_becomes_none(self, store, sample_csv):
        store.load_and_index(sample_csv)
        title = store.get_by_id("s7")
        assert title.rating is None

    def test_present_director_is_not_none(self, store, sample_csv):
        store.load_and_index(sample_csv)
        assert store.get_by_id("s1").director == "Jane Doe"

    def test_present_country_is_not_none(self, store, sample_csv):
        store.load_and_index(sample_csv)
        assert store.get_by_id("s1").country == "United States"

    def test_nullable_fields_are_never_empty_strings(self, store, sample_csv):
        store.load_and_index(sample_csv)
        for show_id in ["s4", "s5", "s6", "s7", "s8"]:
            title = store.get_by_id(show_id)
            assert title.director != ""
            assert title.country != ""
            assert title.rating != ""


# ── Error handling ────────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_missing_file_raises_dataset_load_error(self, store, tmp_path):
        missing = tmp_path / "does_not_exist.csv"
        with pytest.raises(DatasetLoadError, match="not found"):
            store.load_and_index(missing)

    def test_empty_file_raises_dataset_load_error(self, store, tmp_path):
        empty = tmp_path / "empty.csv"
        empty.write_text("")
        with pytest.raises(DatasetLoadError, match="empty"):
            store.load_and_index(empty)

    def test_error_message_includes_file_path(self, store, tmp_path):
        missing = tmp_path / "missing.csv"
        with pytest.raises(DatasetLoadError) as exc_info:
            store.load_and_index(missing)
        assert "missing.csv" in str(exc_info.value)

    def test_header_only_file_raises_dataset_load_error(self, store, tmp_path):
        header_only = tmp_path / "header_only.csv"
        header_only.write_text(
            "show_id,type,title,director,cast,country,date_added,"
            "release_year,rating,duration,listed_in,description\n"
        )
        with pytest.raises(DatasetLoadError, match="empty"):
            store.load_and_index(header_only)
