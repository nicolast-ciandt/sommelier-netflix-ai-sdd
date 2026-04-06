"""Verify DatasetStore Neon DB loading and field normalization."""

from unittest.mock import MagicMock, patch

import pytest

from sommelier.domain.models import DatasetLoadError, DurationInfo, NetflixTitle
from sommelier.infrastructure.dataset_store import DatasetStore
from tests.conftest import FIXTURE_ROWS, _make_store


# ── Successful load ───────────────────────────────────────────────────────────


class TestSuccessfulLoad:
    def test_load_returns_correct_title_count(self):
        ds = _make_store(FIXTURE_ROWS)
        assert ds.title_count() == 10

    def test_load_logs_count_to_stderr(self, capsys):
        _make_store(FIXTURE_ROWS)
        captured = capsys.readouterr()
        assert "10" in captured.err

    def test_loaded_titles_are_accessible_by_id(self):
        ds = _make_store(FIXTURE_ROWS)
        title = ds.get_by_id("s1")
        assert title is not None
        assert title.show_id == "s1"
        assert title.title == "Thriller Night"

    def test_get_by_id_returns_none_for_unknown_id(self):
        ds = _make_store(FIXTURE_ROWS)
        assert ds.get_by_id("s999") is None

    def test_titles_are_netflix_title_instances(self):
        ds = _make_store(FIXTURE_ROWS)
        assert isinstance(ds.get_by_id("s1"), NetflixTitle)


# ── Field normalization: type and basic strings ───────────────────────────────


class TestBasicFieldNormalization:
    def test_movie_type_preserved(self):
        ds = _make_store(FIXTURE_ROWS)
        assert ds.get_by_id("s1").type == "Movie"

    def test_tv_show_type_preserved(self):
        ds = _make_store(FIXTURE_ROWS)
        assert ds.get_by_id("s2").type == "TV Show"

    def test_release_year_is_integer(self):
        ds = _make_store(FIXTURE_ROWS)
        title = ds.get_by_id("s1")
        assert isinstance(title.release_year, int)
        assert title.release_year == 2019

    def test_description_is_string(self):
        ds = _make_store(FIXTURE_ROWS)
        assert ds.get_by_id("s1").description == "A gripping thriller."


# ── Genres normalization ──────────────────────────────────────────────────────


class TestGenresNormalization:
    def test_multi_genre_is_split_correctly(self):
        ds = _make_store(FIXTURE_ROWS)
        title = ds.get_by_id("s1")
        assert "Drama" in title.genres
        assert "Thriller" in title.genres
        assert len(title.genres) == 2

    def test_single_genre_produces_one_element(self):
        ds = _make_store(FIXTURE_ROWS)
        assert ds.get_by_id("s3").genres == ("Drama",)

    def test_genres_with_ampersand_preserved(self):
        ds = _make_store(FIXTURE_ROWS)
        assert "Action & Adventure" in ds.get_by_id("s10").genres

    def test_genres_stripped_of_whitespace(self):
        ds = _make_store(FIXTURE_ROWS)
        for genre in ds.get_by_id("s9").genres:
            assert genre == genre.strip()

    def test_three_genres_parsed_correctly(self):
        ds = _make_store(FIXTURE_ROWS)
        title = ds.get_by_id("s9")
        assert len(title.genres) == 3
        assert "Comedy" in title.genres
        assert "Drama" in title.genres
        assert "Romance" in title.genres


# ── Cast normalization ────────────────────────────────────────────────────────


class TestCastNormalization:
    def test_multi_cast_is_split(self):
        ds = _make_store(FIXTURE_ROWS)
        title = ds.get_by_id("s1")
        assert "Actor A" in title.cast
        assert "Actor B" in title.cast
        assert len(title.cast) == 2

    def test_three_cast_members_parsed(self):
        ds = _make_store(FIXTURE_ROWS)
        assert len(ds.get_by_id("s9").cast) == 3

    def test_empty_cast_becomes_empty_tuple(self):
        ds = _make_store(FIXTURE_ROWS)
        assert ds.get_by_id("s5").cast == ()

    def test_cast_members_stripped(self):
        ds = _make_store(FIXTURE_ROWS)
        for member in ds.get_by_id("s9").cast:
            assert member == member.strip()


# ── Duration normalization ────────────────────────────────────────────────────


class TestDurationNormalization:
    def test_minutes_parsed_to_duration_info(self):
        ds = _make_store(FIXTURE_ROWS)
        assert ds.get_by_id("s1").duration == DurationInfo(value=90, unit="min")

    def test_multiple_seasons_parsed(self):
        ds = _make_store(FIXTURE_ROWS)
        assert ds.get_by_id("s2").duration == DurationInfo(value=3, unit="Seasons")

    def test_single_season_normalized(self):
        ds = _make_store(FIXTURE_ROWS)
        assert ds.get_by_id("s3").duration == DurationInfo(value=1, unit="Seasons")

    def test_ten_seasons_parsed(self):
        ds = _make_store(FIXTURE_ROWS)
        assert ds.get_by_id("s10").duration == DurationInfo(value=10, unit="Seasons")

    def test_missing_duration_becomes_none(self):
        ds = _make_store(FIXTURE_ROWS)
        assert ds.get_by_id("s8").duration is None


# ── Nullable field handling ───────────────────────────────────────────────────


class TestNullableFields:
    def test_empty_director_becomes_none(self):
        ds = _make_store(FIXTURE_ROWS)
        assert ds.get_by_id("s4").director is None

    def test_empty_country_becomes_none(self):
        ds = _make_store(FIXTURE_ROWS)
        assert ds.get_by_id("s6").country is None

    def test_empty_rating_becomes_none(self):
        ds = _make_store(FIXTURE_ROWS)
        assert ds.get_by_id("s7").rating is None

    def test_present_director_is_not_none(self):
        ds = _make_store(FIXTURE_ROWS)
        assert ds.get_by_id("s1").director == "Jane Doe"

    def test_present_country_is_not_none(self):
        ds = _make_store(FIXTURE_ROWS)
        assert ds.get_by_id("s1").country == "United States"

    def test_nullable_fields_are_never_empty_strings(self):
        ds = _make_store(FIXTURE_ROWS)
        for show_id in ["s4", "s5", "s6", "s7", "s8"]:
            title = ds.get_by_id(show_id)
            assert title.director != ""
            assert title.country != ""
            assert title.rating != ""


# ── Error handling ────────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_connection_failure_raises_dataset_load_error(self):
        with patch("psycopg2.connect", side_effect=Exception("connection refused")):
            with pytest.raises(DatasetLoadError, match="Failed to connect"):
                DatasetStore().load_and_index("postgresql://bad/db")

    def test_empty_table_raises_dataset_load_error(self):
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: s
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch("psycopg2.connect", return_value=mock_conn):
            with pytest.raises(DatasetLoadError, match="empty"):
                DatasetStore().load_and_index("postgresql://fake/db")

    def test_query_failure_raises_dataset_load_error(self):
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: s
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.execute.side_effect = Exception("table does not exist")
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch("psycopg2.connect", return_value=mock_conn):
            with pytest.raises(DatasetLoadError, match="Failed to query"):
                DatasetStore().load_and_index("postgresql://fake/db")
