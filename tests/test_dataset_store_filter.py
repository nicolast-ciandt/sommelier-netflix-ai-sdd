"""Verify DatasetStore.filter() structured catalog filtering.

10-row fixture inventory (from conftest.py):
  s1  Movie    PG-13  2019  United States   Drama, Thriller
  s2  TV Show  TV-MA  2021  United Kingdom  Action, Adventure
  s3  TV Show  TV-14  2022  Canada          Drama
  s4  Movie    R      2018  France          Documentary
  s5  Movie    PG     2020  Germany         Comedy
  s6  Movie    G      2019  None(country)   Family
  s7  Movie    None   2017  Australia       Romance
  s8  Movie    NR     2016  Japan           Anime Features
  s9  Movie    TV-PG  2020  Brazil          Comedy, Drama, Romance
  s10 TV Show  TV-Y7  2017  South Korea     Anime Series, Action & Adventure
"""

import pytest

from sommelier.infrastructure.dataset_store import DatasetStore
from sommelier.ports.interfaces import DatasetFilter


# ── No-op filter (empty criteria) ────────────────────────────────────────────


class TestNoOpFilter:
    def test_empty_criteria_returns_all_titles(self, loaded_store):
        assert len(loaded_store.filter(DatasetFilter())) == 10

    def test_empty_criteria_returns_list(self, loaded_store):
        assert isinstance(loaded_store.filter(DatasetFilter()), list)


# ── content_type filter ───────────────────────────────────────────────────────


class TestContentTypeFilter:
    def test_movie_filter_returns_seven_movies(self, loaded_store):
        assert len(loaded_store.filter(DatasetFilter(content_type="Movie"))) == 7

    def test_tv_show_filter_returns_three_shows(self, loaded_store):
        assert len(loaded_store.filter(DatasetFilter(content_type="TV Show"))) == 3

    def test_movie_filter_excludes_tv_shows(self, loaded_store):
        ids = {t.show_id for t in loaded_store.filter(DatasetFilter(content_type="Movie"))}
        assert "s2" not in ids
        assert "s3" not in ids
        assert "s10" not in ids

    def test_tv_show_filter_includes_correct_ids(self, loaded_store):
        ids = {t.show_id for t in loaded_store.filter(DatasetFilter(content_type="TV Show"))}
        assert ids == {"s2", "s3", "s10"}


# ── genres filter (any-of, case-insensitive) ──────────────────────────────────


class TestGenresFilter:
    def test_drama_genre_matches_three_titles(self, loaded_store):
        assert len(loaded_store.filter(DatasetFilter(genres=["Drama"]))) == 3

    def test_drama_genre_returns_correct_ids(self, loaded_store):
        ids = {t.show_id for t in loaded_store.filter(DatasetFilter(genres=["Drama"]))}
        assert ids == {"s1", "s3", "s9"}

    def test_genre_match_is_case_insensitive(self, loaded_store):
        lower = loaded_store.filter(DatasetFilter(genres=["drama"]))
        upper = loaded_store.filter(DatasetFilter(genres=["DRAMA"]))
        mixed = loaded_store.filter(DatasetFilter(genres=["Drama"]))
        assert {t.show_id for t in lower} == {t.show_id for t in upper}
        assert {t.show_id for t in lower} == {t.show_id for t in mixed}

    def test_multi_genre_filter_is_any_of(self, loaded_store):
        ids = {t.show_id for t in loaded_store.filter(DatasetFilter(genres=["Comedy", "Documentary"]))}
        assert ids == {"s4", "s5", "s9"}

    def test_genre_with_ampersand_matches_correctly(self, loaded_store):
        ids = {t.show_id for t in loaded_store.filter(DatasetFilter(genres=["Action & Adventure"]))}
        assert "s10" in ids

    def test_unmatched_genre_returns_empty_list(self, loaded_store):
        assert loaded_store.filter(DatasetFilter(genres=["SciFi-Nonexistent"])) == []

    def test_partial_genre_name_does_not_match(self, loaded_store):
        assert loaded_store.filter(DatasetFilter(genres=["Dram"])) == []


# ── year_min / year_max filter ────────────────────────────────────────────────


class TestYearRangeFilter:
    def test_year_min_filters_out_older_titles(self, loaded_store):
        assert len(loaded_store.filter(DatasetFilter(year_min=2020))) == 4

    def test_year_max_filters_out_newer_titles(self, loaded_store):
        assert len(loaded_store.filter(DatasetFilter(year_max=2017))) == 3

    def test_year_range_combines_min_and_max(self, loaded_store):
        assert len(loaded_store.filter(DatasetFilter(year_min=2019, year_max=2020))) == 4

    def test_year_range_includes_boundary_years(self, loaded_store):
        ids = {t.show_id for t in loaded_store.filter(DatasetFilter(year_min=2022, year_max=2022))}
        assert ids == {"s3"}

    def test_impossible_year_range_returns_empty(self, loaded_store):
        assert loaded_store.filter(DatasetFilter(year_min=2025, year_max=2025)) == []


# ── maturity_ceiling filter (ordered enum) ────────────────────────────────────


class TestMaturityCeilingFilter:
    def test_ceiling_g_returns_only_g_rated(self, loaded_store):
        ids = {t.show_id for t in loaded_store.filter(DatasetFilter(maturity_ceiling="G"))}
        assert ids == {"s6"}

    def test_ceiling_pg_includes_g_and_tv_y_family(self, loaded_store):
        ids = {t.show_id for t in loaded_store.filter(DatasetFilter(maturity_ceiling="PG"))}
        assert "s5" in ids
        assert "s6" in ids
        assert "s10" in ids
        assert "s1" not in ids
        assert "s2" not in ids

    def test_ceiling_tv_ma_includes_most_titles(self, loaded_store):
        ids = {t.show_id for t in loaded_store.filter(DatasetFilter(maturity_ceiling="TV-MA"))}
        assert "s1" in ids
        assert "s2" in ids
        assert "s8" not in ids

    def test_ceiling_excludes_null_rated_titles(self, loaded_store):
        ids = {t.show_id for t in loaded_store.filter(DatasetFilter(maturity_ceiling="TV-MA"))}
        assert "s7" not in ids

    def test_unknown_ceiling_value_returns_empty(self, loaded_store):
        assert loaded_store.filter(DatasetFilter(maturity_ceiling="UNKNOWN-RATING")) == []


# ── country filter (substring match) ─────────────────────────────────────────


class TestCountryFilter:
    def test_exact_country_match(self, loaded_store):
        ids = {t.show_id for t in loaded_store.filter(DatasetFilter(country="France"))}
        assert ids == {"s4"}

    def test_substring_country_match(self, loaded_store):
        ids = {t.show_id for t in loaded_store.filter(DatasetFilter(country="United"))}
        assert ids == {"s1", "s2"}

    def test_country_filter_is_case_insensitive(self, loaded_store):
        lower = loaded_store.filter(DatasetFilter(country="france"))
        upper = loaded_store.filter(DatasetFilter(country="FRANCE"))
        assert {t.show_id for t in lower} == {t.show_id for t in upper}

    def test_country_filter_excludes_none_country_titles(self, loaded_store):
        results = loaded_store.filter(DatasetFilter(country="None"))
        assert all(t.show_id != "s6" for t in results)

    def test_unmatched_country_returns_empty(self, loaded_store):
        assert loaded_store.filter(DatasetFilter(country="Atlantis")) == []


# ── Combined multi-dimension filters ─────────────────────────────────────────


class TestCombinedFilters:
    def test_type_and_genre_combined(self, loaded_store):
        ids = {t.show_id for t in loaded_store.filter(DatasetFilter(content_type="TV Show", genres=["Drama"]))}
        assert ids == {"s3"}

    def test_type_and_year_combined(self, loaded_store):
        ids = {t.show_id for t in loaded_store.filter(DatasetFilter(content_type="Movie", year_min=2020))}
        assert ids == {"s5", "s9"}

    def test_genre_and_year_combined(self, loaded_store):
        ids = {t.show_id for t in loaded_store.filter(DatasetFilter(genres=["Comedy"], year_min=2020))}
        assert ids == {"s5", "s9"}

    def test_all_dimensions_combined_narrows_to_one(self, loaded_store):
        ids = {t.show_id for t in loaded_store.filter(DatasetFilter(
            content_type="Movie", genres=["Drama"],
            year_min=2019, year_max=2019,
            maturity_ceiling="PG-13", country="United States",
        ))}
        assert ids == {"s1"}

    def test_contradictory_filters_return_empty(self, loaded_store):
        assert loaded_store.filter(
            DatasetFilter(content_type="Movie", genres=["Drama"], maturity_ceiling="G")
        ) == []


# ── Zero-result and edge cases ────────────────────────────────────────────────


class TestEdgeCases:
    def test_filter_never_raises_on_no_results(self, loaded_store):
        assert loaded_store.filter(
            DatasetFilter(content_type="Movie", genres=["SciFi"], year_min=2100)
        ) == []

    def test_filter_returns_netflix_title_instances(self, loaded_store):
        from sommelier.domain.models import NetflixTitle
        assert all(isinstance(t, NetflixTitle) for t in loaded_store.filter(DatasetFilter(content_type="Movie")))

    def test_filter_on_empty_store_returns_empty(self, store):
        assert store.filter(DatasetFilter()) == []
