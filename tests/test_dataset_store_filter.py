"""Task 2.2 — Verify DatasetStore.filter() structured catalog filtering.

Fixtures (store, sample_csv, loaded_store) are provided by conftest.py.

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
        results = loaded_store.filter(DatasetFilter())
        assert len(results) == 10

    def test_empty_criteria_returns_list(self, loaded_store):
        results = loaded_store.filter(DatasetFilter())
        assert isinstance(results, list)


# ── content_type filter ───────────────────────────────────────────────────────


class TestContentTypeFilter:
    def test_movie_filter_returns_seven_movies(self, loaded_store):
        results = loaded_store.filter(DatasetFilter(content_type="Movie"))
        assert len(results) == 7

    def test_tv_show_filter_returns_three_shows(self, loaded_store):
        results = loaded_store.filter(DatasetFilter(content_type="TV Show"))
        assert len(results) == 3

    def test_movie_filter_excludes_tv_shows(self, loaded_store):
        results = loaded_store.filter(DatasetFilter(content_type="Movie"))
        ids = {t.show_id for t in results}
        assert "s2" not in ids
        assert "s3" not in ids
        assert "s10" not in ids

    def test_tv_show_filter_includes_correct_ids(self, loaded_store):
        results = loaded_store.filter(DatasetFilter(content_type="TV Show"))
        ids = {t.show_id for t in results}
        assert ids == {"s2", "s3", "s10"}


# ── genres filter (any-of, case-insensitive) ──────────────────────────────────


class TestGenresFilter:
    def test_drama_genre_matches_three_titles(self, loaded_store):
        # s1(Drama,Thriller), s3(Drama), s9(Comedy,Drama,Romance)
        results = loaded_store.filter(DatasetFilter(genres=["Drama"]))
        assert len(results) == 3

    def test_drama_genre_returns_correct_ids(self, loaded_store):
        results = loaded_store.filter(DatasetFilter(genres=["Drama"]))
        ids = {t.show_id for t in results}
        assert ids == {"s1", "s3", "s9"}

    def test_genre_match_is_case_insensitive(self, loaded_store):
        lower = loaded_store.filter(DatasetFilter(genres=["drama"]))
        upper = loaded_store.filter(DatasetFilter(genres=["DRAMA"]))
        mixed = loaded_store.filter(DatasetFilter(genres=["Drama"]))
        assert {t.show_id for t in lower} == {t.show_id for t in upper}
        assert {t.show_id for t in lower} == {t.show_id for t in mixed}

    def test_multi_genre_filter_is_any_of(self, loaded_store):
        # Comedy matches s5, s9; Documentary matches s4 → union = s4, s5, s9
        results = loaded_store.filter(DatasetFilter(genres=["Comedy", "Documentary"]))
        ids = {t.show_id for t in results}
        assert ids == {"s4", "s5", "s9"}

    def test_genre_with_ampersand_matches_correctly(self, loaded_store):
        # "Action & Adventure" is in s10
        results = loaded_store.filter(DatasetFilter(genres=["Action & Adventure"]))
        ids = {t.show_id for t in results}
        assert "s10" in ids

    def test_unmatched_genre_returns_empty_list(self, loaded_store):
        results = loaded_store.filter(DatasetFilter(genres=["SciFi-Nonexistent"]))
        assert results == []

    def test_partial_genre_name_does_not_match(self, loaded_store):
        # "Dram" is NOT "Drama" — must be exact (case-insensitive, not substring)
        results = loaded_store.filter(DatasetFilter(genres=["Dram"]))
        assert results == []


# ── year_min / year_max filter ────────────────────────────────────────────────


class TestYearRangeFilter:
    def test_year_min_filters_out_older_titles(self, loaded_store):
        # release_years: 2016,2017,2017,2018,2019,2019,2020,2020,2021,2022
        # year_min=2020 should include: s5(2020),s9(2020),s2(2021),s3(2022) = 4
        results = loaded_store.filter(DatasetFilter(year_min=2020))
        assert len(results) == 4

    def test_year_max_filters_out_newer_titles(self, loaded_store):
        # year_max=2017 includes: s8(2016),s10(2017),s7(2017) = 3
        results = loaded_store.filter(DatasetFilter(year_max=2017))
        assert len(results) == 3

    def test_year_range_combines_min_and_max(self, loaded_store):
        # 2019 ≤ year ≤ 2020: s1(2019),s6(2019),s5(2020),s9(2020) = 4
        results = loaded_store.filter(DatasetFilter(year_min=2019, year_max=2020))
        assert len(results) == 4

    def test_year_range_includes_boundary_years(self, loaded_store):
        results = loaded_store.filter(DatasetFilter(year_min=2022, year_max=2022))
        ids = {t.show_id for t in results}
        assert ids == {"s3"}

    def test_impossible_year_range_returns_empty(self, loaded_store):
        results = loaded_store.filter(DatasetFilter(year_min=2025, year_max=2025))
        assert results == []


# ── maturity_ceiling filter (ordered enum) ────────────────────────────────────


class TestMaturityCeilingFilter:
    def test_ceiling_g_returns_only_g_rated(self, loaded_store):
        # Only s6 is rated G; G ≤ G
        results = loaded_store.filter(DatasetFilter(maturity_ceiling="G"))
        ids = {t.show_id for t in results}
        assert ids == {"s6"}

    def test_ceiling_pg_includes_g_and_tv_y_family(self, loaded_store):
        # NETFLIX_RATINGS_ORDERED: G,TV-Y,TV-Y7,TV-G,PG,...
        # PG ceiling: G(s6), TV-Y7(s10), PG(s5) → 3 titles
        results = loaded_store.filter(DatasetFilter(maturity_ceiling="PG"))
        ids = {t.show_id for t in results}
        assert "s5" in ids   # PG ≤ PG
        assert "s6" in ids   # G ≤ PG
        assert "s10" in ids  # TV-Y7 ≤ PG
        assert "s1" not in ids  # PG-13 > PG
        assert "s2" not in ids  # TV-MA > PG

    def test_ceiling_tv_ma_includes_most_titles(self, loaded_store):
        # Excludes: s7(None/null rating), s8(NR — not in ordered list beyond TV-MA)
        # TV-MA is index 9; NR is index 11 (beyond TV-MA)
        results = loaded_store.filter(DatasetFilter(maturity_ceiling="TV-MA"))
        ids = {t.show_id for t in results}
        assert "s1" in ids   # PG-13
        assert "s2" in ids   # TV-MA
        assert "s8" not in ids  # NR is beyond TV-MA in NETFLIX_RATINGS_ORDERED

    def test_ceiling_excludes_null_rated_titles(self, loaded_store):
        # s7 has no rating (None) — should be excluded from any ceiling filter
        results = loaded_store.filter(DatasetFilter(maturity_ceiling="TV-MA"))
        ids = {t.show_id for t in results}
        assert "s7" not in ids

    def test_unknown_ceiling_value_returns_empty(self, loaded_store):
        # Unrecognized rating string → no title passes (or consistent empty)
        results = loaded_store.filter(DatasetFilter(maturity_ceiling="UNKNOWN-RATING"))
        assert results == []


# ── country filter (substring match) ─────────────────────────────────────────


class TestCountryFilter:
    def test_exact_country_match(self, loaded_store):
        results = loaded_store.filter(DatasetFilter(country="France"))
        ids = {t.show_id for t in results}
        assert ids == {"s4"}

    def test_substring_country_match(self, loaded_store):
        # "United" matches "United States"(s1) and "United Kingdom"(s2)
        results = loaded_store.filter(DatasetFilter(country="United"))
        ids = {t.show_id for t in results}
        assert ids == {"s1", "s2"}

    def test_country_filter_is_case_insensitive(self, loaded_store):
        lower = loaded_store.filter(DatasetFilter(country="france"))
        upper = loaded_store.filter(DatasetFilter(country="FRANCE"))
        assert {t.show_id for t in lower} == {t.show_id for t in upper}

    def test_country_filter_excludes_none_country_titles(self, loaded_store):
        # s6 has country=None — must not match any substring search
        results = loaded_store.filter(DatasetFilter(country="None"))
        assert all(t.show_id != "s6" for t in results)

    def test_unmatched_country_returns_empty(self, loaded_store):
        results = loaded_store.filter(DatasetFilter(country="Atlantis"))
        assert results == []


# ── Combined multi-dimension filters ─────────────────────────────────────────


class TestCombinedFilters:
    def test_type_and_genre_combined(self, loaded_store):
        # TV Show AND Drama: s3 only (s2 has no Drama)
        results = loaded_store.filter(
            DatasetFilter(content_type="TV Show", genres=["Drama"])
        )
        ids = {t.show_id for t in results}
        assert ids == {"s3"}

    def test_type_and_year_combined(self, loaded_store):
        # Movie AND year >= 2020: s5(2020), s9(2020)
        results = loaded_store.filter(
            DatasetFilter(content_type="Movie", year_min=2020)
        )
        ids = {t.show_id for t in results}
        assert ids == {"s5", "s9"}

    def test_genre_and_year_combined(self, loaded_store):
        # Comedy AND year >= 2020: s5(2020), s9(2020)
        results = loaded_store.filter(
            DatasetFilter(genres=["Comedy"], year_min=2020)
        )
        ids = {t.show_id for t in results}
        assert ids == {"s5", "s9"}

    def test_all_dimensions_combined_narrows_to_one(self, loaded_store):
        # Movie, Drama, 2019-2019, PG-13 ceiling, United States
        results = loaded_store.filter(
            DatasetFilter(
                content_type="Movie",
                genres=["Drama"],
                year_min=2019,
                year_max=2019,
                maturity_ceiling="PG-13",
                country="United States",
            )
        )
        ids = {t.show_id for t in results}
        assert ids == {"s1"}

    def test_contradictory_filters_return_empty(self, loaded_store):
        # Movie type + TV Show would be contradictory, but single type filter
        # G ceiling + TV-MA genre overlap = G ceiling is very restrictive
        results = loaded_store.filter(
            DatasetFilter(content_type="Movie", genres=["Drama"], maturity_ceiling="G")
        )
        # Drama movies: s1(PG-13), s9(TV-PG) — both exceed G ceiling
        assert results == []


# ── Zero-result and edge cases ────────────────────────────────────────────────


class TestEdgeCases:
    def test_filter_never_raises_on_no_results(self, loaded_store):
        # Should return [] not raise
        results = loaded_store.filter(
            DatasetFilter(content_type="Movie", genres=["SciFi"], year_min=2100)
        )
        assert results == []

    def test_filter_returns_netflix_title_instances(self, loaded_store):
        from sommelier.domain.models import NetflixTitle
        results = loaded_store.filter(DatasetFilter(content_type="Movie"))
        assert all(isinstance(t, NetflixTitle) for t in results)

    def test_filter_on_empty_store_returns_empty(self, store):
        # store fixture is not loaded — should return [] without crashing
        results = store.filter(DatasetFilter())
        assert results == []
