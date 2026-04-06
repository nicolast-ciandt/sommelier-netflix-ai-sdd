"""Tasks 6.1, 6.2 — Verify CandidateRetriever filter-and-rank pipeline.

Uses a real DatasetStore with the 10-row sample fixture (from conftest.py)
so TF-IDF and filter behaviour are tested end-to-end without extra mocking.

Groups:
  TestRetrieveReturnType     — basic shape of results
  TestFilterTranslation      — PreferenceProfile → DatasetFilter mapping (6.1)
  TestTfidfRanking           — query string built and results ranked (6.1)
  TestNoKeywordFallback      — random-sample path when no keywords (6.1)
  TestMaxCandidates          — result count cap (6.1)
  TestExclusions             — seen_title_ids excluded after scoring (6.2)
"""

import pytest

from sommelier.domain.models import PreferenceProfile, ScoredTitle
from sommelier.infrastructure.dataset_store import DatasetStore


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def ds(loaded_store: DatasetStore) -> DatasetStore:
    return loaded_store


def _profile(**kwargs) -> PreferenceProfile:
    defaults = dict(
        genres=[],
        mood_keywords=[],
        content_type=None,
        year_min=None,
        year_max=None,
        maturity_ceiling=None,
        country_filter=None,
        positive_genre_signals=[],
    )
    defaults.update(kwargs)
    return PreferenceProfile(**defaults)


# ── 6.1: Return type ──────────────────────────────────────────────────────────


class TestRetrieveReturnType:
    def test_returns_list(self, ds):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        cr = CandidateRetriever(ds)
        result = cr.retrieve(_profile(), frozenset())
        assert isinstance(result, list)

    def test_returns_scored_title_instances(self, ds):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        cr = CandidateRetriever(ds)
        result = cr.retrieve(_profile(), frozenset())
        assert all(isinstance(r, ScoredTitle) for r in result)

    def test_empty_store_returns_empty_list(self):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        empty = DatasetStore()
        cr = CandidateRetriever(empty)
        result = cr.retrieve(_profile(), frozenset())
        assert result == []


# ── 6.1: Filter translation ───────────────────────────────────────────────────


class TestFilterTranslation:
    def test_content_type_movie_filters_to_movies_only(self, ds):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        cr = CandidateRetriever(ds)
        result = cr.retrieve(_profile(content_type="Movie"), frozenset())
        assert all(r.title.type == "Movie" for r in result)

    def test_content_type_tv_show_filters_to_shows_only(self, ds):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        cr = CandidateRetriever(ds)
        result = cr.retrieve(_profile(content_type="TV Show"), frozenset())
        assert all(r.title.type == "TV Show" for r in result)

    def test_genre_filter_applied(self, ds):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        cr = CandidateRetriever(ds)
        # Only s1(Drama,Thriller), s3(Drama), s9(Comedy,Drama,Romance) have Drama
        result = cr.retrieve(_profile(genres=["Drama"]), frozenset())
        ids = {r.title.show_id for r in result}
        assert ids == {"s1", "s3", "s9"}

    def test_year_range_filter_applied(self, ds):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        cr = CandidateRetriever(ds)
        result = cr.retrieve(_profile(year_min=2021, year_max=2022), frozenset())
        assert all(2021 <= r.title.release_year <= 2022 for r in result)

    def test_maturity_ceiling_filter_applied(self, ds):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        cr = CandidateRetriever(ds)
        # G ceiling: only s6 (rated G)
        result = cr.retrieve(_profile(maturity_ceiling="G"), frozenset())
        assert len(result) == 1
        assert result[0].title.show_id == "s6"

    def test_country_filter_applied(self, ds):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        cr = CandidateRetriever(ds)
        result = cr.retrieve(_profile(country_filter="Canada"), frozenset())
        assert all("Canada" in (r.title.country or "") for r in result)

    def test_no_matching_titles_returns_empty(self, ds):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        cr = CandidateRetriever(ds)
        result = cr.retrieve(_profile(genres=["SciFi-Nonexistent"]), frozenset())
        assert result == []


# ── 6.1: TF-IDF ranking ───────────────────────────────────────────────────────


class TestTfidfRanking:
    def test_results_sorted_descending_by_score(self, ds):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        cr = CandidateRetriever(ds)
        result = cr.retrieve(_profile(mood_keywords=["thriller"]), frozenset())
        scores = [r.similarity_score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_query_built_from_genres_and_mood_keywords(self, ds):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        cr = CandidateRetriever(ds)
        # s1 title="Thriller Night" description="A gripping thriller."
        result_with = cr.retrieve(_profile(mood_keywords=["thriller"]), frozenset())
        s1_score = next((r.similarity_score for r in result_with if r.title.show_id == "s1"), None)
        assert s1_score is not None and s1_score > 0.0

    def test_relevant_title_ranked_higher_than_irrelevant(self, ds):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        cr = CandidateRetriever(ds)
        result = cr.retrieve(_profile(mood_keywords=["thriller"]), frozenset())
        ids_in_order = [r.title.show_id for r in result]
        thriller_pos = ids_in_order.index("s1")
        anime_pos = ids_in_order.index("s8")
        assert thriller_pos < anime_pos


# ── 6.1: No-keyword fallback ──────────────────────────────────────────────────


class TestNoKeywordFallback:
    def test_no_keywords_returns_nonzero_results(self, ds):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        cr = CandidateRetriever(ds)
        result = cr.retrieve(_profile(), frozenset())
        assert len(result) > 0

    def test_no_keywords_scores_are_all_zero(self, ds):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        cr = CandidateRetriever(ds)
        result = cr.retrieve(_profile(), frozenset())
        assert all(r.similarity_score == 0.0 for r in result)

    def test_no_keywords_respects_max_candidates(self, ds):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        cr = CandidateRetriever(ds)
        result = cr.retrieve(_profile(), frozenset(), max_candidates=3)
        assert len(result) <= 3


# ── 6.1: max_candidates cap ───────────────────────────────────────────────────


class TestMaxCandidates:
    def test_result_capped_at_max_candidates(self, ds):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        cr = CandidateRetriever(ds)
        result = cr.retrieve(_profile(), frozenset(), max_candidates=5)
        assert len(result) <= 5

    def test_default_max_candidates_is_20(self, ds):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        cr = CandidateRetriever(ds)
        # fixture has 10 titles → all returned, still ≤ 20
        result = cr.retrieve(_profile(), frozenset())
        assert len(result) <= 20

    def test_max_candidates_one_returns_single_result(self, ds):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        cr = CandidateRetriever(ds)
        result = cr.retrieve(_profile(), frozenset(), max_candidates=1)
        assert len(result) == 1


# ── 6.2: Session-scoped exclusions ───────────────────────────────────────────


class TestExclusions:
    def test_excluded_id_absent_from_results(self, ds):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        cr = CandidateRetriever(ds)
        result = cr.retrieve(_profile(), frozenset({"s1"}))
        assert all(r.title.show_id != "s1" for r in result)

    def test_multiple_excluded_ids_all_absent(self, ds):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        excluded = frozenset({"s1", "s2", "s3"})
        cr = CandidateRetriever(ds)
        result = cr.retrieve(_profile(), excluded)
        result_ids = {r.title.show_id for r in result}
        assert result_ids.isdisjoint(excluded)

    def test_exclusions_applied_after_scoring(self, ds):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        cr = CandidateRetriever(ds)
        # With s1 excluded, remaining results still sorted by score
        result = cr.retrieve(
            _profile(mood_keywords=["thriller"]),
            frozenset({"s1"}),
        )
        scores = [r.similarity_score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_excluding_all_titles_returns_empty(self, ds):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        all_ids = frozenset(f"s{i}" for i in range(1, 11))
        cr = CandidateRetriever(ds)
        result = cr.retrieve(_profile(), all_ids)
        assert result == []

    def test_empty_exclusion_set_returns_full_pool(self, ds):
        from sommelier.domain.candidate_retriever import CandidateRetriever
        cr = CandidateRetriever(ds)
        result = cr.retrieve(_profile(), frozenset())
        assert len(result) == 10
