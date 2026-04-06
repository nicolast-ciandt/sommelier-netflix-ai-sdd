"""Verify DatasetStore TF-IDF index and similarity scoring."""

import pytest

from sommelier.domain.models import NetflixTitle, ScoredTitle
from sommelier.infrastructure.dataset_store import DatasetStore


# ── Index build ───────────────────────────────────────────────────────────────


class TestIndexBuild:
    def test_load_and_index_builds_tfidf_index(self, loaded_store):
        assert loaded_store._tfidf_vectorizer is not None
        assert loaded_store._tfidf_matrix is not None

    def test_tfidf_matrix_row_count_matches_title_count(self, loaded_store):
        assert loaded_store._tfidf_matrix.shape[0] == loaded_store.title_count()

    def test_tfidf_matrix_has_nonzero_columns(self, loaded_store):
        assert loaded_store._tfidf_matrix.shape[1] > 0


# ── Return type and structure ─────────────────────────────────────────────────


class TestReturnType:
    def test_returns_list(self, loaded_store):
        result = loaded_store.tfidf_similarity("thriller drama", loaded_store._titles_list[:3])
        assert isinstance(result, list)

    def test_returns_scored_title_instances(self, loaded_store):
        result = loaded_store.tfidf_similarity("action", loaded_store._titles_list[:3])
        assert all(isinstance(item, ScoredTitle) for item in result)

    def test_result_length_equals_candidate_count(self, loaded_store):
        result = loaded_store.tfidf_similarity("drama", loaded_store._titles_list[:5])
        assert len(result) == 5

    def test_scored_title_wraps_netflix_title(self, loaded_store):
        candidates = [loaded_store.get_by_id("s1")]
        result = loaded_store.tfidf_similarity("thriller", candidates)
        assert result[0].title is candidates[0]


# ── Ordering ──────────────────────────────────────────────────────────────────


class TestOrdering:
    def test_results_sorted_descending_by_score(self, loaded_store):
        result = loaded_store.tfidf_similarity("thriller drama", loaded_store._titles_list[:])
        scores = [r.similarity_score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_relevant_title_ranks_above_irrelevant(self, loaded_store):
        s1 = loaded_store.get_by_id("s1")
        s8 = loaded_store.get_by_id("s8")
        result = loaded_store.tfidf_similarity("thriller", [s1, s8])
        assert result[0].title.show_id == "s1"

    def test_anime_query_ranks_anime_title_higher(self, loaded_store):
        s1 = loaded_store.get_by_id("s1")
        s8 = loaded_store.get_by_id("s8")
        result = loaded_store.tfidf_similarity("anime", [s1, s8])
        assert result[0].title.show_id == "s8"


# ── Score range ───────────────────────────────────────────────────────────────


class TestScoreRange:
    def test_similarity_scores_are_between_zero_and_one(self, loaded_store):
        result = loaded_store.tfidf_similarity("drama comedy", loaded_store._titles_list[:])
        for item in result:
            assert 0.0 <= item.similarity_score <= 1.0

    def test_exact_title_word_produces_nonzero_score(self, loaded_store):
        s1 = loaded_store.get_by_id("s1")
        result = loaded_store.tfidf_similarity("Thriller Night", [s1])
        assert result[0].similarity_score > 0.0


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_candidate_list_returns_empty(self, loaded_store):
        assert loaded_store.tfidf_similarity("drama", []) == []

    def test_empty_query_returns_all_candidates_with_zero_score(self, loaded_store):
        candidates = loaded_store._titles_list[:4]
        result = loaded_store.tfidf_similarity("", candidates)
        assert len(result) == 4
        assert all(item.similarity_score == 0.0 for item in result)

    def test_whitespace_only_query_returns_zero_scores(self, loaded_store):
        result = loaded_store.tfidf_similarity("   ", loaded_store._titles_list[:3])
        assert all(item.similarity_score == 0.0 for item in result)

    def test_single_candidate_returns_list_of_one(self, loaded_store):
        result = loaded_store.tfidf_similarity("adventure series", [loaded_store.get_by_id("s2")])
        assert len(result) == 1

    def test_unknown_query_term_returns_zero_scores(self, loaded_store):
        result = loaded_store.tfidf_similarity("xyzzy-nonexistent-term", loaded_store._titles_list[:])
        assert all(item.similarity_score == 0.0 for item in result)
