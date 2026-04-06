"""Task 12.6 — Performance test: retrieval pipeline against a 10,000-row dataset.

Verifies that combined filter() + tfidf_similarity() latency stays under 500 ms
on a 10,000-row fixture, leaving sufficient budget for two LLM API calls within
the 5-second SLA.

The large fixture is built once per session (module scope) to avoid rebuilding
it on every run. TF-IDF index build time is also measured and asserted under 5s.
"""

from __future__ import annotations

import time

import pytest

from sommelier.domain.models import PreferenceProfile
from sommelier.infrastructure.dataset_store import DatasetStore
from sommelier.ports.interfaces import DatasetFilter
from tests.fixtures import write_fixture_csv


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def large_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("perf") / "large.csv"
    return write_fixture_csv(path, n=10_000)


@pytest.fixture(scope="module")
def large_store(large_csv):
    ds = DatasetStore()
    ds.load_and_index(large_csv)
    return ds


# ── Performance tests ─────────────────────────────────────────────────────────


class TestRetrievalPerformance:
    def test_index_build_under_5_seconds(self, large_csv):
        """TF-IDF index build (load_and_index) must complete within 5 seconds."""
        ds = DatasetStore()
        start = time.monotonic()
        ds.load_and_index(large_csv)
        elapsed = time.monotonic() - start
        assert elapsed < 5.0, f"Index build took {elapsed:.2f}s — exceeds 5-second limit"

    def test_filter_plus_tfidf_under_500ms(self, large_store):
        """filter() + tfidf_similarity() combined must complete within 500 ms."""
        criteria = DatasetFilter(genres=["Drama"])
        query = "drama thriller adventure"

        start = time.monotonic()
        candidates = large_store.filter(criteria)
        scored = large_store.tfidf_similarity(query, candidates)
        elapsed = time.monotonic() - start

        assert elapsed < 0.5, (
            f"filter+tfidf took {elapsed*1000:.0f}ms — exceeds 500ms budget"
        )
        assert len(scored) > 0

    def test_filter_only_under_100ms(self, large_store):
        """filter() alone should be well under 100 ms on 10k rows."""
        criteria = DatasetFilter(content_type="Movie", genres=["Action"])

        start = time.monotonic()
        candidates = large_store.filter(criteria)
        elapsed = time.monotonic() - start

        assert elapsed < 0.1, f"filter() took {elapsed*1000:.0f}ms — too slow"
        assert isinstance(candidates, list)

    def test_tfidf_with_empty_query_under_500ms(self, large_store):
        """tfidf_similarity with empty query still returns within budget."""
        criteria = DatasetFilter()
        candidates = large_store.filter(criteria)

        start = time.monotonic()
        scored = large_store.tfidf_similarity("", candidates)
        elapsed = time.monotonic() - start

        assert elapsed < 0.5, (
            f"tfidf (empty query) took {elapsed*1000:.0f}ms — exceeds 500ms budget"
        )
        # Empty query → all scores are 0.0
        assert all(s.similarity_score == 0.0 for s in scored)

    def test_full_retrieval_returns_results(self, large_store):
        """Sanity check: 10k dataset with Drama filter returns non-empty results."""
        criteria = DatasetFilter(genres=["Drama"])
        candidates = large_store.filter(criteria)
        scored = large_store.tfidf_similarity("drama compelling story", candidates)
        assert len(scored) >= 3
