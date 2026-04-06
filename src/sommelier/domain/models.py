"""Shared domain data types for the Netflix Content Recommender.

All value objects are frozen dataclasses to prevent accidental mutation.
Aggregates (Session, PreferenceProfile) use regular dataclasses because
their fields are updated via the immutable-update pattern in SessionManager
(each mutation returns a new instance rather than mutating in place).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ── Value Objects ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DurationInfo:
    """Parsed duration from the dataset's 'duration' field."""

    value: int
    unit: Literal["min", "Seasons"]


@dataclass(frozen=True)
class NetflixTitle:
    """Immutable value object representing a single Netflix catalog entry."""

    show_id: str
    type: Literal["Movie", "TV Show"]
    title: str
    director: str | None
    cast: tuple[str, ...]
    country: str | None
    release_year: int
    rating: str | None
    duration: DurationInfo | None
    genres: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class Message:
    """A single turn in the conversation history."""

    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class PreferenceProfileDelta:
    """Immutable result of one preference or feedback extraction call.

    Carries new signals to be merged into the session's PreferenceProfile.
    Fields default to empty/None so callers only need to set what changed.
    """

    genres: tuple[str, ...] = field(default_factory=tuple)
    mood_keywords: tuple[str, ...] = field(default_factory=tuple)
    content_type: str | None = None
    year_min: int | None = None
    year_max: int | None = None
    maturity_ceiling: str | None = None
    country_filter: str | None = None
    excluded_title_ids: tuple[str, ...] = field(default_factory=tuple)
    positive_genre_signals: tuple[str, ...] = field(default_factory=tuple)
    needs_clarification: bool = False
    clarification_hint: str | None = None
    has_conflict: bool = False
    conflict_description: str | None = None


@dataclass(frozen=True)
class ScoredTitle:
    """A NetflixTitle paired with a TF-IDF cosine similarity score (0.0–1.0)."""

    title: NetflixTitle
    similarity_score: float


@dataclass(frozen=True)
class Recommendation:
    """A title selected for recommendation, optionally annotated with rationale.

    The ``rationale`` field is populated by ResponseGenerator after the
    RecommendationEngine selects the candidates; it defaults to an empty string.
    """

    title: NetflixTitle
    relevance_score: float
    rationale: str = ""


@dataclass(frozen=True)
class NoResultsResult:
    """Returned by RecommendationEngine when no qualifying titles are found."""

    reason: Literal["no_matching_titles", "all_seen"]
    suggestion: str


# ── Aggregates ────────────────────────────────────────────────────────────────


@dataclass
class PreferenceProfile:
    """Accumulated user preference signals for the current session.

    Starts empty and grows additively across conversation turns via
    SessionManager.apply_delta().  List fields use default_factory to
    ensure each Session gets its own independent lists.
    """

    genres: list[str] = field(default_factory=list)
    mood_keywords: list[str] = field(default_factory=list)
    content_type: str | None = None
    year_min: int | None = None
    year_max: int | None = None
    maturity_ceiling: str | None = None
    country_filter: str | None = None
    positive_genre_signals: list[str] = field(default_factory=list)


@dataclass
class Session:
    """Root aggregate representing a single conversation session.

    All mutations are performed by SessionManager using an immutable-update
    pattern (each method returns a new Session instance).  Fields with
    mutable defaults use default_factory to prevent cross-instance sharing.
    """

    id: str
    conversation_history: list[Message] = field(default_factory=list)
    preference_profile: PreferenceProfile = field(default_factory=PreferenceProfile)
    seen_title_ids: frozenset[str] = field(default_factory=frozenset)
    maturity_ceiling_locked: bool = False


# ── Type Aliases ──────────────────────────────────────────────────────────────

# Result of a preference-extraction LLM call (may flag clarification/conflict).
ExtractionResult = PreferenceProfileDelta

# Result of a feedback-extraction LLM call (carries rejected/positive signals).
FeedbackResult = PreferenceProfileDelta

# Output of RecommendationEngine: either a ranked list or an empty-result explanation.
RecommendationOutput = list[Recommendation] | NoResultsResult


# ── Domain Error Types ────────────────────────────────────────────────────────


class DatasetLoadError(Exception):
    """Raised when the Netflix dataset CSV cannot be found, read, or parsed."""


class LLMUnavailableError(Exception):
    """Raised when the LLM provider is unreachable or returns an unexpected error."""


# ── Domain Constants ──────────────────────────────────────────────────────────

# Known Netflix rating strings, ordered from least to most restrictive.
# Used by PreferenceExtractor to validate maturity_ceiling values.
NETFLIX_RATINGS_ORDERED: tuple[str, ...] = (
    "G",
    "TV-Y",
    "TV-Y7",
    "TV-G",
    "PG",
    "TV-PG",
    "PG-13",
    "TV-14",
    "R",
    "TV-MA",
    "NC-17",
    "NR",
)
