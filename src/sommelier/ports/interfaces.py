"""Port interfaces (driven and driving) for the Netflix Content Recommender.

Each port is a ``typing.Protocol`` marked ``@runtime_checkable`` so that
concrete adapter implementations can be verified at test time with
``isinstance``.  No business logic lives here — only contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from sommelier.domain.models import (
    Message,
    NetflixTitle,
    ScoredTitle,
    Session,
)


# ── Dataset Port ──────────────────────────────────────────────────────────────


@dataclass
class DatasetFilter:
    """Criteria passed to ``DatasetPort.filter()``.

    All fields are optional; unset fields (``None``) are ignored by the
    implementation, so callers only specify the dimensions they care about.
    """

    content_type: Literal["Movie", "TV Show"] | None = None
    genres: list[str] | None = None
    year_min: int | None = None
    year_max: int | None = None
    maturity_ceiling: str | None = None
    country: str | None = None


@runtime_checkable
class DatasetPort(Protocol):
    """Driven port: read-only access to the Netflix catalog dataset."""

    def filter(self, criteria: DatasetFilter) -> list[NetflixTitle]:
        """Return all titles matching *criteria*.  Returns ``[]`` on no match."""
        ...

    def get_by_id(self, show_id: str) -> NetflixTitle | None:
        """Return the title with the given ``show_id``, or ``None`` if absent."""
        ...

    def tfidf_similarity(
        self, query: str, candidates: list[NetflixTitle]
    ) -> list[ScoredTitle]:
        """Score *candidates* against *query* using TF-IDF cosine similarity.

        Returns the same candidates sorted by descending ``similarity_score``.
        """
        ...

    def title_count(self) -> int:
        """Return the total number of titles loaded in the catalog."""
        ...


# ── LLM Port ──────────────────────────────────────────────────────────────────


@dataclass
class LLMRequest:
    """Input to a single LLM completion call.

    ``model`` is a semantic role, not a model ID.  The adapter resolves the
    role to the concrete model name (e.g. ``"extraction"`` → Haiku).
    """

    system_prompt: str
    messages: list[Message]
    model: Literal["extraction", "generation"]
    max_tokens: int
    temperature: float = 0.3


@dataclass(frozen=True)
class LLMResponse:
    """Immutable result of a single LLM completion call."""

    content: str
    input_tokens: int
    output_tokens: int


@runtime_checkable
class LLMPort(Protocol):
    """Driven port: access to the large-language-model provider."""

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute a completion and return the model's response.

        Raises ``LLMUnavailableError`` on API failure or network timeout.
        """
        ...


# ── Conversation Port ─────────────────────────────────────────────────────────


@runtime_checkable
class ConversationPort(Protocol):
    """Driving port: the interface through which the CLI adapter talks to the
    application core (``ConversationOrchestrator``).
    """

    def start_session(self) -> tuple[Session, str]:
        """Create a new session and return ``(session, greeting_message)``."""
        ...

    def handle_turn(
        self,
        user_message: str,
        session: Session,
    ) -> tuple[Session, str]:
        """Process one user turn and return ``(updated_session, assistant_response)``."""
        ...
