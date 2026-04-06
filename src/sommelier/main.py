"""Application factory — wires all components via dependency injection.

Task 11.1: build_app() assembles the full object graph from environment
variables and returns a ready-to-use ConversationOrchestrator.
"""

from __future__ import annotations

import os

from sommelier.application.conversation_orchestrator import ConversationOrchestrator
from sommelier.application.recommendation_engine import RecommendationEngine
from sommelier.application.response_generator import ResponseGenerator
from sommelier.application.session_manager import SessionManager
from sommelier.domain.candidate_retriever import CandidateRetriever
from sommelier.domain.preference_extractor import PreferenceExtractor
from sommelier.infrastructure.claude_adapter import ClaudeAdapter
from sommelier.infrastructure.dataset_store import DatasetStore


def build_app() -> ConversationOrchestrator:
    """Instantiate and wire all components; return the orchestrator.

    Reads DATABASE_URL from the environment.
    Raises DatasetLoadError if the connection fails or the table is empty.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    database_url = os.environ.get("DATABASE_URL")
    dataset = DatasetStore()
    dataset.load_and_index(database_url)

    llm = ClaudeAdapter(api_key=api_key)

    # Domain
    extractor = PreferenceExtractor(llm)
    retriever = CandidateRetriever(dataset)

    # Application
    session_manager = SessionManager()
    engine = RecommendationEngine(retriever)
    generator = ResponseGenerator(llm, dataset)

    return ConversationOrchestrator(
        session_manager=session_manager,
        preference_extractor=extractor,
        recommendation_engine=engine,
        response_generator=generator,
        dataset=dataset,
    )
