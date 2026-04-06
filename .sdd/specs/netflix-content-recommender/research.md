# Research & Design Decisions

---
**Purpose**: Capture discovery findings, architectural investigations, and rationale that inform the technical design.

---

## Summary
- **Feature**: `netflix-content-recommender`
- **Discovery Scope**: New Feature (Greenfield) — Full Discovery
- **Key Findings**:
  - A hybrid retrieval approach (structured Pandas filtering + TF-IDF semantic similarity) over the Netflix CSV dataset is sufficient for ≤10,000 titles without requiring a vector database or embeddings API, keeping the system self-contained and fast.
  - The Claude Messages API multi-turn conversation pattern (accumulating `messages` list per session) is the correct primitive for maintaining conversation context; structured preference extraction via a dedicated LLM call with a strict JSON output instruction avoids brittle regex parsing.
  - Hexagonal architecture (Ports & Adapters) is the best fit: it isolates the domain core (preference extraction, candidate retrieval, ranking) from the LLM adapter and the CLI adapter, enabling independent testing and future UI swaps without core changes.

---

## Research Log

### Netflix Titles Dataset Structure
- **Context**: The design must ground all recommendations in a real dataset; understanding its schema is critical for the DatasetStore and filtering contracts.
- **Sources Consulted**: Well-established Kaggle dataset "Netflix Movies and TV Shows" (shivamb/netflix-shows); widely documented in public literature.
- **Findings**:
  - Fields: `show_id` (str), `type` (Movie | TV Show), `title` (str), `director` (str, nullable), `cast` (str, nullable), `country` (str, nullable), `date_added` (str, nullable), `release_year` (int), `rating` (str — TV-MA, PG-13, R, PG, TV-14, TV-G, TV-PG, NR, G, TV-Y, TV-Y7, NC-17), `duration` (str — "90 min" or "2 Seasons"), `listed_in` (str — comma-separated genres), `description` (str).
  - ~8,800 titles as of the most recent version; well within the 10,000-title performance requirement.
  - `listed_in` contains multi-value genres as a single comma-separated string; must be split and normalized at load time.
  - `description` is the richest free-text field for semantic similarity matching.
- **Implications**: DatasetStore must normalize `listed_in` into a list and index descriptions for TF-IDF; filtering must support partial text match on `country`, range match on `release_year`, and exact match on `type` and `rating`.

### LLM Integration — Claude Messages API
- **Context**: Requirements demand natural-language understanding, preference extraction, rationale generation, and multilingual response; all require an LLM.
- **Sources Consulted**: Anthropic Python SDK (`anthropic` package ~0.47+); Claude Messages API (`POST /v1/messages`); established multi-turn conversation patterns.
- **Findings**:
  - Multi-turn conversation is achieved by accumulating a `messages: list[dict]` with alternating `user` / `assistant` roles; the full list is sent with every request — stateless on the server side.
  - System prompt is a top-level `system: str` field (not a message), ideal for injecting the sommelier persona, dataset constraints, and output format instructions.
  - Structured output (JSON preference profile) is reliably obtained by instructing the model in the system prompt and using a separate extraction call with `max_tokens` capped at ~300.
  - `claude-haiku-4-5-20251001` is the right model for lightweight extraction calls (low latency, low cost); `claude-sonnet-4-6` for richer rationale generation and complex intent disambiguation.
  - Tool use is not required — JSON-in-system-prompt is simpler and sufficient for this use case.
- **Implications**: Two LLM call types: (1) intent/preference extraction (Haiku, short output), (2) recommendation response generation (Sonnet, longer output). Session history list is the sole state that must be maintained per turn.

### Retrieval Strategy for Small Corpus
- **Context**: Requirement 3.2 prohibits hallucinated titles; all suggestions must come from the dataset. Requirement 2.5 requires relevance ranking. Requirement 7.1 requires sub-5-second response.
- **Sources Consulted**: scikit-learn TF-IDF + cosine similarity documentation; established RAG patterns for small corpora; FAISS and ChromaDB documentation.
- **Findings**:
  - For a corpus of ≤10,000 short text documents (title + description ≈ 50–120 words each), TF-IDF vectorization with cosine similarity (scikit-learn `TfidfVectorizer` + `cosine_similarity`) runs in <200 ms on a modern CPU — well within the 5-second budget.
  - A full vector database (ChromaDB, FAISS) introduces operational overhead (index serialization, embedding generation latency) that is unjustified at this scale.
  - Claude embeddings API (`text-embedding-3-small`) would improve semantic quality but adds API latency per query and cost; deferred to a future optimization.
  - Preferred pipeline: structured Pandas pre-filtering (type, genre, year, rating, country) → TF-IDF cosine similarity on `description` field → top-N candidates → LLM re-ranking with rationale generation.
- **Implications**: No vector database dependency. `scikit-learn` and `pandas` are sufficient for the retrieval layer. TF-IDF index is built once at startup from the loaded dataset.

### Conversational State Management
- **Context**: Requirements 4.4 and 5.1 require session-scoped preference accumulation and multi-turn context.
- **Findings**:
  - In-memory session state (Python dataclass) is sufficient for a CLI application; no persistence required by the requirements.
  - The `PreferenceProfile` must be a mutable accumulator, not a replacement on each turn — new signals are merged with existing ones.
  - The LLM message history and the structured `PreferenceProfile` are separate concerns: history drives conversational coherence; the profile drives retrieval filtering.
  - A `seen_title_ids` set within the session enforces the no-repeat requirement (2.4) without LLM involvement.
- **Implications**: `Session` dataclass holds both `conversation_history: list[Message]` and `preference_profile: PreferenceProfile` and `seen_title_ids: set[str]`.

### Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| Hexagonal (Ports & Adapters) | Domain core isolated behind interfaces; adapters for LLM, dataset, UI | High testability, swap LLM/UI without touching domain | More initial boilerplate | Best fit: multiple adapters (CLI today, web later) |
| Layered (MVC) | Controller → Service → Repository | Familiar, simple | Domain logic tends to leak into controllers | Adequate for simple CRUD; undersized here |
| Pipeline | Linear stages: input → extract → retrieve → rank → respond | Easy to trace | Rigid; hard to handle feedback loops and refinement turns | Good mental model but not the physical structure |
| Event-driven | Async events between components | Scalable | Heavyweight for a CLI app with no concurrency needs | Premature for this scope |

**Selected**: Hexagonal. The application has two external actors (LLM provider, CLI user) and one external resource (CSV dataset); hexagonal cleanly models all three as ports with adapters, while the domain core remains pure and testable.

---

## Design Decisions

### Decision: Two-Model LLM Strategy
- **Context**: Preference extraction requires structured output with low latency; recommendation responses require richer natural language quality.
- **Alternatives Considered**:
  1. Single model for everything (Sonnet) — simpler, higher cost and latency for extraction calls.
  2. Single model for everything (Haiku) — lower cost, but may produce lower-quality rationale text.
  3. Two models — Haiku for extraction, Sonnet for generation.
- **Selected Approach**: Two-model strategy.
- **Rationale**: Extraction calls are frequent (every user turn) and require only JSON output; Haiku handles this reliably and fast. Rationale generation benefits from Sonnet's quality. The `LLMPort` interface abstracts model selection from domain logic.
- **Trade-offs**: Two model configurations to maintain; mitigated by a single `LLMPort` interface.
- **Follow-up**: Benchmark extraction accuracy with Haiku during implementation; fall back to Sonnet if quality is insufficient.

### Decision: TF-IDF over Embeddings API
- **Context**: Semantic similarity is needed for relevance ranking, but the dataset is small.
- **Alternatives Considered**:
  1. Claude Embeddings API — higher semantic quality, adds API latency and cost.
  2. Sentence Transformers (local model) — good quality, adds ~500 MB model dependency.
  3. TF-IDF + cosine similarity (scikit-learn) — fast, zero external dependency, adequate for keyword-rich descriptions.
- **Selected Approach**: TF-IDF at MVP; embeddings as future enhancement.
- **Rationale**: Netflix descriptions are relatively keyword-rich (genre keywords, actor names, plot nouns). TF-IDF captures these well and runs entirely in-process with no API calls, keeping latency low and the system self-contained.
- **Trade-offs**: Lower semantic quality for abstract queries (e.g., "something that makes me feel nostalgic"). Mitigated by the LLM pre-processing step that converts vague mood signals into concrete genre/keyword terms before the similarity query.
- **Follow-up**: Evaluate embedding-based similarity after MVP if user satisfaction is low for mood-based queries.

### Decision: CLI Interface at MVP
- **Context**: Requirements specify a conversational interface but not a specific modality.
- **Alternatives Considered**:
  1. Web application (FastAPI + React) — richer UI, higher build cost.
  2. CLI with Rich library — zero frontend complexity, fast to implement, sufficient for conversational UX.
  3. Telegram / Slack bot — requires third-party account setup.
- **Selected Approach**: CLI with `rich` library for formatted output.
- **Rationale**: The core value is in the recommendation engine, not the UI shell. CLI enables full end-to-end delivery quickly. The hexagonal architecture ensures the CLI adapter can be replaced with a web adapter later without touching domain logic.
- **Trade-offs**: Less discoverable for non-technical users; deferred to a future interface adapter.

### Decision: In-Process Dataset (no database)
- **Context**: Requirement 3.1 requires loading the dataset at startup; no persistence requirements exist beyond the session.
- **Alternatives Considered**:
  1. SQLite — persistent, queryable with SQL, slight startup overhead.
  2. Pandas in-memory — fastest queries for filtered reads, zero setup.
  3. DuckDB — fast analytical queries on CSV directly.
- **Selected Approach**: Pandas in-memory with startup load.
- **Rationale**: The dataset is read-only and small (≤10,000 rows). Pandas loads a CSV of this size in <1 second. All filtering operations (genre, year, rating, type) are simple boolean mask operations. No write path exists, so database ACID properties are unnecessary.
- **Trade-offs**: Data is reloaded on each application restart (acceptable — no mutations). Memory footprint ~5–15 MB for 10k rows.

---

## Risks & Mitigations
- **LLM extraction instability**: Claude may occasionally return malformed JSON for preference extraction → Mitigated by strict system prompt with schema, JSON validation in `PreferenceExtractor`, and a fallback that treats the raw text as a keyword query.
- **5-second latency breach**: LLM API round-trips are the dominant latency source → Mitigated by using Haiku for extraction, limiting `max_tokens`, and running retrieval in-process.
- **Dataset quality gaps**: Many rows have null `director`, `cast`, or `country` → Mitigated by defensive null handling in `DatasetStore`; filters on nullable fields are applied only when explicitly requested.
- **Multilingual LLM response (Requirement 5.5)**: Claude natively supports multilingual response when instructed to reply in the user's language → Risk is low; system prompt must include the instruction explicitly.
- **Session memory growth**: Long conversations accumulate large `messages` lists sent to the API on every turn → Mitigated by a configurable `max_history_turns` truncation strategy (keep last N turns + system summary).

---

## References
- Anthropic Python SDK — `anthropic` package, `client.messages.create()` multi-turn pattern
- Netflix Movies and TV Shows dataset — Kaggle (shivamb/netflix-shows); CSV schema documented above
- scikit-learn TF-IDF — `sklearn.feature_extraction.text.TfidfVectorizer` + `sklearn.metrics.pairwise.cosine_similarity`
- Hexagonal Architecture (Alistair Cockburn, 2005) — ports-and-adapters pattern
- Rich Python library — terminal formatting for CLI interface
