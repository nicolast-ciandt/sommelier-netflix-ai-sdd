# Implementation Plan

---

- [ ] 1. Set up project infrastructure and define shared domain contracts
- [x] 1.1 Initialize the Python project with dependency management and environment configuration
  - Create `pyproject.toml` with all required dependencies: `anthropic`, `pandas`, `scikit-learn`, `rich`, `python-dotenv`, `pytest`, `pytest-mock`
  - Configure `.env.example` documenting required environment variables: `ANTHROPIC_API_KEY`, `DATASET_PATH`, `EXTRACTION_MODEL`, `GENERATION_MODEL`, `MAX_HISTORY_TURNS`
  - Add a `.gitignore` that excludes `.env`, `__pycache__`, and virtual environment directories
  - _Requirements: 3.3, 7.1_

- [x] 1.2 Define all shared domain data types used across components
  - Define the `NetflixTitle` value object with all normalized fields: `show_id`, `type`, `title`, `director`, `cast`, `country`, `release_year`, `rating`, `duration`, `genres`, `description`
  - Define `DurationInfo` (value, unit), `Message` (role, content), `PreferenceProfile` (all signal fields), `PreferenceProfileDelta`, `Session` aggregate, `ScoredTitle`, `Recommendation`, `NoResultsResult`
  - Define `ExtractionResult` and `FeedbackResult` union types for preference extraction output
  - Define the shared error types: `DatasetLoadError`, `LLMUnavailableError`
  - _Requirements: 1.1, 2.1, 2.2, 3.1, 4.4, 5.1_

- [x] 1.3 Define the port interfaces that decouple all infrastructure from domain logic
  - Define `DatasetPort` with `filter()`, `get_by_id()`, `tfidf_similarity()`, and `title_count()` signatures
  - Define `LLMPort` with a `complete()` method accepting an `LLMRequest` (model role, system prompt, messages, max_tokens) and returning `LLMResponse`
  - Define `ConversationPort` with `start_session()` and `handle_turn()` signatures
  - Ensure all port definitions use strict Python type hints; no use of untyped or `Any` types
  - _Requirements: 1.1, 2.1, 3.1, 3.2_

---

- [ ] 2. Build the dataset store
- [x] 2.1 (P) Load and normalize the Netflix CSV into an in-memory catalog at startup
  - Read the CSV from the path configured in the environment; raise `DatasetLoadError` with a clear message if the file is missing, empty, or malformed
  - Normalize `listed_in` (split comma-separated string into `list[str]`), `cast` (same), `release_year` (coerce to int), and `duration` into `DurationInfo`
  - Treat all nullable fields (`director`, `country`, `rating`) safely — missing values become `None`, never empty strings
  - Log the loaded title count and file path to stderr on successful startup
  - _Requirements: 3.1, 3.3, 7.3_

- [x] 2.2 (P) Implement structured catalog filtering
  - Support boolean-mask filtering by `type`, `genres` (any-of match, case-insensitive), `release_year` range, `rating` maturity ceiling (ordered enum comparison), and `country` (substring match)
  - Apply only the filters present in the `DatasetFilter` criteria object; ignore unset fields
  - Return an empty list when no titles match — never raise on zero results
  - _Requirements: 2.3, 3.2, 3.4, 6.1_

- [x] 2.3 (P) Build the TF-IDF index and expose similarity scoring
  - Fit a `TfidfVectorizer` on a concatenation of `title` and `description` for every title at load time; store the fitted vectorizer and the sparse matrix
  - Implement `tfidf_similarity(query, candidates)` that transforms the query, computes cosine similarity against the candidate sub-matrix, and returns the candidates sorted descending by score
  - Verify index build time is well under 5 seconds for a 10,000-row dataset
  - _Requirements: 2.5, 3.2, 7.3_

---

- [ ] 3. Build the LLM adapter
- [x] 3.1 (P) Implement the Anthropic SDK adapter for the LLM port
  - Instantiate the `anthropic.Anthropic` client using the API key from the environment
  - Route requests with `model="extraction"` to `claude-haiku-4-5-20251001` and `model="generation"` to `claude-sonnet-4-6`
  - Serialize `LLMRequest.messages` to the SDK's `[{"role": ..., "content": ...}]` format and include the `system` prompt as a top-level field
  - Return a typed `LLMResponse` with `content`, `input_tokens`, and `output_tokens`
  - _Requirements: 1.1, 2.2, 5.5_

- [x] 3.2 (P) Add error handling and observability to the LLM adapter
  - Catch `anthropic.APIError`, network timeouts, and any SDK exception; wrap them in `LLMUnavailableError` with a descriptive message
  - Log model name, token counts, and latency in milliseconds for every completed call
  - _Requirements: 7.1, 7.2_

---

- [ ] 4. Build the session manager
- [x] 4.1 (P) Implement session creation and conversation history management
  - Create a new `Session` with a UUID id, empty history, empty `PreferenceProfile`, empty `seen_title_ids`, and `maturity_ceiling_locked=False`
  - Implement `append_message()` using an immutable update pattern (return a new `Session` rather than mutating in place)
  - Truncate `conversation_history` to the last `MAX_HISTORY_TURNS` entries when the limit is reached, preserving the most recent turns
  - _Requirements: 5.1_

- [x] 4.2 (P) Implement preference profile accumulation
  - Merge a `PreferenceProfileDelta` into the existing `PreferenceProfile` additively: append new genres and keywords without discarding prior signals; update `content_type`, `year_min`, `year_max`, `country_filter` only when the delta provides a non-null value
  - Accumulate `positive_genre_signals` from positive feedback across turns
  - _Requirements: 4.2, 4.3, 4.4_

- [x] 4.3 (P) Implement seen-title registry, rejection handling, and maturity ceiling locking
  - Implement `register_shown_titles()` that adds shown `show_id` values to the session's `seen_title_ids` set
  - Implement `apply_rejected_titles()` that adds rejected IDs to `seen_title_ids`, preventing them from appearing in future rounds
  - Implement `lock_maturity_ceiling()` that sets the ceiling and flips `maturity_ceiling_locked=True`; subsequent calls to raise the ceiling are silently ignored when locked
  - _Requirements: 2.4, 4.1, 4.4, 6.2_

---

- [ ] 5. Build the preference extractor
- [x] 5.1 Design and implement the preference extraction system prompt
  - Write a system prompt that instructs the model to return only a JSON object matching the `PreferenceProfileDelta` schema (genres, mood_keywords, content_type, year_min, year_max, maturity_ceiling, country_filter, excluded_title_ids, positive_genre_signals, needs_clarification, clarification_hint, has_conflict, conflict_description)
  - Include schema definition and a worked example in the system prompt to stabilize output format
  - Parse the LLM response with `json.loads`; on parse failure, return a fallback delta with `needs_clarification=True` and the raw message treated as keyword input
  - _Requirements: 1.1, 6.1_

- [x] 5.2 Implement ambiguity and conflict detection
  - Set `needs_clarification=True` and populate `clarification_hint` when the query is too short or yields no extractable signals
  - Set `has_conflict=True` and populate `conflict_description` when extracted signals contradict each other
  - Validate extracted `maturity_ceiling` values against the known Netflix rating enum; replace unrecognized values with `None`
  - _Requirements: 1.2, 1.3_

- [x] 5.3 Implement feedback extraction mode
  - Add a `mode="feedback"` variant that instructs the LLM to parse rejection signals (populating `excluded_title_ids`) and positive reinforcement signals (populating `positive_genre_signals`) from user feedback messages
  - Ensure feedback extraction merges into the existing profile without clearing prior preference signals
  - _Requirements: 1.4, 4.2, 4.3_

---

- [ ] 6. Build the candidate retriever
- [x] 6.1 Implement the combined filter-and-rank retrieval pipeline
  - Translate the current `PreferenceProfile` into a `DatasetFilter` and apply it via the dataset port to obtain a pre-filtered candidate pool
  - Build a query string from the profile's `genres` and `mood_keywords` and score the filtered pool using `DatasetPort.tfidf_similarity()`
  - Return up to `max_candidates` (default 20) `ScoredTitle` objects sorted by descending similarity score
  - When no genre/mood keywords exist, return a random sample of filtered candidates with `similarity_score=0.0`
  - _Requirements: 2.3, 2.5, 3.2_

- [x] 6.2 Apply session-scoped exclusions before returning candidates
  - Remove any `ScoredTitle` whose `show_id` is in the session's `seen_title_ids` before returning the ranked list
  - Ensure exclusions are applied after filtering and scoring to avoid skewing TF-IDF similarity computation
  - _Requirements: 2.4, 4.1_

---

- [ ] 7. Build the recommendation engine
- [x] 7.1 (P) Implement recommendation orchestration with result-count enforcement
  - Call the candidate retriever with the current `PreferenceProfile` and the session's `seen_title_ids`
  - Trim the ranked candidate list to between 3 and 10 items; if fewer than 3 candidates are available after all filters and exclusions, return a `NoResultsResult` rather than a partial list
  - Wrap each selected `NetflixTitle` in a `Recommendation` object (rationale left empty at this stage — populated by the response generator)
  - _Requirements: 2.1, 2.4_

- [x] 7.2 (P) Handle empty-results and produce structured no-results output
  - Detect whether the empty result is due to zero matching titles or all candidates being already seen
  - Return a `NoResultsResult` with the appropriate `reason` and a human-readable `suggestion` (e.g., "Try relaxing the genre filter" or "You've seen all matching titles — try broadening your search")
  - _Requirements: 6.3_

---

- [ ] 8. Build the response generator
- [x] 8.1 (P) Implement recommendation response generation with per-title rationale
  - Send the list of recommended `NetflixTitle` objects plus the current `PreferenceProfile` summary to the Sonnet model and request a formatted response that includes a personalized rationale for each title
  - Inject the user's detected language into the system prompt so the response is returned in that language
  - Include title, type, release year, and genres in the structured context sent to the LLM
  - _Requirements: 2.2, 5.5_

- [x] 8.2 (P) Implement title detail responses and catalog-miss handling
  - For title detail questions, retrieve the full `NetflixTitle` from the dataset port using `get_by_id()` and pass the description and metadata to the LLM for a concise answer
  - When the user asks about a title not present in the dataset, return a user-friendly message confirming it is not in the current catalog without hallucinating details
  - _Requirements: 5.3, 5.4_

- [x] 8.3 (P) Implement clarification, conflict, and no-results response generation
  - Generate a natural-language clarifying question using the `clarification_hint` from the preference extractor result
  - Generate a natural-language explanation for `NoResultsResult` that reflects the actual reason and the suggested relaxation
  - All generated responses must respect the user's language (inject language into every system prompt)
  - _Requirements: 1.2, 1.3, 5.5, 6.3_

---

- [ ] 9. Build the conversation orchestrator
- [x] 9.1 Implement session start and turn intent routing
  - On `start_session()`, create a new session via the session manager, generate a greeting message with ResponseGenerator, and return both
  - On each `handle_turn()`, first append the user message to the session, then classify the turn intent: new recommendation, feedback/refinement, title detail question, or out-of-catalog query
  - _Requirements: 1.4, 5.1, 5.2_

- [x] 9.2 Implement the recommendation turn pipeline end-to-end
  - Route recommendation and refinement turns through: PreferenceExtractor → clarification branch or RecommendationEngine → ResponseGenerator → SessionManager (register shown titles)
  - Route feedback turns through: PreferenceExtractor (feedback mode) → SessionManager (apply delta) → RecommendationEngine → ResponseGenerator
  - Route title detail turns through: ResponseGenerator (title detail or catalog-miss path)
  - _Requirements: 1.2, 1.3, 2.1, 2.2, 4.1, 4.2, 4.3, 5.3, 5.4_

- [x] 9.3 Implement the timeout guard and error recovery
  - Wrap every LLM and retrieval call in a 5-second deadline; on timeout or `LLMUnavailableError`, return a user-friendly retry prompt without crashing the session
  - On any unexpected exception, log the error and return a generic recovery message, keeping the session state unchanged
  - _Requirements: 7.1, 7.2_

---

- [ ] 10. Build the CLI adapter
- [x] 10.1 Implement Rich-formatted conversation display
  - Render each assistant response in a styled `rich.panel.Panel` with title "Sommelier"; render each user turn with a contrasting style
  - Format recommendation lists with numbered entries showing title, type, year, genres, and rationale as a readable Rich layout
  - Display startup information (dataset loaded, title count) and the greeting message before accepting user input
  - _Requirements: 2.2, 3.1, 5.2_

- [x] 10.2 Implement the main conversation loop
  - Accept user input line by line; pass each turn to `ConversationOrchestrator.handle_turn()` and display the response
  - Handle `KeyboardInterrupt` and an explicit `quit`/`exit` command with a graceful farewell message
  - Pass updated session state from each turn into the next call
  - _Requirements: 5.1, 7.2_

---

- [ ] 11. Wire all components and verify end-to-end flow
- [x] 11.1 Assemble the application via dependency injection
  - Instantiate `DatasetStore` (with dataset path from env), `ClaudeAdapter`, `SessionManager`, `PreferenceExtractor`, `CandidateRetriever`, `RecommendationEngine`, `ResponseGenerator`, and `ConversationOrchestrator`
  - Inject adapters into domain components through their port interfaces; no component should hold a direct reference to a concrete adapter class
  - Create an application entry point (`__main__.py` or equivalent) that boots the dataset store, wires all components, and hands control to the CLI adapter
  - _Requirements: 3.1, 3.3_

- [x] 11.2 Verify the full conversation flow with a real dataset sample
  - Run a smoke test: start a session, enter a free-text preference query, verify 3–10 recommendations are returned with rationale, enter a rejection, verify the rejected title is absent from the next round, enter a follow-up constraint, verify it is applied without clearing prior preferences
  - Confirm responses appear within 5 seconds for a typical query on the full dataset
  - _Requirements: 1.1, 1.4, 2.1, 2.4, 4.1, 4.3, 7.1_

---

- [ ] 12. Write the automated test suite
- [x] 12.1 (P) Unit test the dataset store
  - Test CSV load success with a 10-row fixture; verify `listed_in` normalization, null handling, and `DatasetLoadError` on missing file
  - Test each filter dimension in isolation: type, genre (case-insensitive), year range, maturity ceiling, country
  - Test `tfidf_similarity()` returns candidates in descending score order and handles an empty query gracefully
  - _Requirements: 3.1, 3.3, 3.4, 7.3_

- [x] 12.2 (P) Unit test preference extraction
  - Test extraction with valid JSON LLM response: verify all fields are mapped to the delta correctly
  - Test malformed JSON response: verify fallback to `needs_clarification=True` with raw text as keyword
  - Test ambiguous input (single-word query): verify `needs_clarification=True` and non-empty `clarification_hint`
  - Test conflicting input: verify `has_conflict=True` and non-empty `conflict_description`
  - Test feedback mode: verify `excluded_title_ids` and `positive_genre_signals` are populated correctly
  - _Requirements: 1.1, 1.2, 1.3, 4.2, 4.3_

- [x] 12.3 (P) Unit test session manager state transitions
  - Test preference profile accumulation over three turns: verify signals compound, not replace
  - Test `maturity_ceiling_locked`: verify ceiling cannot be raised once locked
  - Test `seen_title_ids` accumulation: verify IDs from multiple rounds are all excluded
  - Test history truncation: verify history length never exceeds `MAX_HISTORY_TURNS`
  - _Requirements: 2.4, 4.1, 4.2, 4.3, 4.4, 5.1, 6.2_

- [x] 12.4 Integration test the full recommendation turn cycle
  - Use a real `DatasetStore` with a 50-title fixture and a mocked `LLMPort`
  - Verify `ConversationOrchestrator.handle_turn()` with a preference query returns a session with updated history and between 3–10 recommendations
  - Verify the maturity filter is respected end-to-end when a maturity preference is in the profile
  - _Requirements: 1.1, 2.1, 2.3, 6.1, 6.2_

- [x] 12.5 Integration test the feedback and refinement flow
  - Start a session, run one recommendation turn, then send a rejection message; verify the rejected title is absent from the next round's results
  - Run a follow-up instruction turn (e.g., "only from the 90s") and verify the year filter is applied on top of existing genre preferences
  - _Requirements: 4.1, 4.2, 4.3_

- [x]* 12.6 Performance test the retrieval pipeline against a 10,000-row dataset
  - Load a 10,000-row fixture into `DatasetStore` and measure combined `filter()` + `tfidf_similarity()` latency
  - Assert total retrieval latency is under 500 ms to leave budget for two LLM API calls within the 5-second SLA
  - _Requirements: 7.1, 7.3_
