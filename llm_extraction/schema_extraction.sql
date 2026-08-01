-- Additive extraction schema (NOT part of schema.sql; applied via
-- run_extraction.init_extraction_schema). Holds the structured parts of the
-- LLM's output that have no home on the signal table. severity/confidence go
-- directly onto signal; everything else structured + auditable lives here.

-- event_extraction: one row per successfully-extracted signal. Always written
-- together with the signal's severity/confidence, so an extraction record never
-- exists without a corresponding non-NULL severity.
CREATE TABLE IF NOT EXISTS event_extraction (
    signal_id           INTEGER PRIMARY KEY,   -- FK to signal.signal_id; one per signal
    event_type          TEXT,                  -- closed set (see validator.EVENT_TYPES)
    extracted_entities  TEXT,                  -- JSON list of entity strings
    llm_provider        TEXT,                  -- 'gemini' | 'groq' (who actually served it)
    llm_model           TEXT,
    raw_llm_response    TEXT,                  -- verbatim model output, for audit
    extracted_at        TEXT,
    FOREIGN KEY (signal_id) REFERENCES signal (signal_id) ON DELETE CASCADE
);
