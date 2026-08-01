-- Additive explanations schema (NOT part of schema.sql; applied via
-- generate_explanation's init function). RAG narratives grounded in retrieved
-- canonical evidence, with citations validated against what was actually retrieved.

-- explanation: one row per generation attempt (new timestamped rows; explanations
-- may legitimately be regenerated as evidence/model state changes). is_real_data_model
-- is FIXED at generation time (copied from the linked prediction, or 0 when none/synthetic)
-- and must never be re-inferred later.
CREATE TABLE IF NOT EXISTS explanation (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    resource_id          TEXT,
    prediction_id        INTEGER,             -- FK to prediction; NULL in evidence-only mode
    narrative            TEXT,                -- NULL unless status='ok'
    citations            TEXT,                -- JSON list of {signal_id, excerpt}; NULL unless 'ok'
    evidence_count       INTEGER NOT NULL,
    llm_provider         TEXT,                -- 'gemini'|'groq'|NULL
    is_real_data_model   INTEGER NOT NULL,    -- 0/1, fixed at generation time
    status               TEXT NOT NULL,        -- 'ok'|'insufficient_evidence'|'llm_error'|'invalid_citations'
    generated_at         TEXT NOT NULL,
    FOREIGN KEY (resource_id) REFERENCES resource (resource_id) ON DELETE CASCADE,
    FOREIGN KEY (prediction_id) REFERENCES prediction (id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_explanation_resource ON explanation (resource_id);
