-- Additive labels schema (NOT part of schema.sql; applied via build_labels'
-- init function). Two provenance-tagged label sources coexist per resource/week.

-- label: training labels for the fusion model. Curated (hand-authored domain
-- knowledge) and price_derived (mechanical anomaly rule) rows are kept DISTINCT
-- and may disagree; no reconciliation happens here. One row max per
-- (resource_id, week_start, source).
CREATE TABLE IF NOT EXISTS label (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    resource_id    TEXT NOT NULL,
    week_start     TEXT NOT NULL,          -- ISO Monday of the week, 'YYYY-MM-DD'
    label          INTEGER,                -- 0 or 1 when status='ok'; NULL when insufficient_data
    source         TEXT NOT NULL,          -- 'curated_episode' | 'price_derived'
    status         TEXT NOT NULL,          -- 'ok' | 'insufficient_data'
    justification  TEXT,                   -- required for curated; computed note for price_derived
    created_at     TEXT,
    UNIQUE (resource_id, week_start, source),
    FOREIGN KEY (resource_id) REFERENCES resource (resource_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_label_resource_week ON label (resource_id, week_start);
