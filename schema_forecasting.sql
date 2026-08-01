-- Additive forecasting schema (NOT part of schema.sql; applied via
-- forecasting orchestration's init function). Stores forward-looking forecasts
-- with honest uncertainty and explicit status flags.

-- forecast_result: one row per (resource, series_type, horizon) for a successful
-- run; OR a single sentinel row (horizon='all') when a whole series is
-- insufficient_data or hit a model_error. status is a closed set.
CREATE TABLE IF NOT EXISTS forecast_result (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    resource_id         TEXT,                 -- FK to resource; always set (no macro rows here)
    series_type         TEXT NOT NULL,        -- 'price_proxy' | 'demand_intent_activity'
    horizon             TEXT NOT NULL,        -- '2w' | '1m' | '3m' | '6m' | 'all' (sentinel)
    run_at              TEXT NOT NULL,
    prophet_point       REAL,                 -- nullable (NULL on insufficient/model_error)
    prophet_lower       REAL,
    prophet_upper       REAL,
    holt_winters_point  REAL,                 -- point-only sanity check (no interval)
    chronos_point       REAL,                 -- nullable: optional 3rd model (zero-shot Chronos); NULL if unavailable
    disagreement_pct    REAL,                 -- Prophet-vs-Holt-Winters (unchanged semantics)
    status              TEXT NOT NULL,         -- 'ok'|'insufficient_data'|'model_error'|'unstable_disagreement'
    notes               TEXT,
    FOREIGN KEY (resource_id) REFERENCES resource (resource_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_forecast_resource_series
    ON forecast_result (resource_id, series_type);
