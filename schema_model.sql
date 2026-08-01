-- Additive model schema (NOT part of schema.sql; applied via train_fusion_model's
-- init function). Records every training run's provenance and every prediction,
-- so a real-data model can never be confused with a synthetic-validation one.

-- training_run: mandatory, machine-readable provenance for each trained model.
CREATE TABLE IF NOT EXISTS training_run (
    run_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at               TEXT NOT NULL,
    data_mode            TEXT NOT NULL,      -- 'real' | 'synthetic_validation'
    n_examples           INTEGER NOT NULL,
    n_positive           INTEGER,
    date_range_start     TEXT,               -- NULL for synthetic (no real dates)
    date_range_end       TEXT,
    random_seed          INTEGER NOT NULL,
    model_path           TEXT NOT NULL,
    calibration_applied  INTEGER NOT NULL,   -- 0/1
    notes                TEXT
);

-- prediction: one row per inference, linked to the training_run that served it.
CREATE TABLE IF NOT EXISTS prediction (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                  INTEGER NOT NULL,
    resource_id             TEXT,
    week_start              TEXT,
    horizon                 TEXT,
    raw_score               REAL,
    calibrated_probability  REAL,
    top_shap_features       TEXT,            -- JSON; NULL when SHAP gated off
    predicted_at            TEXT,
    FOREIGN KEY (run_id) REFERENCES training_run (run_id) ON DELETE CASCADE
);
