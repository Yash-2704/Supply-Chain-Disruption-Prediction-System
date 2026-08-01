-- Supply Chain Disruption Prediction System — foundational schema (SQLite).
-- Resource-first data model: `resource` is the primary entity; producers,
-- facilities, signals and time-series attach to it. JSON-shaped columns are
-- stored as TEXT holding valid JSON and parsed in Python.

-- resource: the primary entity. One row per tracked industrial resource.
CREATE TABLE IF NOT EXISTS resource (
    resource_id                  TEXT PRIMARY KEY,   -- stable slug, e.g. 'dram'
    name                         TEXT NOT NULL,
    category                     TEXT,
    hs_codes                     TEXT,               -- JSON array of HS code strings
    aliases                      TEXT,               -- JSON array of name variants
    substitutes                  TEXT,               -- JSON array of substitute resources
    hhi_country                  REAL,               -- country-concentration HHI (0-1)
    top_producing_countries      TEXT,               -- JSON array of {country, share} objects
    expansion_lead_time_months   INTEGER             -- months to bring new capacity online
);

-- producer: a company that produces one or more resources.
CREATE TABLE IF NOT EXISTS producer (
    producer_id  TEXT PRIMARY KEY,   -- stable slug, e.g. 'samsung'
    name         TEXT NOT NULL,
    tickers      TEXT                 -- JSON array of exchange ticker strings
);

-- producer_resource: many-to-many junction linking producers to the resources they make.
CREATE TABLE IF NOT EXISTS producer_resource (
    producer_id  TEXT NOT NULL,
    resource_id  TEXT NOT NULL,
    PRIMARY KEY (producer_id, resource_id),
    FOREIGN KEY (producer_id) REFERENCES producer (producer_id) ON DELETE CASCADE,
    FOREIGN KEY (resource_id) REFERENCES resource (resource_id) ON DELETE CASCADE
);

-- facility: a physical production/fab site belonging to a producer.
CREATE TABLE IF NOT EXISTS facility (
    facility_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    producer_id  TEXT NOT NULL,
    country      TEXT,
    city         TEXT,
    FOREIGN KEY (producer_id) REFERENCES producer (producer_id) ON DELETE CASCADE
);

-- signal: raw ingested disruption signals attached to a resource (and optionally a producer).
-- Intentionally left empty in this task; later ingestion prompts write here.
CREATE TABLE IF NOT EXISTS signal (
    signal_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT,
    signal_type  TEXT,
    resource_id  TEXT NOT NULL,
    producer_id  TEXT,               -- nullable
    country      TEXT,               -- nullable
    value        REAL,               -- nullable
    unit         TEXT,               -- nullable
    severity     REAL,               -- nullable
    confidence   REAL,               -- nullable
    source_name  TEXT,
    source_url   TEXT,
    raw_text     TEXT,
    ingested_at  TEXT,
    FOREIGN KEY (resource_id) REFERENCES resource (resource_id) ON DELETE CASCADE,
    FOREIGN KEY (producer_id) REFERENCES producer (producer_id) ON DELETE SET NULL
);

-- price_series: time-series of observed prices for a resource.
CREATE TABLE IF NOT EXISTS price_series (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    resource_id  TEXT,                 -- NULLABLE: macro (resource-agnostic) rows use NULL
    timestamp    TEXT,
    price_usd    REAL,
    unit         TEXT,
    source       TEXT,
    FOREIGN KEY (resource_id) REFERENCES resource (resource_id) ON DELETE CASCADE
);

-- capacity_record: reported production-capacity figures per producer/resource/year.
CREATE TABLE IF NOT EXISTS capacity_record (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    producer_id     TEXT NOT NULL,
    resource_id     TEXT NOT NULL,
    year            INTEGER,
    capacity_value  REAL,
    unit            TEXT,
    source          TEXT,
    FOREIGN KEY (producer_id) REFERENCES producer (producer_id) ON DELETE CASCADE,
    FOREIGN KEY (resource_id) REFERENCES resource (resource_id) ON DELETE CASCADE
);
