-- Additive clustering schema (NOT part of schema.sql; applied via
-- clustering.embed_and_cluster.init_clustering_schema). Records near-duplicate
-- cluster membership for news_raw signals without ever mutating the signal table.

-- news_cluster_membership: one row per news_raw signal. Each signal belongs to
-- exactly one cluster and is either the canonical representative of that cluster
-- (is_canonical=1) or redundant coverage of the canonical row (is_canonical=0).
CREATE TABLE IF NOT EXISTS news_cluster_membership (
    signal_id    TEXT PRIMARY KEY,        -- str(signal.signal_id); one row per signal
    cluster_id   TEXT NOT NULL,           -- stable id: 'c_<canonical_signal_id>'
    is_canonical INTEGER NOT NULL DEFAULT 0,  -- 1 = the cluster's representative row
    processed_at TEXT,
    FOREIGN KEY (signal_id) REFERENCES signal (signal_id) ON DELETE CASCADE
);

-- Fast lookup of all members of a cluster (and its canonical row).
CREATE INDEX IF NOT EXISTS idx_ncm_cluster ON news_cluster_membership (cluster_id);
