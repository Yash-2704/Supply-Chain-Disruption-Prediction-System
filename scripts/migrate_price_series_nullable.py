"""Migration: make price_series.resource_id NULLABLE (drop NOT NULL).

SQLite cannot drop a column constraint in place, so this uses the standard
table-rebuild procedure: create a new table with a nullable resource_id, copy
all rows, drop the old table, rename. The FK to resource(resource_id) is kept
(it applies only to non-NULL values). Idempotent: a no-op if already nullable.

Run:  python scripts/migrate_price_series_nullable.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402


def _is_nullable(conn) -> bool:
    for _cid, name, _t, notnull, _d, _pk in conn.execute("PRAGMA table_info(price_series)"):
        if name == "resource_id":
            return notnull == 0
    return False


def migrate(db_path=None) -> str:
    conn = db_init.get_connection(db_path)
    try:
        if _is_nullable(conn):
            return "already_nullable"
        before = conn.execute("SELECT COUNT(*) FROM price_series").fetchone()[0]
        conn.execute("PRAGMA foreign_keys = OFF;")
        conn.executescript(
            """
            BEGIN;
            CREATE TABLE price_series_new (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                resource_id  TEXT,                 -- now NULLABLE (macro rows use NULL)
                timestamp    TEXT,
                price_usd    REAL,
                unit         TEXT,
                source       TEXT,
                FOREIGN KEY (resource_id) REFERENCES resource (resource_id) ON DELETE CASCADE
            );
            INSERT INTO price_series_new (id, resource_id, timestamp, price_usd, unit, source)
                SELECT id, resource_id, timestamp, price_usd, unit, source FROM price_series;
            DROP TABLE price_series;
            ALTER TABLE price_series_new RENAME TO price_series;
            COMMIT;
            """
        )
        conn.execute("PRAGMA foreign_keys = ON;")
        after = conn.execute("SELECT COUNT(*) FROM price_series").fetchone()[0]
        assert before == after, f"row count changed during migration: {before} -> {after}"
        conn.commit()
        return f"migrated ({after} rows preserved)"
    finally:
        conn.close()


if __name__ == "__main__":
    result = migrate()
    print(f"price_series.resource_id migration: {result}")
