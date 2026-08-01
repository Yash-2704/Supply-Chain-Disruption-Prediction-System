"""Initialize a fresh SQLite database from schema.sql.

Idempotent: drops existing tables (FK-safe order) and recreates them, so
running this twice never errors or duplicates schema. Paths are resolved
relative to this file's project root — no absolute paths are hardcoded.

Override the DB location with the SUPPLY_CHAIN_DB env var (used by tests).
"""
import os
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = PROJECT_ROOT / "schema.sql"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "supply_chain.db"


def _load_local_env(env_file: Path = PROJECT_ROOT / ".env.local") -> None:
    """Populate os.environ from a local KEY=VALUE secrets file (if present).
    Only sets a variable that isn't already in the environment, so a real
    environment variable always wins. No dependency on python-dotenv."""
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# Auto-load local secrets when the shared DB module is imported (every script
# imports db_init), so API keys are available without manual `export`s.
_load_local_env()

# Drop order respects FK dependencies (children before parents).
_TABLES_IN_DROP_ORDER = [
    "capacity_record",
    "price_series",
    "signal",
    "facility",
    "producer_resource",
    "producer",
    "resource",
]


def get_db_path() -> Path:
    """Resolve the database path, honoring the SUPPLY_CHAIN_DB override."""
    override = os.environ.get("SUPPLY_CHAIN_DB")
    return Path(override) if override else DEFAULT_DB_PATH


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Return a connection with foreign key enforcement enabled."""
    path = Path(db_path) if db_path else get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON;")
    # Wait (rather than immediately erroring "database is locked") when another
    # writer holds the lock — several ingest scripts may run concurrently, and
    # each commits sub-second, so a 30s wait is ample and avoids spurious failures.
    conn.execute("PRAGMA busy_timeout = 30000;")
    return conn


def init_db(db_path: Path | None = None) -> Path:
    """Create a fresh database from schema.sql. Returns the DB path."""
    path = Path(db_path) if db_path else get_db_path()
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    conn = get_connection(path)
    try:
        for table in _TABLES_IN_DROP_ORDER:
            conn.execute(f"DROP TABLE IF EXISTS {table};")
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()
    return path


if __name__ == "__main__":
    created = init_db()
    print(f"Initialized database at {created}")
