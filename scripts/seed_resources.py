"""Seed the SQLite database with the 4 MVP resources and 5 producers.

Reads data/resource_seed.json and data/alias_dictionary.json, (re)creates a
fresh schema via db_init, and inserts resource/producer/junction rows using
INSERT OR REPLACE on stable slug PKs so re-running is idempotent.

hhi_country is computed from each resource's country shares so it stays
internally consistent with top_producing_countries.
"""
import json
import sys
from pathlib import Path

# Make the project root importable so `import db_init` works when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
RESOURCE_SEED_PATH = DATA_DIR / "resource_seed.json"
ALIAS_DICT_PATH = DATA_DIR / "alias_dictionary.json"

# Which resources each producer makes (many-to-many junction).
PRODUCER_RESOURCES = {
    "samsung": ["dram", "hbm", "nand"],
    "sk_hynix": ["dram", "hbm", "nand"],
    "micron": ["dram", "hbm", "nand"],
    "nvidia": ["gpu"],
    "tsmc": ["gpu"],
}


def _hhi(countries: list[dict]) -> float:
    """Herfindahl-Hirschman index (0-1) over country shares."""
    return round(sum(float(c.get("share", 0)) ** 2 for c in countries), 4)


def seed(db_path: Path | None = None) -> Path:
    resource_seed = json.loads(RESOURCE_SEED_PATH.read_text(encoding="utf-8"))
    alias_dict = json.loads(ALIAS_DICT_PATH.read_text(encoding="utf-8"))

    # Fresh schema, then populate.
    path = db_init.init_db(db_path)
    conn = db_init.get_connection(db_path)
    try:
        for r in resource_seed["resources"]:
            countries = r.get("top_producing_countries", [])
            conn.execute(
                """
                INSERT OR REPLACE INTO resource (
                    resource_id, name, category, hs_codes, aliases, substitutes,
                    hhi_country, top_producing_countries, expansion_lead_time_months
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r["resource_id"],
                    r["name"],
                    r.get("category"),
                    json.dumps(r.get("hs_codes", [])),
                    json.dumps(r.get("aliases", [])),
                    json.dumps(r.get("substitutes", [])),
                    _hhi(countries),
                    json.dumps(countries),
                    r.get("expansion_lead_time_months"),
                ),
            )

        for p in alias_dict["producers"]:
            conn.execute(
                "INSERT OR REPLACE INTO producer (producer_id, name, tickers) VALUES (?, ?, ?)",
                (p["producer_id"], p["canonical_name"], json.dumps(p.get("tickers", []))),
            )

        for producer_id, resource_ids in PRODUCER_RESOURCES.items():
            for resource_id in resource_ids:
                conn.execute(
                    "INSERT OR REPLACE INTO producer_resource (producer_id, resource_id) VALUES (?, ?)",
                    (producer_id, resource_id),
                )

        conn.commit()
    finally:
        conn.close()
    return path


if __name__ == "__main__":
    seeded = seed()
    print(f"Seeded 4 resources and 5 producers into {seeded}")
