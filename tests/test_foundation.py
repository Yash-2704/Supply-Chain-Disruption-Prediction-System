"""Foundation tests for the supply-chain data layer (plain unittest, no deps).

Run: python tests/test_foundation.py   (or: pytest tests/)

The suite is hermetic: it points SUPPLY_CHAIN_DB at a temp file and runs
db_init.py + seed_resources.py as real subprocesses, so it exercises the
actual entry points without touching data/supply_chain.db.
"""
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_TABLES = {
    "resource", "producer", "producer_resource", "facility",
    "signal", "price_series", "capacity_record",
}
EXPECTED_RESOURCES = {"DRAM", "HBM", "NAND", "GPUs"}
EXPECTED_PRODUCERS = {"Samsung", "SK Hynix", "Micron", "Nvidia", "TSMC"}


class FoundationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls._tmp.name) / "supply_chain.db"
        env = {**os.environ, "SUPPLY_CHAIN_DB": str(cls.db_path)}
        # Run the real scripts back to back, as a clean checkout would.
        subprocess.run([sys.executable, "db_init.py"], cwd=PROJECT_ROOT,
                       env=env, check=True, capture_output=True)
        subprocess.run([sys.executable, "scripts/seed_resources.py"], cwd=PROJECT_ROOT,
                       env=env, check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def test_db_file_exists(self):
        self.assertTrue(self.db_path.exists(), "database file was not created")

    def test_all_seven_tables_exist(self):
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        finally:
            conn.close()
        self.assertEqual({r[0] for r in rows}, EXPECTED_TABLES)

    def test_table_columns(self):
        expected_cols = {
            "resource": {"resource_id", "name", "category", "hs_codes", "aliases",
                         "substitutes", "hhi_country", "top_producing_countries",
                         "expansion_lead_time_months"},
            "producer": {"producer_id", "name", "tickers"},
            "producer_resource": {"producer_id", "resource_id"},
            "facility": {"facility_id", "producer_id", "country", "city"},
            "signal": {"signal_id", "timestamp", "signal_type", "resource_id",
                       "producer_id", "country", "value", "unit", "severity",
                       "confidence", "source_name", "source_url", "raw_text",
                       "ingested_at"},
            "price_series": {"id", "resource_id", "timestamp", "price_usd", "unit", "source"},
            "capacity_record": {"id", "producer_id", "resource_id", "year",
                                "capacity_value", "unit", "source"},
        }
        conn = self._conn()
        try:
            for table, cols in expected_cols.items():
                info = conn.execute(f"PRAGMA table_info({table})").fetchall()
                self.assertEqual({row[1] for row in info}, cols, f"columns mismatch in {table}")
        finally:
            conn.close()

    def test_exactly_four_resources(self):
        conn = self._conn()
        try:
            names = [r[0] for r in conn.execute("SELECT name FROM resource").fetchall()]
        finally:
            conn.close()
        self.assertEqual(len(names), 4)
        self.assertEqual(set(names), EXPECTED_RESOURCES)

    def test_exactly_five_producers(self):
        conn = self._conn()
        try:
            names = [r[0] for r in conn.execute("SELECT name FROM producer").fetchall()]
        finally:
            conn.close()
        self.assertEqual(len(names), 5)
        self.assertEqual(set(names), EXPECTED_PRODUCERS)

    def test_every_producer_has_a_resource(self):
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT p.producer_id, COUNT(pr.resource_id) "
                "FROM producer p LEFT JOIN producer_resource pr "
                "ON p.producer_id = pr.producer_id GROUP BY p.producer_id"
            ).fetchall()
        finally:
            conn.close()
        self.assertEqual(len(rows), 5)
        for producer_id, count in rows:
            self.assertGreaterEqual(count, 1, f"{producer_id} has no resources")

    def test_foreign_key_enforced(self):
        conn = self._conn()
        try:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO producer_resource (producer_id, resource_id) VALUES (?, ?)",
                    ("samsung", "does_not_exist"),
                )
                conn.commit()
        finally:
            conn.close()

    def test_resource_json_fields_valid_and_nonempty(self):
        conn = self._conn()
        try:
            rows = conn.execute("SELECT name, aliases, substitutes FROM resource").fetchall()
        finally:
            conn.close()
        for name, aliases, substitutes in rows:
            parsed_aliases = json.loads(aliases)
            parsed_subs = json.loads(substitutes)
            self.assertTrue(parsed_aliases, f"{name} has empty aliases")
            self.assertTrue(parsed_subs, f"{name} has empty substitutes")

    def test_alias_dictionary_valid_with_four_plus_aliases(self):
        alias_path = PROJECT_ROOT / "data" / "alias_dictionary.json"
        data = json.loads(alias_path.read_text(encoding="utf-8"))
        producers = data["producers"]
        self.assertEqual(len(producers), 5)
        for p in producers:
            self.assertGreaterEqual(
                len(p["aliases"]), 4,
                f"{p['producer_id']} has fewer than 4 aliases",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
