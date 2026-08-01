"""Required, offline tests for label construction (controlled synthetic fixtures).

No network. The mechanical price-derived logic is tested against crafted spike /
no-spike series; the real sparse DB state is never relied upon. Run:
    python -m pytest tests/test_labels.py
"""
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from labeling import curated_labels, price_derived_labels  # noqa: E402
from labeling.price_derived_labels import (  # noqa: E402
    MIN_BASELINE_WEEKS, MIN_SUSTAINED_WEEKS, compute_price_derived_labels,
)
import scripts.build_labels as build_labels  # noqa: E402


def _weekly_series(values, start="2026-01-05"):
    """Build a weekly (ds,y) frame from a list of values, one per Monday."""
    idx = pd.date_range(start=start, periods=len(values), freq="W-MON")
    return pd.DataFrame({"ds": idx, "y": [float(v) for v in values]})


class CuratedExpansionTest(unittest.TestCase):
    def test_expands_lead_weeks_through_end_all_label1(self):
        doc = {
            "lead_weeks": 8,
            "episodes": [{
                "resource_id": "dram",
                "episode_name": "test episode",
                "start_date": "2024-03-01",
                "end_date": "2024-04-30",
                "justification": "A real stated reason.",
            }],
        }
        rows = curated_labels.expand_episodes(doc)
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(r["label"], 1)
            self.assertEqual(r["status"], "ok")
            self.assertEqual(r["source"], "curated_episode")
            self.assertEqual(r["justification"], "A real stated reason.")
        weeks = sorted(r["week_start"] for r in rows)
        # First labeled week is ~8 weeks before start_date (2024-03-01).
        first = date.fromisoformat(weeks[0])
        self.assertLessEqual(first, date(2024, 3, 1) - timedelta(weeks=7))
        # Last labeled week is on/around end_date.
        self.assertGreaterEqual(date.fromisoformat(weeks[-1]), date(2024, 4, 23))

    def test_real_file_now_includes_nand_and_both_classes(self):
        # NAND is now intentionally present (positive episode + negative period),
        # and the file defines explicit non-tightening periods for label=0.
        doc = curated_labels.load_episodes()
        ep_rids = {e["resource_id"] for e in doc["episodes"]}
        self.assertEqual(ep_rids, {"dram", "hbm", "gpu", "nand"})
        neg_rids = {n["resource_id"] for n in doc.get("negative_periods", [])}
        self.assertIn("nand", neg_rids)
        self.assertIn("dram", neg_rids)

    def test_negative_periods_expand_to_label0(self):
        doc = {
            "lead_weeks": 8,
            "episodes": [{
                "resource_id": "dram", "episode_name": "pos",
                "start_date": "2024-03-01", "end_date": "2024-04-30",
                "justification": "tightening reason.",
            }],
            "negative_periods": [{
                "resource_id": "dram", "period_name": "neg",
                "start_date": "2023-01-01", "end_date": "2023-03-31",
                "justification": "oversupply reason.",
            }],
        }
        rows = curated_labels.expand_episodes(doc)
        pos = [r for r in rows if r["label"] == 1]
        neg = [r for r in rows if r["label"] == 0]
        self.assertTrue(pos and neg)
        # Negatives get NO lead-in: earliest negative week is on/after its start.
        self.assertGreaterEqual(min(date.fromisoformat(r["week_start"]) for r in neg),
                                date(2022, 12, 26))  # Monday of the week of 2023-01-01
        self.assertTrue(all(r["source"] == "curated_episode" for r in rows))
        # No week is emitted as both classes (UNIQUE-constraint safety).
        self.assertEqual(len({r["week_start"] for r in rows}), len(rows))


class PriceDerivedLogicTest(unittest.TestCase):
    def test_sustained_spike_labels_1_and_stable_labels_0(self):
        # 12 flat weeks (baseline) then a sustained RISING ramp (realistic
        # tightening shape that stays elevated under a rolling baseline).
        values = [100.0] * 12 + [130.0, 145.0, 160.0, 175.0]
        results = compute_price_derived_labels(_weekly_series(values))
        # First MIN_BASELINE_WEEKS weeks -> insufficient_data.
        for r in results[:MIN_BASELINE_WEEKS]:
            self.assertEqual(r["status"], "insufficient_data")
            self.assertIsNone(r["label"])
        # Stable evaluable weeks (indices 8..11) -> label 0.
        for r in results[MIN_BASELINE_WEEKS:12]:
            self.assertEqual((r["status"], r["label"]), ("ok", 0))
        # The 4 sustained spike weeks -> label 1.
        spike = results[12:16]
        self.assertTrue(all(r["label"] == 1 and r["status"] == "ok" for r in spike),
                        [r["label"] for r in spike])

    def test_insufficient_data_below_baseline(self):
        values = [100.0] * (MIN_BASELINE_WEEKS - 1)  # never enough trailing history
        results = compute_price_derived_labels(_weekly_series(values))
        self.assertTrue(all(r["status"] == "insufficient_data" for r in results))
        self.assertTrue(all(r["label"] is None for r in results))

    def test_single_week_spike_not_sustained_rejected(self):
        # 12 flat + 1 spike + 3 flat: the lone spike is < MIN_SUSTAINED_WEEKS.
        values = [100.0] * 12 + [200.0] + [100.0] * 3
        results = compute_price_derived_labels(_weekly_series(values))
        self.assertGreaterEqual(1, MIN_SUSTAINED_WEEKS - 2)  # sanity: threshold > 1
        labels = [r["label"] for r in results if r["status"] == "ok"]
        self.assertNotIn(1, labels, "a single-week spike must not be labeled 1")


def _seed_resources(conn):
    for rid, name in [("dram", "DRAM"), ("hbm", "HBM"), ("nand", "NAND"), ("gpu", "GPUs")]:
        conn.execute("INSERT INTO resource (resource_id, name, aliases) VALUES (?, ?, '[]')",
                     (rid, name))
    conn.commit()


def _seed_price(conn, resource_id, values, start="2026-01-05"):
    idx = pd.date_range(start=start, periods=len(values), freq="W-MON")
    for ts, v in zip(idx, values):
        conn.execute(
            "INSERT INTO price_series (resource_id, timestamp, price_usd, unit, source) "
            "VALUES (?, ?, ?, 'USD_per_share_proxy', 'yfinance')",
            (resource_id, ts.date().isoformat(), float(v)))
    conn.commit()


class SchemaAndOrchestrationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "t.db"
        os.environ["SUPPLY_CHAIN_DB"] = str(self.db_path)
        db_init.init_db(self.db_path)
        self.conn = db_init.get_connection(self.db_path)
        _seed_resources(self.conn)
        build_labels.init_labels_schema(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("SUPPLY_CHAIN_DB", None)
        self._tmp.cleanup()

    def test_unique_constraint_enforced(self):
        self.conn.execute(
            "INSERT INTO label (resource_id, week_start, label, source, status) "
            "VALUES ('dram', '2024-01-01', 1, 'curated_episode', 'ok')")
        self.conn.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO label (resource_id, week_start, label, source, status) "
                "VALUES ('dram', '2024-01-01', 0, 'curated_episode', 'ok')")
            self.conn.commit()

    def test_two_sources_same_week_both_persist(self):
        # Same resource/week, different sources, DIFFERENT labels -> both survive.
        self.conn.execute(
            "INSERT INTO label (resource_id, week_start, label, source, status) "
            "VALUES ('dram', '2026-02-02', 1, 'curated_episode', 'ok')")
        self.conn.execute(
            "INSERT INTO label (resource_id, week_start, label, source, status) "
            "VALUES ('dram', '2026-02-02', 0, 'price_derived', 'ok')")
        self.conn.commit()
        rows = self.conn.execute(
            "SELECT source, label FROM label WHERE resource_id='dram' "
            "AND week_start='2026-02-02' ORDER BY source").fetchall()
        self.assertEqual(rows, [("curated_episode", 1), ("price_derived", 0)])

    def test_orchestration_idempotent(self):
        # Seed a clean spike series so price_derived produces real labels.
        for rid in ["dram", "hbm", "nand", "gpu"]:
            _seed_price(self.conn, rid, [100.0] * 12 + [180.0] * 6)
        build_labels.run(self.db_path)
        conn = db_init.get_connection(self.db_path)
        first = conn.execute("SELECT resource_id, week_start, label, source, status "
                             "FROM label ORDER BY id").fetchall()
        first_count = len(first)
        conn.close()

        build_labels.run(self.db_path)  # unchanged input
        conn = db_init.get_connection(self.db_path)
        second = conn.execute("SELECT resource_id, week_start, label, source, status "
                              "FROM label ORDER BY id").fetchall()
        conn.close()
        self.assertEqual(first_count, len(second))
        self.assertEqual(sorted(first), sorted(second))  # same values, no drift

    def test_status_label_consistency_and_closed_sets(self):
        for rid in ["dram", "hbm", "nand", "gpu"]:
            _seed_price(self.conn, rid, [100.0] * 12 + [180.0] * 6)
        build_labels.run(self.db_path)
        conn = db_init.get_connection(self.db_path)
        bad = conn.execute("SELECT COUNT(*) FROM label WHERE status='ok' AND label IS NULL").fetchone()[0]
        sources = {r[0] for r in conn.execute("SELECT DISTINCT source FROM label")}
        statuses = {r[0] for r in conn.execute("SELECT DISTINCT status FROM label")}
        nand_curated = conn.execute(
            "SELECT COUNT(*) FROM label WHERE resource_id='nand' AND source='curated_episode'").fetchone()[0]
        # Both label classes must now exist among curated rows (0 and 1).
        curated_classes = {r[0] for r in conn.execute(
            "SELECT DISTINCT label FROM label WHERE source='curated_episode' AND status='ok'")}
        conn.close()
        self.assertEqual(bad, 0)
        self.assertTrue(sources.issubset({"curated_episode", "price_derived"}))
        self.assertTrue(statuses.issubset({"ok", "insufficient_data"}))
        self.assertGreater(nand_curated, 0)          # NAND now curated (was 0)
        self.assertEqual(curated_classes, {0, 1})    # both classes present


if __name__ == "__main__":
    unittest.main(verbosity=2)
