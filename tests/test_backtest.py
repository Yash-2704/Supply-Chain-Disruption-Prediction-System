"""Required, fully-offline tests for the backtest harness.

Centerpiece: an ADVERSARIAL lookahead test that proves post-cutoff rows never
influence the as-of result. Controlled synthetic fixtures with known timestamps;
no dependency on real historical data. Run:  python -m pytest tests/test_backtest.py
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from backtest import cutoff as co  # noqa: E402
from backtest import fixtures as fx  # noqa: E402
from backtest import run_backtest as rb  # noqa: E402


def _fresh_db(path):
    db_init.init_db(path)
    conn = db_init.get_connection(path)
    conn.execute("INSERT INTO resource (resource_id, name, aliases) VALUES ('dram','DRAM','[]')")
    conn.commit()
    return conn


def _add_price(conn, rid, iso_date, price):
    conn.execute("INSERT INTO price_series (resource_id, timestamp, price_usd, unit, source) "
                 "VALUES (?, ?, ?, 'USD_per_share_proxy', 'synthetic_fixture')",
                 (rid, iso_date, price))


def _add_signal(conn, rid, ts, severity):
    conn.execute("INSERT INTO signal (signal_type, resource_id, timestamp, severity, source_name) "
                 "VALUES ('news_raw', ?, ?, ?, 'synthetic_fixture')", (rid, ts, severity))


class TimestampParsingTest(unittest.TestCase):
    def test_handles_all_project_formats(self):
        self.assertIsNotNone(co.parse_timestamp("20240630T114500Z"))  # GDELT compact
        self.assertIsNotNone(co.parse_timestamp("2024-06-30"))         # ISO date
        self.assertIsNotNone(co.parse_timestamp("2024-06-30T11:45:00"))  # ISO datetime
        self.assertIsNone(co.parse_timestamp(None))
        self.assertIsNone(co.parse_timestamp("not-a-date"))

    def test_cutoff_comparison_and_unparseable_excluded(self):
        self.assertTrue(co.is_at_or_before("2024-04-30", "2024-04-30"))   # inclusive
        self.assertTrue(co.is_at_or_before("2024-04-29", "2024-04-30"))
        self.assertFalse(co.is_at_or_before("2024-05-01", "2024-04-30"))
        self.assertFalse(co.is_at_or_before("garbage", "2024-04-30"))     # excluded, never leaked


class AdversarialLookaheadTest(unittest.TestCase):
    """THE most important test: prove future rows cannot influence the as-of result."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.past_path = Path(self._tmp.name) / "past_only.db"
        self.mixed_path = Path(self._tmp.name) / "with_future.db"
        self.cutoff = "2024-04-30"

    def tearDown(self):
        self._tmp.cleanup()

    def _seed_common_past(self, conn):
        # 22 weekly pre-cutoff points forming a MILD trend, + some scored signals.
        # Deterministic weekly dates from 2023-11-06 stepping 7 days, all <= cutoff.
        import datetime as _dt
        base = 100.0
        start = _dt.date(2023, 11, 6)
        for wk in range(22):
            d = (start + _dt.timedelta(weeks=wk)).isoformat()  # up to ~2024-04-07, <= cutoff
            _add_price(conn, "dram", d, base + wk * 1.0)
        _add_signal(conn, "dram", "2024-01-15", 3)
        _add_signal(conn, "dram", "2024-03-15", 5)
        conn.commit()

    def test_future_rows_do_not_influence_asof_features(self):
        # DB #1: only pre-cutoff data.
        conn_past = _fresh_db(self.past_path)
        self._seed_common_past(conn_past)

        # DB #2: identical pre-cutoff data PLUS an extreme post-cutoff spike + future signals.
        conn_mixed = _fresh_db(self.mixed_path)
        self._seed_common_past(conn_mixed)
        for d, p in [("2024-05-05", 9999.0), ("2024-06-05", 12345.0), ("2025-01-05", 50000.0)]:
            _add_price(conn_mixed, "dram", d, p)          # huge future spikes
        _add_signal(conn_mixed, "dram", "2024-05-10", 10)  # future severity
        _add_signal(conn_mixed, "dram", "2025-02-01", 10)
        conn_mixed.commit()

        feats_past = co.compute_asof_features(conn_past, "dram", self.cutoff)
        feats_mixed = co.compute_asof_features(conn_mixed, "dram", self.cutoff)

        # (1) STRONGEST proof: presence/absence of future rows yields IDENTICAL as-of features.
        self.assertEqual(feats_past, feats_mixed,
                         "future rows leaked into the as-of feature computation!")

        # (2) No filtered price is from after the cutoff, and the huge spike never appears.
        prices_mixed = co.prices_asof(conn_mixed, "dram", self.cutoff)
        self.assertTrue(all(co.is_at_or_before(ts, self.cutoff) for ts, _ in prices_mixed))
        self.assertTrue(all(p < 9999.0 for _, p in prices_mixed),
                        "a post-cutoff spike price leaked into the as-of set")

        # (3) latest as-of price reflects the last PRE-cutoff week, not the future spike.
        self.assertLess(feats_mixed["latest_price"], 9999.0)

        # (4) future signals excluded: scored-signal count matches the past-only DB.
        self.assertEqual(feats_mixed["n_scored_signals"], feats_past["n_scored_signals"])

        conn_past.close()
        conn_mixed.close()


class FixtureReproducibilityTest(unittest.TestCase):
    def test_same_seed_same_data_and_spans_window(self):
        a = fx.generate_synthetic_history("dram", seed=123)
        b = fx.generate_synthetic_history("dram", seed=123)
        self.assertEqual(a, b)  # reproducible
        # genuinely spans the documented historical window
        price_dates = [d for d, _ in a["prices"]]
        self.assertTrue(min(price_dates) < "2023-12-01")
        self.assertTrue(max(price_dates) > "2025-01-01")
        self.assertGreater(len(a["prices"]), 100)

    def test_different_seed_differs(self):
        a = fx.generate_synthetic_history("dram", seed=1)
        b = fx.generate_synthetic_history("dram", seed=2)
        self.assertNotEqual(a["prices"], b["prices"])


class IsolationTest(unittest.TestCase):
    def test_backtest_leaves_real_db_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            real_path = Path(d) / "real.db"
            bt_path = Path(d) / "bt.db"
            # stand-in "real" DB with some rows.
            conn = _fresh_db(real_path)
            _add_price(conn, "dram", "2026-06-01", 500.0)
            conn.commit()
            conn.close()

            before = rb.real_db_row_counts(real_path)
            rb.run_backtest("dram", "2024-04-30", "synthetic_fixture",
                            backtest_db_path=bt_path, real_db_path=real_path)
            after = rb.real_db_row_counts(real_path)
            self.assertEqual(before, after)  # real DB byte-for-byte row counts unchanged


class OutcomeComparisonTest(unittest.TestCase):
    def test_tightening_signal_flagged(self):
        episode = {"resource_id": "dram", "justification": "x"}
        feats = {"anomaly_label_latest": 1, "price_zscore_latest": 3.0}
        out = rb.compare_to_expected_outcome(feats, "rising", episode)
        self.assertEqual(out["expected_outcome"], "tightening")
        self.assertTrue(out["direction_match"])

    def test_no_signal_not_forced(self):
        episode = {"resource_id": "dram"}
        feats = {"anomaly_label_latest": 0, "price_zscore_latest": 0.1}
        out = rb.compare_to_expected_outcome(feats, "not_rising", episode)
        self.assertFalse(out["direction_match"])
        self.assertEqual(out["observed_signal"], "no_tightening_signal")


class HarnessOutputHonestyTest(unittest.TestCase):
    def test_excluded_stages_reported_not_silently_skipped(self):
        # The harness must name every excluded stage with a reason.
        names = [n for n, _ in rb.STAGES_EXCLUDED]
        self.assertIn("fusion model train/predict",
                      [n for n in names])
        self.assertTrue(all(reason for _, reason in rb.STAGES_EXCLUDED))

    def test_synthetic_summary_unmistakably_qualified(self):
        with tempfile.TemporaryDirectory() as d:
            real_path = Path(d) / "real.db"
            _fresh_db(real_path).close()
            bt_path = Path(d) / "bt.db"
            summary = rb.run_backtest("dram", "2024-04-30", "synthetic_fixture",
                                      backtest_db_path=bt_path, real_db_path=real_path)
        self.assertEqual(summary["data_source"], "synthetic_fixture")
        # provenance must be attached to the headline result, not inferable-only.
        self.assertIn("comparison", summary)
        self.assertTrue(summary["real_db_unchanged"])

    def test_real_historical_aborts_rather_than_fabricates(self):
        with tempfile.TemporaryDirectory() as d:
            real_path = Path(d) / "real.db"
            _fresh_db(real_path).close()
            summary = rb.run_backtest("dram", "2024-04-30", "real_historical",
                                      backtest_db_path=Path(d) / "bt.db", real_db_path=real_path)
        self.assertEqual(summary["status"], "aborted_no_real_historical_data")


if __name__ == "__main__":
    unittest.main(verbosity=2)
