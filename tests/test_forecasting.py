"""Required, offline tests for the forecasting stage.

Prophet/statsmodels ARE allowed to run (local computation, not network); we only
control the INPUT data and assert structural properties — never exact forecast
values. Run:  python -m pytest tests/test_forecasting.py
"""
import os
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from forecasting import series_builder, forecaster  # noqa: E402
from forecasting.forecaster import MIN_DATA_POINTS, HORIZONS, ForecastError  # noqa: E402
import scripts.run_forecasting as rf  # noqa: E402


def _base_db(path):
    db_init.init_db(path)
    conn = db_init.get_connection(path)
    for rid, name in [("dram", "DRAM"), ("hbm", "HBM"), ("nand", "NAND"), ("gpu", "GPUs")]:
        conn.execute("INSERT INTO resource (resource_id, name, aliases) VALUES (?, ?, '[]')",
                     (rid, name))
    conn.commit()
    return conn


def _seed_price(conn, resource_id, n_points, start=date(2026, 1, 5), step_days=7,
                base=100.0, slope=1.0, unit="USD_per_share_proxy"):
    """Seed n_points weekly-spaced proxy rows with a gentle linear trend."""
    for i in range(n_points):
        ts = (start + timedelta(days=i * step_days)).isoformat()
        conn.execute(
            "INSERT INTO price_series (resource_id, timestamp, price_usd, unit, source) "
            "VALUES (?, ?, ?, ?, 'yfinance')",
            (resource_id, ts, base + slope * i, unit))
    conn.commit()


class SeriesBuilderTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.conn = _base_db(Path(self._tmp.name) / "t.db")

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()

    def test_price_series_shape_and_scope(self):
        _seed_price(self.conn, "dram", 15)
        _seed_price(self.conn, "gpu", 15, base=500.0)            # other resource
        # a non-proxy-unit row for dram that must be excluded (macro-style unit).
        self.conn.execute(
            "INSERT INTO price_series (resource_id, timestamp, price_usd, unit, source) "
            "VALUES ('dram', '2026-01-05', 999.0, 'USD_index', 'world_bank_pink_sheet')")
        self.conn.commit()

        df = series_builder.build_price_series("dram", self.conn)
        self.assertEqual(list(df.columns), ["ds", "y"])
        self.assertGreaterEqual(len(df), 12)
        # The excluded macro-unit row (999.0) must not appear.
        self.assertNotIn(999.0, list(df["y"]))
        # gpu rows (base 500) must not leak into dram series.
        self.assertTrue(all(v < 500 for v in df["y"]))

    def test_demand_intent_severity_weighted_sum_and_scope(self):
        # Two demand rows same week (severities 3 & 5 -> weekly sum 8), one other
        # week (sev 4), plus a severity-NULL row and an other-resource row (excluded).
        rows = [
            ("dram", "demand_intent_raw", "2026-01-05", 3),
            ("dram", "demand_intent_raw", "2026-01-06", 5),
            ("dram", "demand_intent_raw", "2026-01-13", 4),
            ("dram", "demand_intent_raw", "2026-01-20", None),   # unscored -> excluded
            ("gpu", "demand_intent_raw", "2026-01-05", 9),       # other resource
            ("dram", "news_raw", "2026-01-05", 7),               # wrong signal_type
        ]
        for rid, st, ts, sev in rows:
            self.conn.execute(
                "INSERT INTO signal (signal_type, resource_id, timestamp, severity, source_name) "
                "VALUES (?, ?, ?, ?, 'x')", (st, rid, ts, sev))
        self.conn.commit()
        df = series_builder.build_demand_intent_series("dram", self.conn)
        self.assertEqual(list(df.columns), ["ds", "y"])
        # First week sum = 3+5 = 8; no 9 (gpu) and no 7 (news_raw).
        self.assertIn(8.0, list(df["y"]))
        self.assertNotIn(9.0, list(df["y"]))
        self.assertNotIn(7.0, list(df["y"]))


class ForecasterStructureTest(unittest.TestCase):
    def _clean_series(self, n=30):
        import pandas as pd
        idx = pd.date_range("2026-01-04", periods=n, freq="W")
        return pd.DataFrame({"ds": idx, "y": [100.0 + 2.0 * i for i in range(n)]})

    def test_prophet_bounds_ordered_at_every_horizon(self):
        out = forecaster.run_prophet(self._clean_series(30))
        self.assertEqual(set(out.keys()), set(HORIZONS))
        for key, vals in out.items():
            self.assertLessEqual(vals["lower"], vals["point"], f"{key} lower<=point")
            self.assertLessEqual(vals["point"], vals["upper"], f"{key} point<=upper")

    def test_holt_winters_returns_all_horizons(self):
        out = forecaster.run_holt_winters(self._clean_series(30))
        self.assertEqual(set(out.keys()), set(HORIZONS))
        for v in out.values():
            self.assertIsInstance(v, float)

    def test_disagreement_math(self):
        self.assertAlmostEqual(forecaster.compute_disagreement(100.0, 100.0), 0.0)
        self.assertAlmostEqual(forecaster.compute_disagreement(100.0, 150.0), 50.0/1.5, places=1)


class OrchestrationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "t.db"
        os.environ["SUPPLY_CHAIN_DB"] = str(self.db_path)
        self.conn = _base_db(self.db_path)
        # Keep these tests hermetic/fast: Chronos is OPTIONAL, so default it to
        # "unavailable" (raises -> caught, chronos_point stays NULL). Individual
        # tests below opt in to a stubbed Chronos to assert storage behavior.
        self._chronos_patch = mock.patch.object(
            forecaster, "run_chronos", side_effect=ForecastError("chronos disabled in test"))
        self._chronos_patch.start()
        self.addCleanup(self._chronos_patch.stop)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("SUPPLY_CHAIN_DB", None)
        self._tmp.cleanup()

    def test_insufficient_data_does_not_call_models(self):
        _seed_price(self.conn, "dram", 5)  # < MIN_DATA_POINTS weekly points
        with mock.patch.object(forecaster, "run_prophet") as mp, \
             mock.patch.object(forecaster, "run_holt_winters") as mh:
            rf.run(self.db_path)
        mp.assert_not_called()
        mh.assert_not_called()
        conn = db_init.get_connection(self.db_path)
        st = conn.execute("SELECT DISTINCT status FROM forecast_result WHERE resource_id='dram' "
                          "AND series_type='price_proxy'").fetchall()
        conn.close()
        self.assertEqual(st, [("insufficient_data",)])

    def test_ok_status_and_ordered_bounds(self):
        for rid in ["dram", "hbm", "nand", "gpu"]:
            _seed_price(self.conn, rid, 30, slope=1.0)
        rf.run(self.db_path)
        conn = db_init.get_connection(self.db_path)
        ok = conn.execute(
            "SELECT prophet_lower, prophet_point, prophet_upper FROM forecast_result "
            "WHERE status='ok'").fetchall()
        # macro/null guard + closed status set
        null_price = conn.execute(
            "SELECT COUNT(*) FROM forecast_result WHERE series_type='price_proxy' "
            "AND resource_id IS NULL").fetchone()[0]
        statuses = {r[0] for r in conn.execute("SELECT DISTINCT status FROM forecast_result")}
        conn.close()
        self.assertTrue(ok, "expected some ok rows")
        for lo, pt, up in ok:
            self.assertLessEqual(lo, pt)
            self.assertLessEqual(pt, up)
        self.assertEqual(null_price, 0)
        self.assertTrue(statuses.issubset(
            {"ok", "insufficient_data", "model_error", "unstable_disagreement"}))

    def test_forced_disagreement_sets_unstable(self):
        _seed_price(self.conn, "dram", 30)
        # Force a large divergence between the two models.
        prophet_ret = {k: {"point": 100.0, "lower": 90.0, "upper": 110.0} for k in HORIZONS}
        hw_ret = {k: 500.0 for k in HORIZONS}  # ~80% disagreement -> unstable
        with mock.patch.object(forecaster, "run_prophet", return_value=prophet_ret), \
             mock.patch.object(forecaster, "run_holt_winters", return_value=hw_ret):
            rf.run(self.db_path)
        conn = db_init.get_connection(self.db_path)
        statuses = {r[0] for r in conn.execute(
            "SELECT status FROM forecast_result WHERE resource_id='dram' "
            "AND series_type='price_proxy'")}
        conn.close()
        self.assertIn("unstable_disagreement", statuses)

    def test_model_error_recorded_and_loop_continues(self):
        for rid in ["dram", "hbm", "nand", "gpu"]:
            _seed_price(self.conn, rid, 30)
        # Make every Prophet call raise; every price series must record model_error
        # and the loop must proceed through all resources without crashing.
        with mock.patch.object(forecaster, "run_prophet", side_effect=ForecastError("boom")):
            rc = rf.run(self.db_path)
        self.assertEqual(rc, 0)  # loop completed without crashing
        conn = db_init.get_connection(self.db_path)
        errs = conn.execute("SELECT COUNT(*) FROM forecast_result WHERE status='model_error'").fetchone()[0]
        conn.close()
        self.assertGreaterEqual(errs, 4)  # all 4 price series errored, loop continued

    def test_idempotent_replace_latest(self):
        for rid in ["dram", "hbm", "nand", "gpu"]:
            _seed_price(self.conn, rid, 30)
        rf.run(self.db_path)
        conn = db_init.get_connection(self.db_path)
        first = conn.execute("SELECT COUNT(*) FROM forecast_result").fetchone()[0]
        first_runat = conn.execute("SELECT DISTINCT run_at FROM forecast_result").fetchall()
        conn.close()
        rf.run(self.db_path)  # replace-latest: same count, not doubled
        conn = db_init.get_connection(self.db_path)
        second = conn.execute("SELECT COUNT(*) FROM forecast_result").fetchone()[0]
        conn.close()
        self.assertEqual(first, second)  # replaced, not accumulated
        self.assertEqual(len(first_runat), 1)

    def test_chronos_point_stored_when_available(self):
        _seed_price(self.conn, "dram", 30, slope=1.0)
        stub = {k: 123.0 for k in HORIZONS}
        # Opt in to a stubbed Chronos (overriding the setUp default) and assert its
        # point is persisted per horizon while status stays Prophet-vs-HW driven.
        with mock.patch.object(forecaster, "run_chronos", return_value=stub):
            rf.run(self.db_path)
        conn = db_init.get_connection(self.db_path)
        pts = conn.execute(
            "SELECT chronos_point FROM forecast_result WHERE resource_id='dram' "
            "AND series_type='price_proxy' AND horizon IN ('2w','1m','3m','6m')").fetchall()
        conn.close()
        self.assertTrue(pts)
        self.assertTrue(all(p[0] == 123.0 for p in pts))

    def test_chronos_failure_is_non_fatal(self):
        # setUp already forces run_chronos to raise; the series must still succeed
        # with Prophet+HW and chronos_point NULL — Chronos is never required.
        _seed_price(self.conn, "dram", 30, slope=1.0)
        rc = rf.run(self.db_path)
        self.assertEqual(rc, 0)
        conn = db_init.get_connection(self.db_path)
        rows = conn.execute(
            "SELECT status, chronos_point FROM forecast_result WHERE resource_id='dram' "
            "AND series_type='price_proxy' AND horizon='1m'").fetchall()
        conn.close()
        self.assertTrue(rows)
        self.assertIn(rows[0][0], {"ok", "unstable_disagreement"})
        self.assertIsNone(rows[0][1])  # chronos_point NULL when Chronos unavailable


if __name__ == "__main__":
    unittest.main(verbosity=2)
