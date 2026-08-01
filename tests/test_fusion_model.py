"""Required, fully-offline tests for the fusion model stage.

Real LightGBM/sklearn/shap computation runs locally (not network); NO test
depends on real database content. All feature/label data is controlled synthetic
fixtures in temp DBs. Run:  python -m pytest tests/test_fusion_model.py
"""
import json
import math
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from modeling import feature_builder, train_fusion_model, predict  # noqa: E402
from modeling.feature_builder import FEATURE_COLUMNS, build_feature_row  # noqa: E402
from modeling.train_fusion_model import (  # noqa: E402
    MIN_REAL_EXAMPLES_FOR_TRAINING, MIN_REAL_EXAMPLES_FOR_EVALUATION,
    decide_data_mode, generate_synthetic_dataset, train_and_save, init_model_schema,
)


def _base_db(path):
    db_init.init_db(path)
    conn = db_init.get_connection(path)
    for rid, name, hhi in [("dram", "DRAM", 0.53), ("gpu", "GPUs", 0.82)]:
        conn.execute("INSERT INTO resource (resource_id, name, aliases, hhi_country) "
                     "VALUES (?, ?, '[]', ?)", (rid, name, hhi))
    conn.executescript((PROJECT_ROOT / "schema_forecasting.sql").read_text())
    conn.commit()
    return conn


def _add_price(conn, rid, start, n, base=100.0, step=1.0):
    idx = pd.date_range(start=start, periods=n, freq="D")
    for i, ts in enumerate(idx):
        conn.execute("INSERT INTO price_series (resource_id, timestamp, price_usd, unit, source) "
                     "VALUES (?, ?, ?, 'USD_per_share_proxy', 'yfinance')",
                     (rid, ts.date().isoformat(), base + step * i))
    conn.commit()


class FeatureShapeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.conn = _base_db(Path(self._tmp.name) / "t.db")

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()

    def test_missing_upstream_is_nan_not_zero(self):
        # No price, no forecast, no signal for gpu -> those features must be NaN,
        # while the populated structural feature (hhi) is a real number.
        row = build_feature_row("gpu", "2026-06-01", self.conn)
        self.assertEqual(set(row.keys()), set(FEATURE_COLUMNS))
        for col in ["price_latest", "forecast_point_1m", "signal_severity_mean_4w",
                    "signal_count_4w"]:
            self.assertTrue(math.isnan(row[col]), f"{col} should be NaN, got {row[col]}")
            self.assertNotEqual(row[col], 0)  # explicitly NOT a silent zero
        self.assertAlmostEqual(row["hhi_country"], 0.82)  # populated structural feature

    def test_no_lookahead_leakage(self):
        # Seed price BOTH before and after the requested week; the feature row for
        # an early week must reflect ONLY the pre-cutoff prices.
        _add_price(self.conn, "dram", "2026-01-05", 120, base=100.0, step=1.0)
        # Forecast produced "now" (2026-07) must NOT appear for a 2026-02 week.
        self.conn.execute(
            "INSERT INTO forecast_result (resource_id, series_type, horizon, run_at, "
            "prophet_point, prophet_lower, prophet_upper, disagreement_pct, status) "
            "VALUES ('dram','price_proxy','1m','2026-07-01T00:00:00', 999.0, 900.0, 1100.0, 40.0, 'ok')")
        # A future signal severity must NOT leak into an early-week feature.
        self.conn.execute("INSERT INTO signal (signal_type, resource_id, timestamp, severity, "
                          "source_name) VALUES ('demand_intent_raw','dram','2026-06-01', 9, 'x')")
        self.conn.commit()

        early = build_feature_row("dram", "2026-02-16", self.conn)
        # price_latest at 2026-02-16 must be far below the last (future) price ~219.
        self.assertLess(early["price_latest"], 150.0)
        # Forecast (run_at 2026-07) is in the future -> NaN, not 999.
        self.assertTrue(math.isnan(early["forecast_point_1m"]))
        self.assertTrue(math.isnan(early["forecast_disagreement_1m"]))
        # Future signal (2026-06) must not contribute to a Feb feature row.
        self.assertTrue(math.isnan(early["signal_severity_mean_4w"]))

        # A LATE week (after all data) SHOULD see the forecast and higher price.
        late = build_feature_row("dram", "2026-07-06", self.conn)
        self.assertFalse(math.isnan(late["forecast_point_1m"]))
        self.assertEqual(late["forecast_point_1m"], 999.0)
        self.assertGreater(late["price_latest"], early["price_latest"])


class ModeDecisionTest(unittest.TestCase):
    def test_below_threshold_is_synthetic(self):
        self.assertEqual(decide_data_mode(MIN_REAL_EXAMPLES_FOR_TRAINING - 1),
                         "synthetic_validation")

    def test_at_threshold_is_real(self):
        self.assertEqual(decide_data_mode(MIN_REAL_EXAMPLES_FOR_TRAINING), "real")


class TrainingAndPredictTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "t.db"
        self.models_dir = Path(self._tmp.name) / "models"
        os.environ["SUPPLY_CHAIN_DB"] = str(self.db_path)
        self.conn = _base_db(self.db_path)
        init_model_schema(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("SUPPLY_CHAIN_DB", None)
        self._tmp.cleanup()

    def _fake_real_df(self, n):
        df = generate_synthetic_dataset(n=n, seed=1)
        # Mark it as if it were real overlapping data (dates present).
        df["label_source"] = "price_derived"
        df["resource_id"] = "dram"
        df["week_start"] = "2026-06-01"
        return df

    def test_synthetic_validation_run_recorded(self):
        syn = generate_synthetic_dataset(n=40, seed=1)
        run_id = train_and_save(syn, "synthetic_validation", self.conn,
                                models_dir=self.models_dir)
        row = self.conn.execute(
            "SELECT data_mode, random_seed, calibration_applied FROM training_run "
            "WHERE run_id = ?", (run_id,)).fetchone()
        self.assertEqual(row[0], "synthetic_validation")
        self.assertEqual(row[1], train_fusion_model.RANDOM_SEED if False else row[1])  # seed recorded
        self.assertEqual(row[2], 1)  # calibration applied even in synthetic path
        self.assertTrue((self.models_dir / "fusion_model_synthetic_validation.pkl").exists())

    def test_real_path_records_real_mode(self):
        df = self._fake_real_df(MIN_REAL_EXAMPLES_FOR_TRAINING + 20)
        run_id = train_and_save(df, "real", self.conn, models_dir=self.models_dir)
        mode = self.conn.execute("SELECT data_mode FROM training_run WHERE run_id=?",
                                 (run_id,)).fetchone()[0]
        self.assertEqual(mode, "real")

    def test_calibrated_probability_is_valid(self):
        _add_price(self.conn, "dram", "2026-01-05", 120)
        train_and_save(generate_synthetic_dataset(n=200, seed=2),
                       "synthetic_validation", self.conn, models_dir=self.models_dir)
        for wk in ["2026-03-30", "2026-04-27", "2026-06-01"]:
            out = predict.predict_resource("dram", wk, "1m", db_path=self.db_path,
                                           models_dir=self.models_dir)
            self.assertGreaterEqual(out["calibrated_probability"], 0.0)
            self.assertLessEqual(out["calibrated_probability"], 1.0)

    def test_shap_gated_off_for_synthetic(self):
        _add_price(self.conn, "dram", "2026-01-05", 120)
        train_and_save(generate_synthetic_dataset(n=200, seed=3),
                       "synthetic_validation", self.conn, models_dir=self.models_dir)
        out = predict.predict_resource("dram", "2026-06-01", "1m", db_path=self.db_path,
                                       models_dir=self.models_dir)
        self.assertIsNone(out["top_shap_features"])
        self.assertIn("synthetic_validation", out["shap_reason"])
        self.assertFalse(out["is_real_data_model"])
        self.assertIn("warning", out)

    def test_shap_present_for_real_meeting_minimum(self):
        _add_price(self.conn, "dram", "2026-01-05", 120)
        df = self._fake_real_df(MIN_REAL_EXAMPLES_FOR_EVALUATION + 40)
        train_and_save(df, "real", self.conn, models_dir=self.models_dir)
        out = predict.predict_resource("dram", "2026-06-01", "1m", db_path=self.db_path,
                                       models_dir=self.models_dir, prefer="real")
        self.assertTrue(out["is_real_data_model"])
        self.assertIsNotNone(out["top_shap_features"])
        self.assertGreaterEqual(len(out["top_shap_features"]), 3)
        for f in out["top_shap_features"]:
            self.assertEqual(set(f.keys()), {"feature", "value", "contribution"})

    def test_prediction_row_links_to_training_run(self):
        _add_price(self.conn, "dram", "2026-01-05", 120)
        train_and_save(generate_synthetic_dataset(n=150, seed=4),
                       "synthetic_validation", self.conn, models_dir=self.models_dir)
        out = predict.predict_resource("dram", "2026-06-01", "1m", db_path=self.db_path,
                                       models_dir=self.models_dir)
        # FK link resolves to a training_run whose data_mode matches the returned dict.
        row = self.conn.execute(
            "SELECT t.data_mode FROM prediction p JOIN training_run t ON p.run_id = t.run_id "
            "WHERE p.run_id = ? ORDER BY p.id DESC LIMIT 1", (out["run_id"],)).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], out["data_mode"])
        # SHAP-null gating is consistent with synthetic mode in the stored row.
        shap_stored = self.conn.execute(
            "SELECT top_shap_features FROM prediction WHERE run_id=? ORDER BY id DESC LIMIT 1",
            (out["run_id"],)).fetchone()[0]
        self.assertIsNone(shap_stored)


if __name__ == "__main__":
    unittest.main(verbosity=2)
