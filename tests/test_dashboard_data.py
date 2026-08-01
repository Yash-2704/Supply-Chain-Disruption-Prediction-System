"""Required, fully-offline tests for the dashboard data-access contract.

Seeded temp DB only — no dependency on real sparse data. Deliberately imports NO
streamlit/plotly/UI library (grep-verifiable) to prove the layer is UI-free.
Run:  python -m pytest tests/test_dashboard_data.py
"""
import dataclasses
import json
import os
import sys
import unittest
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from dashboard_data import queries  # noqa: E402
from dashboard_data.contracts import (  # noqa: E402
    EXPLANATION_NOT_ATTEMPTED, PredictionSummary, ResourceSummary, ResourceDetail, AlertFeedItem,
)


def _seed(path):
    """Rich resource 'dram' (prediction+forecast+explanation+episodes) and empty 'nand'."""
    db_init.init_db(path)
    conn = db_init.get_connection(path)
    for f in ["schema_forecasting.sql", "schema_labels.sql", "schema_model.sql",
              "schema_explanations.sql"]:
        conn.executescript((PROJECT_ROOT / f).read_text())
    for rid, name in [("dram", "DRAM"), ("nand", "NAND")]:
        conn.execute("INSERT INTO resource (resource_id, name, aliases) VALUES (?, ?, '[]')",
                     (rid, name))

    # --- rich dram ---
    conn.execute("INSERT INTO training_run (run_id, run_at, data_mode, n_examples, random_seed, "
                 "model_path, calibration_applied) VALUES (7, 't', 'synthetic_validation', 300, 42, 'm.pkl', 1)")
    conn.execute("INSERT INTO prediction (id, run_id, resource_id, week_start, horizon, raw_score, "
                 "calibrated_probability, top_shap_features, predicted_at) "
                 "VALUES (1, 7, 'dram', 'w', '1m', 0.31, 0.44, NULL, '2026-07-01T10:00:00')")
    # forecast: price_proxy has 3 ok + 1 unstable across horizons -> collapses to unstable.
    for hz, st, pt, lo, hi, hw, dg in [
        ("2w", "ok", 100.0, 90.0, 110.0, 98.0, 5.0),
        ("1m", "ok", 102.0, 92.0, 112.0, 101.0, 6.0),
        ("3m", "ok", 105.0, 95.0, 115.0, 104.0, 9.0),
        ("6m", "unstable_disagreement", 110.0, 90.0, 130.0, 60.0, 45.0)]:
        conn.execute("INSERT INTO forecast_result (resource_id, series_type, horizon, run_at, "
                     "prophet_point, prophet_lower, prophet_upper, holt_winters_point, "
                     "disagreement_pct, status) VALUES ('dram','price_proxy',?,?,?,?,?,?,?,?)",
                     (hz, "2026-07-01", pt, lo, hi, hw, dg, st))
    # a HW-only row to test point_value fallback (prophet_point NULL).
    conn.execute("INSERT INTO forecast_result (resource_id, series_type, horizon, run_at, "
                 "prophet_point, prophet_lower, prophet_upper, holt_winters_point, "
                 "disagreement_pct, status) VALUES "
                 "('dram','demand_intent_activity','all','2026-07-01', NULL, NULL, NULL, 55.5, NULL, 'ok')")
    conn.execute("INSERT INTO explanation (resource_id, prediction_id, narrative, citations, "
                 "evidence_count, llm_provider, is_real_data_model, status, generated_at) "
                 "VALUES ('dram', 1, '⚠️ NOTE disclaimer... DRAM tightening.', "
                 "'[{\"signal_id\": 4, \"excerpt\": \"DRAM sued\"}]', 4, 'gemini', 0, 'ok', "
                 "'2026-07-02T12:00:00')")
    conn.execute("INSERT INTO price_series (resource_id, timestamp, price_usd, unit, source) "
                 "VALUES ('dram','2026-06-01', 123.45, 'USD_per_share_proxy', 'yfinance')")
    # curated episode label rows for dram (episode_name enriched from JSON if matched).
    for wk in ["2024-01-01", "2024-02-05", "2024-03-04"]:
        conn.execute("INSERT INTO label (resource_id, week_start, label, source, status, "
                     "justification) VALUES ('dram', ?, 1, 'curated_episode', 'ok', "
                     "'A real DRAM tightening reason.')", (wk,))
    conn.commit()
    conn.close()


class DashboardDataTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "t.db"
        _seed(self.db_path)
        self.conn = db_init.get_connection(self.db_path)

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()

    def test_rich_resource_summary_sourced_correctly(self):
        s = queries.get_resource_summary("dram", conn=self.conn)
        self.assertIsInstance(s, ResourceSummary)
        self.assertEqual(s.name, "DRAM")
        # prediction spot-checks
        self.assertIsNotNone(s.latest_prediction)
        self.assertEqual(s.latest_prediction.run_id, 7)
        self.assertEqual(s.latest_prediction.data_mode, "synthetic_validation")
        self.assertFalse(s.latest_prediction.is_real_data_model)
        self.assertAlmostEqual(s.latest_prediction.calibrated_probability, 0.44)
        self.assertIsNone(s.latest_prediction.top_shap_features)  # NULL -> None, not []
        # forecast collapse: price_proxy has an unstable horizon -> unstable overall.
        self.assertEqual(s.latest_forecast_status_by_series["price_proxy"], "unstable_disagreement")
        self.assertEqual(s.latest_forecast_status_by_series["demand_intent_activity"], "ok")
        self.assertEqual(s.latest_explanation_status, "ok")

    def test_empty_resource_summary_all_honest_none(self):
        s = queries.get_resource_summary("nand", conn=self.conn)
        self.assertEqual(s.resource_id, "nand")
        self.assertEqual(s.name, "NAND")
        self.assertIsNone(s.latest_prediction)                 # no prediction -> None, not a fake 0.0
        self.assertEqual(s.latest_forecast_status_by_series, {})  # no forecasts -> empty dict
        self.assertEqual(s.latest_explanation_status, EXPLANATION_NOT_ATTEMPTED)  # sentinel, not 'insufficient_evidence'

    def test_leaderboard_one_row_per_resource(self):
        board = queries.get_leaderboard(conn=self.conn)
        ids = [r.resource_id for r in board]
        self.assertEqual(sorted(ids), ["dram", "nand"])  # both present incl. the empty one
        self.assertEqual(len(board), len(queries.get_all_resources(conn=self.conn)))

    def test_point_value_prefers_prophet_then_hw(self):
        detail = queries.get_resource_detail("dram", conn=self.conn)
        by = {(f.series_type, f.horizon): f for f in detail.forecast_breakdown}
        # prophet present -> prophet_point surfaced with its interval.
        pp = by[("price_proxy", "2w")]
        self.assertEqual(pp.point_value, 100.0)
        self.assertEqual((pp.lower_bound, pp.upper_bound), (90.0, 110.0))
        # prophet NULL, HW present -> HW point, NO interval.
        hw = by[("demand_intent_activity", "all")]
        self.assertEqual(hw.point_value, 55.5)
        self.assertIsNone(hw.lower_bound)
        self.assertIsNone(hw.upper_bound)

    def test_detail_historical_context_present_and_absent(self):
        dram = queries.get_resource_detail("dram", conn=self.conn)
        self.assertEqual(len(dram.historical_context), 1)
        ep = dram.historical_context[0]
        self.assertEqual(ep.justification, "A real DRAM tightening reason.")
        self.assertEqual(ep.start_date, "2024-01-01")
        self.assertEqual(ep.end_date, "2024-03-04")
        # nand has NO curated episodes -> honest empty list (matches real nand).
        nand = queries.get_resource_detail("nand", conn=self.conn)
        self.assertEqual(nand.historical_context, [])

    def test_alert_feed_sorted_and_limited(self):
        feed = queries.get_alert_feed(limit=1, conn=self.conn)
        self.assertEqual(len(feed), 1)
        # newest timestamp is the explanation (2026-07-02) over the prediction (2026-07-01).
        self.assertEqual(feed[0].item_type, "explanation")
        self.assertEqual(feed[0].timestamp, "2026-07-02T12:00:00")

    def test_alert_feed_no_padding_when_fewer_than_limit(self):
        feed = queries.get_alert_feed(limit=100, conn=self.conn)
        # exactly the real rows: 1 prediction + 1 explanation = 2, never padded to 100.
        self.assertEqual(len(feed), 2)
        types = sorted(i.item_type for i in feed)
        self.assertEqual(types, ["explanation", "prediction"])
        # prediction item's short_summary is the raw probability (float), not a string.
        pred = next(i for i in feed if i.item_type == "prediction")
        self.assertEqual(pred.short_summary, 0.44)
        self.assertIsNone(pred.status)

    def test_all_contracts_are_asdict_serializable(self):
        objs = [
            queries.get_resource_summary("dram", conn=self.conn),
            queries.get_resource_summary("nand", conn=self.conn),
            queries.get_resource_detail("dram", conn=self.conn),
        ] + queries.get_leaderboard(conn=self.conn) + queries.get_alert_feed(conn=self.conn)
        for o in objs:
            d = dataclasses.asdict(o)              # must not raise
            json.dumps(d, default=str)             # must be JSON-serializable

    def test_no_ui_imports_in_layer(self):
        import dashboard_data.queries as q
        import dashboard_data.contracts as c
        for mod in (q, c):
            src = Path(mod.__file__).read_text()
            for banned in ("import streamlit", "import plotly", "from streamlit", "from plotly"):
                self.assertNotIn(banned, src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
