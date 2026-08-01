"""Offline tests for the merged dashboard integration.

Pure logic / config only — no torch, no model training (that runs in the export
adapter's subprocesses). Verifies the config is internally consistent and the
chronological split behaves correctly.
Run:  python -m pytest tests/test_merge_dashboard.py
"""
import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from risk_dashboard import config as cfg  # noqa: E402
from modeling import benchmark_data  # noqa: E402


class ConfigConsistencyTest(unittest.TestCase):
    def test_resource_maps_align(self):
        self.assertEqual(set(cfg.RESOURCES), set(cfg.RESOURCE_ID))
        # ID_TO_DISPLAY is the exact inverse of RESOURCE_ID
        self.assertEqual(cfg.ID_TO_DISPLAY, {v: k for k, v in cfg.RESOURCE_ID.items()})
        for r in cfg.RESOURCES:
            self.assertIn(r, cfg.RESOURCE_MONOGRAM)
            self.assertIn(r, cfg.RESOURCE_COLORS)

    def test_four_models_have_colors(self):
        for m in ["LightGBM", "XGBoost", "FT-Transformer", "TabPFN"]:
            self.assertIn(m, cfg.MODEL_COLORS)

    def test_disruption_events_shape(self):
        self.assertTrue(cfg.DISRUPTION_EVENTS)
        for e in cfg.DISRUPTION_EVENTS:
            for k in ("start", "end", "description", "resources", "severity"):
                self.assertIn(k, e)
            # resources must be known display names
            for r in e["resources"]:
                self.assertIn(r, cfg.RESOURCES)

    def test_risk_thresholds_ordered(self):
        t = cfg.RISK_THRESHOLDS
        self.assertLess(t["low"], t["medium"])
        self.assertLess(t["medium"], t["high"])


class ChronologicalSplitTest(unittest.TestCase):
    def _df(self, n=100):
        weeks = pd.date_range("2023-01-02", periods=n, freq="W-MON").date.astype(str)
        return pd.DataFrame({
            "week_start": list(weeks),
            "label": [i % 2 for i in range(n)],
            "resource_id": ["dram"] * n,
        })

    def test_split_is_time_ordered_and_sized(self):
        train, test = benchmark_data.chronological_split(self._df(100), test_ratio=0.2)
        self.assertEqual(len(train), 80)
        self.assertEqual(len(test), 20)
        # every train week is <= every test week (no future leakage into train)
        self.assertLessEqual(train["week_start"].max(), test["week_start"].min())

    def test_meta_columns_present(self):
        self.assertEqual(benchmark_data.META_COLS, ["resource_id", "week_start"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
