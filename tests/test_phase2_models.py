"""Offline tests for the Phase-2 off-the-shelf local models:
  * llm_extraction.local_event_classifier  (bart-large-mnli zero-shot event_type)
  * modeling.finbert_sentiment             (FinBERT sentiment, extra feature)
  * forecasting.forecaster.run_chronos     (Chronos zero-shot, unit-level)

No model downloads: the transformers pipeline / chronos pipeline are mocked, so
these run fully offline and fast. Real end-to-end smoke checks were done manually.
Run:  python -m pytest tests/test_phase2_models.py
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from llm_extraction import local_event_classifier as lec  # noqa: E402
from llm_extraction.validator import EVENT_TYPES  # noqa: E402
from modeling import finbert_sentiment as fb  # noqa: E402
from modeling import llm_reasoner  # noqa: E402
from forecasting import forecaster  # noqa: E402
from forecasting.forecaster import HORIZONS, ForecastError  # noqa: E402


def _reset(mod):
    """Reset a module's lazy-pipeline cache between tests."""
    mod._pipe = None
    mod._load_failed = False


# ---- bart zero-shot event_type --------------------------------------------

class LocalEventClassifierTest(unittest.TestCase):
    def setUp(self):
        _reset(lec)
        self.addCleanup(lambda: _reset(lec))

    def test_labels_match_validator_closed_set(self):
        # Guard against drift: every candidate id must be a valid event_type.
        self.assertEqual(set(lec._LABEL_HYPOTHESES), set(EVENT_TYPES))

    def test_maps_top_hypothesis_back_to_event_type_id(self):
        target_hyp = lec._LABEL_HYPOTHESES["supply_disruption"]

        def fake_pipe(text, candidates, multi_label=False):
            return {"labels": [target_hyp] + [c for c in candidates if c != target_hyp],
                    "scores": [0.91] + [0.02] * (len(candidates) - 1)}

        with mock.patch.object(lec, "_get_pipeline", return_value=fake_pipe):
            out = lec.classify_event_type("fab fire halts output")
        self.assertEqual(out[0], "supply_disruption")
        self.assertIn(out[0], EVENT_TYPES)
        self.assertAlmostEqual(out[1], 0.91, places=5)

    def test_empty_text_returns_none(self):
        self.assertIsNone(lec.classify_event_type("  "))

    def test_unavailable_model_returns_none(self):
        with mock.patch.object(lec, "_get_pipeline", return_value=None):
            self.assertIsNone(lec.classify_event_type("some text"))

    def test_inference_error_returns_none(self):
        def boom(*a, **k):
            raise RuntimeError("cuda oom")
        with mock.patch.object(lec, "_get_pipeline", return_value=boom):
            self.assertIsNone(lec.classify_event_type("text"))


# ---- FinBERT sentiment -----------------------------------------------------

class FinbertSentimentTest(unittest.TestCase):
    def setUp(self):
        _reset(fb)
        self.addCleanup(lambda: _reset(fb))

    def test_signed_score_from_distribution(self):
        dist = [{"label": "positive", "score": 0.1},
                {"label": "negative", "score": 0.8},
                {"label": "neutral", "score": 0.1}]
        with mock.patch.object(fb, "_get_pipeline", return_value=lambda t, **k: [dist]):
            out = fb.score_sentiment("prices collapse on oversupply")
        self.assertEqual(out["label"], "negative")
        self.assertAlmostEqual(out["score"], 0.1 - 0.8, places=6)  # pos - neg

    def test_flat_list_shape_supported(self):
        dist = [{"label": "positive", "score": 0.7},
                {"label": "negative", "score": 0.2},
                {"label": "neutral", "score": 0.1}]
        with mock.patch.object(fb, "_get_pipeline", return_value=lambda t, **k: dist):
            out = fb.score_sentiment("shortage tightens supply")
        self.assertEqual(out["label"], "positive")
        self.assertAlmostEqual(out["score"], 0.5, places=6)

    def test_empty_and_unavailable_return_none(self):
        self.assertIsNone(fb.score_sentiment(""))
        with mock.patch.object(fb, "_get_pipeline", return_value=None):
            self.assertIsNone(fb.score_sentiment("text"))


# ---- Chronos (unit-level; real pipeline mocked) ----------------------------

class ChronosUnitTest(unittest.TestCase):
    def tearDown(self):
        forecaster._chronos_pipe = None
        forecaster._chronos_failed = False

    def _df(self, n=20):
        return pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=n, freq="W"),
                             "y": [100.0 + i for i in range(n)]})

    def test_reads_median_quantile_per_horizon(self):
        import torch
        steps = max(HORIZONS.values())
        # quantiles[B=1, H=steps, Q=3]; median (index 1) = horizon index value.
        q = torch.zeros(1, steps, 3)
        for h in range(steps):
            q[0, h, 1] = 500.0 + h
        fake = mock.Mock()
        fake.predict_quantiles.return_value = (q, torch.zeros(1, steps))
        with mock.patch.object(forecaster, "_get_chronos_pipeline", return_value=fake):
            out = forecaster.run_chronos(self._df())
        self.assertEqual(set(out), set(HORIZONS))
        # e.g. '2w' -> horizon index 1 -> 500+1
        self.assertAlmostEqual(out["2w"], 501.0, places=4)

    def test_raises_forecasterror_when_unavailable(self):
        with mock.patch.object(forecaster, "_get_chronos_pipeline", return_value=None):
            with self.assertRaises(ForecastError):
                forecaster.run_chronos(self._df())

    def test_too_short_series_raises(self):
        fake = mock.Mock()
        with mock.patch.object(forecaster, "_get_chronos_pipeline", return_value=fake):
            with self.assertRaises(ForecastError):
                forecaster.run_chronos(self._df(1))


class _FakeClf:
    """Minimal sklearn-like classifier that tolerates NaN features — stands in
    for TabPFN so we can test the train/predict wiring without its gated weights."""
    def fit(self, X, y):
        import numpy as _np
        self._p = float(_np.mean(_np.asarray(y))) if len(y) else 0.5
        return self

    def predict_proba(self, X):
        import numpy as _np
        n = len(X)
        p = min(max(self._p, 0.01), 0.99)
        return _np.column_stack([_np.full(n, 1 - p), _np.full(n, p)])


class TabpfnFusionTest(unittest.TestCase):
    """TabPFN wiring (model selection, filename suffix, bundle tag, SHAP gating).
    The estimator is mocked, so no TabPFN weights / TABPFN_TOKEN are needed."""

    def setUp(self):
        from modeling import train_fusion_model as tf
        self.tf = tf
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "t.db"
        os.environ["SUPPLY_CHAIN_DB"] = str(self.db_path)
        db_init.init_db(self.db_path)
        self.conn = db_init.get_connection(self.db_path)
        tf.init_model_schema(self.conn)
        self.conn.execute("INSERT INTO resource (resource_id,name,aliases) VALUES ('dram','DRAM','[]')")
        self.conn.commit()
        self.mdir = Path(self._tmp.name) / "models"

    def tearDown(self):
        self.conn.close()
        os.environ.pop("SUPPLY_CHAIN_DB", None)
        self._tmp.cleanup()

    def test_unknown_model_type_raises(self):
        with self.assertRaises(ValueError):
            self.tf._new_estimator("xgboost_typo", seed=1)

    def test_tabpfn_selected_suffixed_and_tagged(self):
        df = self.tf.generate_synthetic_dataset(n=40)
        with mock.patch.object(self.tf, "_new_estimator", return_value=_FakeClf()):
            self.tf.train_and_save(df, "synthetic_validation", self.conn,
                                   models_dir=self.mdir, model_type="tabpfn")
        import pickle
        path = self.mdir / "fusion_model_synthetic_validation_tabpfn.pkl"
        self.assertTrue(path.exists())  # suffixed filename, coexists with lightgbm
        with open(path, "rb") as fh:
            bundle = pickle.load(fh)
        self.assertEqual(bundle["model_type"], "tabpfn")

    def test_lightgbm_default_filename_unchanged(self):
        # Estimator mocked (not real LightGBM): this file also imports torch for the
        # Chronos tests, and training real LightGBM in a torch-loaded process
        # segfaults via the macOS libomp conflict. Real LightGBM training is
        # covered in tests/test_fusion_model.py (a torch-free process).
        df = self.tf.generate_synthetic_dataset(n=40)
        with mock.patch.object(self.tf, "_new_estimator", return_value=_FakeClf()):
            self.tf.train_and_save(df, "synthetic_validation", self.conn, models_dir=self.mdir)
        self.assertTrue((self.mdir / "fusion_model_synthetic_validation.pkl").exists())

    def test_predict_gates_shap_for_non_tree(self):
        from modeling import predict
        # Save as a 'real'-mode TabPFN model so the SHAP gate reaches the non-tree
        # branch (the synthetic-mode gate would short-circuit first).
        df = self.tf.generate_synthetic_dataset(n=40)
        with mock.patch.object(self.tf, "_new_estimator", return_value=_FakeClf()):
            self.tf.train_and_save(df, "real", self.conn,
                                   models_dir=self.mdir, model_type="tabpfn")
        self.conn.executescript((PROJECT_ROOT / "schema_forecasting.sql").read_text())
        self.conn.execute("INSERT INTO price_series (resource_id,timestamp,price_usd,unit,source) "
                          "VALUES ('dram','2025-01-02',100.0,'USD_per_share_proxy','yfinance')")
        self.conn.commit()
        out = predict.predict_resource("dram", "2025-01-06", "1m", db_path=self.db_path,
                                       models_dir=self.mdir, prefer="real", model_type="tabpfn")
        self.assertTrue(out["is_real_data_model"])
        self.assertIsNone(out["top_shap_features"])
        self.assertIn("non-tree", out["shap_reason"])


class LlmReasonerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "t.db"
        os.environ["SUPPLY_CHAIN_DB"] = str(self.db_path)
        db_init.init_db(self.db_path)
        conn = db_init.get_connection(self.db_path)
        # feature_builder reads forecast_result (an additive-schema table).
        conn.executescript((PROJECT_ROOT / "schema_forecasting.sql").read_text(encoding="utf-8"))
        conn.execute("INSERT INTO resource (resource_id, name, aliases) VALUES ('dram','DRAM','[]')")
        conn.execute("INSERT INTO price_series (resource_id, timestamp, price_usd, unit, source) "
                     "VALUES ('dram','2025-01-02',100.0,'USD_per_share_proxy','yfinance')")
        conn.commit()
        conn.close()

    def tearDown(self):
        os.environ.pop("SUPPLY_CHAIN_DB", None)
        self._tmp.cleanup()

    def test_valid_judgment_is_labeled_uncalibrated(self):
        with mock.patch.object(llm_reasoner.llm_client, "available_providers", return_value=["groq"]), \
             mock.patch.object(llm_reasoner.llm_client, "generate_text",
                               return_value=('{"tightening_probability": 0.7, "rationale": "z-score elevated"}', "groq")):
            out = llm_reasoner.reason_tightening("dram", "2025-01-06", "1m", db_path=self.db_path)
        self.assertEqual(out["tightening_probability"], 0.7)
        self.assertFalse(out["is_calibrated"])
        self.assertFalse(out["is_real_data_model"])
        self.assertEqual(out["llm_provider"], "groq")
        self.assertIn("features_used", out)
        self.assertNotIn("error", out)

    def test_no_provider_returns_error(self):
        with mock.patch.object(llm_reasoner.llm_client, "available_providers", return_value=[]):
            out = llm_reasoner.reason_tightening("dram", "2025-01-06", db_path=self.db_path)
        self.assertEqual(out["error"], "no_llm_provider")
        self.assertIsNone(out["tightening_probability"])

    def test_out_of_range_probability_rejected(self):
        with mock.patch.object(llm_reasoner.llm_client, "available_providers", return_value=["groq"]), \
             mock.patch.object(llm_reasoner.llm_client, "generate_text",
                               return_value=('{"tightening_probability": 1.8, "rationale": "x"}', "groq")):
            out = llm_reasoner.reason_tightening("dram", "2025-01-06", db_path=self.db_path)
        self.assertEqual(out["error"], "probability_out_of_range")
        self.assertIsNone(out["tightening_probability"])

    def test_unparseable_output_returns_error(self):
        with mock.patch.object(llm_reasoner.llm_client, "available_providers", return_value=["groq"]), \
             mock.patch.object(llm_reasoner.llm_client, "generate_text",
                               return_value=("not json at all", "groq")):
            out = llm_reasoner.reason_tightening("dram", "2025-01-06", db_path=self.db_path)
        self.assertEqual(out["error"], "unparseable_llm_output")


if __name__ == "__main__":
    unittest.main(verbosity=2)
