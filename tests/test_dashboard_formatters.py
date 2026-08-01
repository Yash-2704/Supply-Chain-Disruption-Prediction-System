"""Required, fully-offline tests for the dashboard's pure formatter functions.

No Streamlit runtime involved. Run:  python -m pytest tests/test_dashboard_formatters.py
"""
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard_ui import formatters as fmt  # noqa: E402
from dashboard_data.contracts import EXPLANATION_NOT_ATTEMPTED  # noqa: E402

# The closed forecast-status set, confirmed against schema_forecasting.sql / the
# forecasting stage (status ∈ {ok, insufficient_data, model_error, unstable_disagreement}).
FORECAST_STATUSES = ["ok", "insufficient_data", "model_error", "unstable_disagreement"]
# Explanation status set from schema_explanations.sql + the not_attempted sentinel.
EXPLANATION_STATUSES = ["ok", "insufficient_evidence", "llm_error", "invalid_citations",
                        EXPLANATION_NOT_ATTEMPTED]


class ProbabilityTest(unittest.TestCase):
    def test_real_float_formats_as_percent(self):
        self.assertEqual(fmt.format_probability(0.444), "44.4%")
        self.assertEqual(fmt.format_probability(0.0), "0.0%")   # a REAL zero probability
        self.assertEqual(fmt.format_probability(1.0), "100.0%")

    def test_none_is_distinct_non_numeric_message(self):
        out = fmt.format_probability(None)
        self.assertEqual(out, fmt.PROBABILITY_UNAVAILABLE_MSG)
        self.assertNotIn("%", out)          # not a fake percentage
        self.assertNotEqual(out, "0%")
        self.assertNotEqual(out, "")


class DataModeBadgeTest(unittest.TestCase):
    def test_true_vs_false_distinct(self):
        real = fmt.format_data_mode_badge(True)
        synth = fmt.format_data_mode_badge(False)
        self.assertNotEqual(real, synth)

    def test_false_signals_not_validated(self):
        synth = fmt.format_data_mode_badge(False)
        self.assertIn("⚠", synth)
        self.assertIn("SYNTHETIC", synth.upper())
        self.assertIn("NOT", synth.upper())  # unambiguous "not validated" signal


class ForecastStatusTest(unittest.TestCase):
    def test_every_status_distinct_sentence(self):
        outputs = {s: fmt.format_forecast_status(s) for s in FORECAST_STATUSES}
        self.assertEqual(len(set(outputs.values())), len(FORECAST_STATUSES))  # all distinct
        for s, msg in outputs.items():
            self.assertTrue(msg and not msg.startswith("⚠ Unrecognized"))

    def test_unrecognized_is_flagged_not_reassuring(self):
        out = fmt.format_forecast_status("everything_is_fine")
        self.assertIn("Unrecognized", out)
        self.assertNotIn("available", out.lower())  # never a success-sounding default


class ExplanationStatusTest(unittest.TestCase):
    def test_not_attempted_vs_insufficient_evidence_distinct(self):
        a = fmt.format_explanation_status("not_attempted")
        b = fmt.format_explanation_status("insufficient_evidence")
        self.assertNotEqual(a, b)
        # sanity: each conveys its specific meaning
        self.assertIn("yet", a.lower())              # never run
        self.assertIn("no qualifying evidence", b.lower())

    def test_all_statuses_distinct(self):
        outs = {s: fmt.format_explanation_status(s) for s in EXPLANATION_STATUSES}
        self.assertEqual(len(set(outs.values())), len(EXPLANATION_STATUSES))

    def test_unrecognized_flagged(self):
        self.assertIn("Unrecognized", fmt.format_explanation_status("bogus"))


class MissingPredictionTest(unittest.TestCase):
    def test_distinct_from_probability_none_and_zero(self):
        msg = fmt.format_missing_prediction()
        self.assertNotEqual(msg, fmt.format_probability(None))
        self.assertNotEqual(msg, fmt.format_probability(0.0))
        self.assertNotEqual(msg, "0%")
        self.assertNotEqual(msg, "")
        self.assertIn("no prediction", msg.lower())


class PurityTest(unittest.TestCase):
    def test_formatters_have_no_ui_imports(self):
        src = (PROJECT_ROOT / "dashboard_ui" / "formatters.py").read_text()
        for banned in ("import streamlit", "import plotly",
                       "from streamlit", "from plotly"):
            self.assertNotIn(banned, src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
