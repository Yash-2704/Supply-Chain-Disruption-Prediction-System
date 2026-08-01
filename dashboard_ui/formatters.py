"""Pure presentation-logic functions for the dashboard.

NO streamlit/plotly imports, no side effects, no randomness — every function
takes plain data and returns plain data, so it is unit-testable without a UI.
These functions ONLY format/label what the contract already provides; they never
invent categories, scores, thresholds, or risk tiers.
"""
from typing import Optional

# Fixed, canonical badge strings — one per state, reused at every call site.
BADGE_SYNTHETIC = "⚠ SYNTHETIC — NOT A VALIDATED PREDICTION"
BADGE_REAL = "✓ Real model"

# Distinct absence messages (must not be confusable with each other or with "0%").
MISSING_PREDICTION_MSG = "No prediction has been generated for this resource yet."
PROBABILITY_UNAVAILABLE_MSG = "Not available"

_FORECAST_STATUS = {
    "ok": "Forecast available.",
    "insufficient_data": "Not enough history to forecast.",
    "model_error": "Forecast model failed to run.",
    "unstable_disagreement": "Forecast unstable — the two models disagree sharply.",
}

_EXPLANATION_STATUS = {
    "ok": "Explanation generated.",
    # 'not_attempted' and 'insufficient_evidence' are deliberately DISTINCT:
    "not_attempted": "No explanation has been generated for this resource yet.",
    "insufficient_evidence": "Explanation attempted, but no qualifying evidence was found.",
    "llm_error": "Explanation could not be generated (the language-model call failed).",
    "invalid_citations": "Explanation was rejected because its citations could not be verified.",
}


def format_probability(value: Optional[float]) -> str:
    """A calibrated probability (0.0-1.0) as a percentage string, or an honest
    absence message when None. Never returns '0%' or '' for a missing value."""
    if value is None:
        return PROBABILITY_UNAVAILABLE_MSG
    return f"{value * 100:.1f}%"


def format_data_mode_badge(is_real_data_model: bool) -> str:
    """One canonical badge string per state. The False (synthetic) string
    unambiguously signals 'not validated'."""
    return BADGE_REAL if is_real_data_model else BADGE_SYNTHETIC


def format_forecast_status(status: str) -> str:
    """Plain-language sentence per forecast status. An unrecognized status is
    visibly flagged (never silently mapped to a reassuring message)."""
    if status not in _FORECAST_STATUS:
        return f"⚠ Unrecognized forecast status: {status!r}"
    return _FORECAST_STATUS[status]


def format_explanation_status(status: str) -> str:
    """Plain-language sentence per explanation status, keeping 'not_attempted'
    and 'insufficient_evidence' distinct. Unrecognized status is visibly flagged."""
    if status not in _EXPLANATION_STATUS:
        return f"⚠ Unrecognized explanation status: {status!r}"
    return _EXPLANATION_STATUS[status]


def format_missing_prediction() -> str:
    """The message shown when latest_prediction is None (no row exists at all) —
    distinct from format_probability(None), which describes a present row whose
    probability value is null."""
    return MISSING_PREDICTION_MSG
