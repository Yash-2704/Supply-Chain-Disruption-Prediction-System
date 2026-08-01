"""The data contract: typed dataclasses the UI stage renders verbatim.

Choice: `dataclasses` (not TypedDict) — they give a concrete runtime structure,
compose when nested, are `dataclasses.asdict()`-serializable, and give tests a
real constructor. Every Optional field documents what None/missing means, because
the next stage treats these shapes as ground truth and adds no interpretation.

Sentinels used across the contract:
  - A whole nested object being None (e.g. latest_prediction=None) means the
    underlying row does not exist at all.
  - A field being None inside a present object means the row exists but that
    value is genuinely null/unknown — never a computed zero.
  - EXPLANATION_NOT_ATTEMPTED distinguishes "no explanation run yet" from the
    'insufficient_evidence' status (an attempt that found no evidence).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Sentinel for latest_explanation_status when NO explanation row exists at all.
# Distinct from 'insufficient_evidence' (an attempt was made and found nothing).
EXPLANATION_NOT_ATTEMPTED = "not_attempted"


@dataclass
class PriceSeriesPoint:
    date: Optional[str]        # ISO date of the observation; None only if upstream timestamp missing
    price_usd: Optional[float] # observed proxy price; None if not recorded
    unit: Optional[str]        # e.g. 'USD_per_share_proxy'; None if missing (do NOT treat as a commodity price)
    source: Optional[str]      # e.g. 'yfinance'; None if missing


@dataclass
class ForecastSummary:
    series_type: str           # 'price_proxy' | 'demand_intent_activity'
    horizon: str               # '2w'|'1m'|'3m'|'6m', or 'all' for a whole-series status sentinel row
    status: str                # 'ok'|'insufficient_data'|'model_error'|'unstable_disagreement'
    point_value: Optional[float]  # prophet_point if present, else holt_winters_point; None if neither
    lower_bound: Optional[float]  # prophet_lower; None when unavailable or only HW point exists (HW has no interval)
    upper_bound: Optional[float]  # prophet_upper; None likewise
    disagreement_pct: Optional[float]  # Prophet-vs-HW % disagreement; None if not computed


@dataclass
class PredictionSummary:
    # NOTE: the ENTIRE PredictionSummary is None (see ResourceSummary.latest_prediction)
    # when no prediction row exists for the resource. The fields below describe a row
    # that DOES exist.
    run_id: Optional[int]                  # training_run that produced it
    data_mode: Optional[str]               # 'real' | 'synthetic_validation'
    is_real_data_model: bool               # False for synthetic_validation (read from data_mode, not narrative)
    calibrated_probability: Optional[float]  # None => the prediction row exists but its probability is null
    horizon: Optional[str]
    top_shap_features: Optional[list]      # parsed JSON pass-through; None when SHAP was gated off
    predicted_at: Optional[str]            # ISO timestamp; None only if a row exists with no timestamp


@dataclass
class ExplanationSummary:
    status: str                # 'ok'|'insufficient_evidence'|'llm_error'|'invalid_citations'
    narrative: Optional[str]   # VERBATIM incl. built-in disclaimer if present; None unless status='ok'
    citations: Optional[list]  # parsed JSON list pass-through; None unless status='ok'
    evidence_count: int        # canonical evidence items found (0 is a real count here, not a missing sentinel)
    is_real_data_model: bool   # copied from the row's own column, never re-derived from narrative
    generated_at: Optional[str]


@dataclass
class HistoricalEpisode:
    episode_name: Optional[str]  # from curated_episodes.json reference file; None if unmatchable
    justification: Optional[str] # one-sentence real reason, from the curated label rows
    start_date: Optional[str]    # earliest labeled week for this episode (reference range, NOT a current signal)
    end_date: Optional[str]      # latest labeled week for this episode


@dataclass
class ResourceSummary:
    resource_id: str
    name: Optional[str]                                  # None only if resource_id is unknown
    latest_prediction: Optional[PredictionSummary]       # None => no prediction exists for this resource
    latest_forecast_status_by_series: Dict[str, str]     # series_type -> collapsed status; {} if no forecasts
    latest_explanation_status: str                       # a status string, or EXPLANATION_NOT_ATTEMPTED


@dataclass
class ResourceDetail:
    resource_id: str
    name: Optional[str]
    latest_prediction: Optional[PredictionSummary]
    latest_forecast_status_by_series: Dict[str, str]
    latest_explanation_status: str
    price_history: List[PriceSeriesPoint]                # all proxy price points for the resource (may be empty)
    forecast_breakdown: List[ForecastSummary]            # every latest series/horizon row (may be empty)
    explanation: Optional[ExplanationSummary]            # full latest explanation; None if EXPLANATION_NOT_ATTEMPTED
    historical_context: List[HistoricalEpisode]          # curated-episode REFERENCE context; empty if none


@dataclass
class AlertFeedItem:
    resource_id: str
    resource_name: Optional[str]
    item_type: str                    # 'prediction' | 'explanation'
    timestamp: Optional[str]          # predicted_at / generated_at
    is_real_data_model: bool
    status: Optional[str]             # explanation's status; None for prediction items (no status concept)
    # Raw pass-through, NOT a formatted display string:
    #   prediction  -> calibrated_probability (float or None)
    #   explanation -> narrative[:ALERT_SUMMARY_CHARS] (str or None)
    short_summary: Optional[Any]


# Max chars of narrative passed through as an alert's raw short_summary (still raw data, not formatting).
ALERT_SUMMARY_CHARS = 200
