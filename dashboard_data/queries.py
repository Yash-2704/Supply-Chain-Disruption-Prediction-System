"""Read-only query functions producing the dashboard contract (contracts.py).

Connection pattern: every public function accepts an optional `conn`. If None,
it opens one via db_init.get_connection() and closes it; if provided (tests, or
batch callers), it uses it and does NOT close it. No function writes, ever.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import db_init
from dashboard_data.contracts import (
    ALERT_SUMMARY_CHARS, EXPLANATION_NOT_ATTEMPTED, AlertFeedItem, ExplanationSummary,
    ForecastSummary, HistoricalEpisode, PredictionSummary, PriceSeriesPoint,
    ResourceDetail, ResourceSummary,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURATED_EPISODES_PATH = PROJECT_ROOT / "data" / "curated_episodes.json"

# Priority when collapsing a series' multiple horizon rows into one status.
_STATUS_PRIORITY = ["model_error", "insufficient_data", "unstable_disagreement", "ok"]


def _resolve(conn):
    """Return (conn, should_close)."""
    if conn is not None:
        return conn, False
    return db_init.get_connection(), True


def _json_or_none(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


# --------------------------------------------------------------------------- #
# resources
# --------------------------------------------------------------------------- #

def get_all_resources(conn=None) -> List[dict]:
    """[{id, name}] for every row in `resource`, queried dynamically."""
    conn, close = _resolve(conn)
    try:
        return [{"id": r[0], "name": r[1]}
                for r in conn.execute("SELECT resource_id, name FROM resource ORDER BY resource_id")]
    finally:
        if close:
            conn.close()


# --------------------------------------------------------------------------- #
# per-resource building blocks
# --------------------------------------------------------------------------- #

def _latest_prediction(conn, resource_id) -> Optional[PredictionSummary]:
    row = conn.execute(
        "SELECT p.run_id, t.data_mode, p.calibrated_probability, p.horizon, "
        "p.top_shap_features, p.predicted_at "
        "FROM prediction p JOIN training_run t ON p.run_id = t.run_id "
        "WHERE p.resource_id = ? ORDER BY p.predicted_at DESC, p.id DESC LIMIT 1",
        (resource_id,)).fetchone()
    if row is None:
        return None  # no prediction exists at all
    return PredictionSummary(
        run_id=row[0], data_mode=row[1], is_real_data_model=(row[1] == "real"),
        calibrated_probability=row[2], horizon=row[3],
        top_shap_features=_json_or_none(row[4]), predicted_at=row[5])


def _latest_forecast_rows(conn, resource_id) -> List[tuple]:
    """Latest run's forecast rows for the resource (max run_at per series_type)."""
    rows = conn.execute(
        "SELECT series_type, horizon, status, prophet_point, prophet_lower, "
        "prophet_upper, holt_winters_point, disagreement_pct, run_at "
        "FROM forecast_result WHERE resource_id = ?", (resource_id,)).fetchall()
    latest_run = {}
    for r in rows:
        series_type, run_at = r[0], r[8]
        if series_type not in latest_run or run_at > latest_run[series_type]:
            latest_run[series_type] = run_at
    return [r for r in rows if r[8] == latest_run.get(r[0])]


def _forecast_summaries(conn, resource_id) -> List[ForecastSummary]:
    out = []
    for series_type, horizon, status, p_pt, p_lo, p_hi, hw_pt, disagree, _run in \
            _latest_forecast_rows(conn, resource_id):
        # point_value: prefer prophet_point; fall back to HW only if prophet is None.
        if p_pt is not None:
            point, lower, upper = p_pt, p_lo, p_hi
        elif hw_pt is not None:
            point, lower, upper = hw_pt, None, None  # HW has no interval
        else:
            point, lower, upper = None, None, None
        out.append(ForecastSummary(
            series_type=series_type, horizon=horizon, status=status,
            point_value=point, lower_bound=lower, upper_bound=upper,
            disagreement_pct=disagree))
    return out


def _forecast_status_by_series(summaries: List[ForecastSummary]) -> Dict[str, str]:
    by_series: Dict[str, set] = {}
    for f in summaries:
        by_series.setdefault(f.series_type, set()).add(f.status)
    collapsed = {}
    for series_type, statuses in by_series.items():
        collapsed[series_type] = next(
            (s for s in _STATUS_PRIORITY if s in statuses), sorted(statuses)[0])
    return collapsed


def _latest_explanation(conn, resource_id) -> Optional[ExplanationSummary]:
    row = conn.execute(
        "SELECT status, narrative, citations, evidence_count, is_real_data_model, generated_at "
        "FROM explanation WHERE resource_id = ? ORDER BY generated_at DESC, id DESC LIMIT 1",
        (resource_id,)).fetchone()
    if row is None:
        return None  # never attempted
    return ExplanationSummary(
        status=row[0], narrative=row[1], citations=_json_or_none(row[2]),
        evidence_count=row[3] if row[3] is not None else 0,
        is_real_data_model=bool(row[4]), generated_at=row[5])


def _price_history(conn, resource_id) -> List[PriceSeriesPoint]:
    rows = conn.execute(
        "SELECT timestamp, price_usd, unit, source FROM price_series "
        "WHERE resource_id = ? AND unit = 'USD_per_share_proxy' ORDER BY timestamp",
        (resource_id,)).fetchall()
    return [PriceSeriesPoint(date=r[0], price_usd=r[1], unit=r[2], source=r[3]) for r in rows]


def _historical_context(conn, resource_id) -> List[HistoricalEpisode]:
    """Curated-episode REFERENCE context from the label table, episode_name enriched
    from the read-only curated_episodes.json (None if unmatchable)."""
    rows = conn.execute(
        "SELECT justification, MIN(week_start), MAX(week_start) FROM label "
        "WHERE resource_id = ? AND source = 'curated_episode' GROUP BY justification",
        (resource_id,)).fetchall()
    if not rows:
        return []
    name_by_just = _episode_names_by_justification()
    return [HistoricalEpisode(
        episode_name=name_by_just.get(just), justification=just,
        start_date=start, end_date=end) for just, start, end in rows]


def _episode_names_by_justification() -> Dict[str, str]:
    try:
        doc = json.loads(CURATED_EPISODES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {e.get("justification"): e.get("episode_name")
            for e in doc.get("episodes", []) if e.get("justification")}


# --------------------------------------------------------------------------- #
# public views
# --------------------------------------------------------------------------- #

def get_resource_summary(resource_id, conn=None) -> ResourceSummary:
    conn, close = _resolve(conn)
    try:
        name_row = conn.execute("SELECT name FROM resource WHERE resource_id = ?",
                                (resource_id,)).fetchone()
        summaries = _forecast_summaries(conn, resource_id)
        explanation = _latest_explanation(conn, resource_id)
        return ResourceSummary(
            resource_id=resource_id,
            name=name_row[0] if name_row else None,
            latest_prediction=_latest_prediction(conn, resource_id),
            latest_forecast_status_by_series=_forecast_status_by_series(summaries),
            latest_explanation_status=(explanation.status if explanation
                                       else EXPLANATION_NOT_ATTEMPTED))
    finally:
        if close:
            conn.close()


def get_leaderboard(conn=None) -> List[ResourceSummary]:
    conn, close = _resolve(conn)
    try:
        # One row per resource, regardless of individual sparsity; never drops a resource.
        return [get_resource_summary(r["id"], conn=conn) for r in get_all_resources(conn=conn)]
    finally:
        if close:
            conn.close()


def get_resource_detail(resource_id, conn=None) -> ResourceDetail:
    conn, close = _resolve(conn)
    try:
        summary = get_resource_summary(resource_id, conn=conn)  # reuse, don't duplicate
        return ResourceDetail(
            resource_id=summary.resource_id,
            name=summary.name,
            latest_prediction=summary.latest_prediction,
            latest_forecast_status_by_series=summary.latest_forecast_status_by_series,
            latest_explanation_status=summary.latest_explanation_status,
            price_history=_price_history(conn, resource_id),
            forecast_breakdown=_forecast_summaries(conn, resource_id),
            explanation=_latest_explanation(conn, resource_id),
            historical_context=_historical_context(conn, resource_id))
    finally:
        if close:
            conn.close()


def get_alert_feed(limit=20, conn=None) -> List[AlertFeedItem]:
    conn, close = _resolve(conn)
    try:
        names = {r["id"]: r["name"] for r in get_all_resources(conn=conn)}
        items: List[AlertFeedItem] = []

        for run_id, rid, prob, predicted_at, data_mode in conn.execute(
            "SELECT p.run_id, p.resource_id, p.calibrated_probability, p.predicted_at, "
            "t.data_mode FROM prediction p JOIN training_run t ON p.run_id = t.run_id"):
            items.append(AlertFeedItem(
                resource_id=rid, resource_name=names.get(rid), item_type="prediction",
                timestamp=predicted_at, is_real_data_model=(data_mode == "real"),
                status=None, short_summary=prob))  # raw probability, no formatting

        for rid, status, narrative, is_real, generated_at in conn.execute(
            "SELECT resource_id, status, narrative, is_real_data_model, generated_at "
            "FROM explanation"):
            summary = narrative[:ALERT_SUMMARY_CHARS] if narrative else None
            items.append(AlertFeedItem(
                resource_id=rid, resource_name=names.get(rid), item_type="explanation",
                timestamp=generated_at, is_real_data_model=bool(is_real),
                status=status, short_summary=summary))

        # Newest first; None timestamps sort last. Return only real rows, never padded.
        items.sort(key=lambda it: (it.timestamp is not None, it.timestamp or ""), reverse=True)
        return items[:limit]
    finally:
        if close:
            conn.close()
