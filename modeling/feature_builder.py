"""Single source of truth for fusion-model feature construction.

Every feature is built strictly from data available AS OF the requested week:
all queries cut off at the end (Sunday) of that ISO week, and forecast features
are attached only if the forecast's run_at falls at-or-before that cutoff — so a
historical feature row can never leak future data. A missing upstream value is
NaN (genuinely "unknown"), NEVER silently 0.
"""
import logging

import numpy as np
import pandas as pd

from labeling import week_start  # reuse the established ISO-week-Monday grid

logger = logging.getLogger(__name__)

# The feature set, defined ONCE. (feature -> what it is / upstream source.)
FEATURE_COLUMNS = [
    "price_latest",               # price_series: latest weekly proxy value at-or-before the week
    "price_trend_4w",             # price_series: % change of proxy over the trailing 4 weeks
    "price_zscore_8w",            # price_series: z of latest vs trailing 8-week baseline (mirrors labeling)
    "forecast_point_1m",          # forecast_result: Prophet 1-month point (only if forecast available as-of week)
    "forecast_interval_width_1m", # forecast_result: Prophet upper-lower at 1m (uncertainty width)
    "forecast_disagreement_1m",   # forecast_result: Prophet-vs-Holt-Winters disagreement % at 1m
    "forecast_unstable_flag",     # forecast_result: 1.0 if price_proxy forecast unstable, 0.0 if ok, NaN if none
    "signal_severity_mean_4w",    # signal: mean LLM severity of scored signals in trailing 4 weeks (NaN if none)
    "signal_count_4w",            # signal: # of severity-scored signals in trailing 4 weeks (NaN if severity unpopulated)
    "hhi_country",                # resource: country-concentration HHI (structural), NaN if unpopulated
]

_NAN = float("nan")


def _week_bounds(week_start_str):
    """(monday, sunday_cutoff) Timestamps for the ISO week of `week_start_str`."""
    monday = pd.Timestamp(week_start(week_start_str))
    return monday, monday + pd.Timedelta(days=6)


def _weekly_prices_upto(conn, resource_id, cutoff) -> pd.Series:
    """Weekly-mean proxy price series for the resource, timestamp <= cutoff."""
    rows = conn.execute(
        "SELECT timestamp, price_usd FROM price_series "
        "WHERE resource_id = ? AND unit = 'USD_per_share_proxy' AND timestamp <= ?",
        (resource_id, cutoff.date().isoformat()),
    ).fetchall()
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows, columns=["ds", "y"])
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"])
    return df.set_index("ds")["y"].sort_index().resample("W").mean().dropna()


def build_feature_row(resource_id: str, week_start_str: str, conn) -> dict:
    """Build one FEATURE_COLUMNS-shaped feature dict for (resource, week).
    Uses only data at-or-before the end of that week (no lookahead)."""
    _, cutoff = _week_bounds(week_start_str)
    row = {c: _NAN for c in FEATURE_COLUMNS}

    # --- price current-state features ---
    prices = _weekly_prices_upto(conn, resource_id, cutoff)
    if len(prices) >= 1:
        row["price_latest"] = float(prices.iloc[-1])
    if len(prices) >= 5:
        prev = float(prices.iloc[-5])
        if prev != 0:
            row["price_trend_4w"] = (float(prices.iloc[-1]) / prev - 1.0) * 100.0
    if len(prices) >= 9:
        window = prices.iloc[-9:-1]  # trailing 8 weeks, excluding current
        mean, std = float(window.mean()), float(window.std(ddof=0))
        if std > 0:
            row["price_zscore_8w"] = (float(prices.iloc[-1]) - mean) / std

    # --- forecast features (only if the forecast existed as of this week) ---
    fc_rows = conn.execute(
        "SELECT horizon, run_at, prophet_point, prophet_lower, prophet_upper, "
        "disagreement_pct, status FROM forecast_result "
        "WHERE resource_id = ? AND series_type = 'price_proxy' AND run_at <= ?",
        (resource_id, cutoff.isoformat()),
    ).fetchall()
    if fc_rows:
        statuses = {r[6] for r in fc_rows}
        if "unstable_disagreement" in statuses:
            row["forecast_unstable_flag"] = 1.0
        elif "ok" in statuses:
            row["forecast_unstable_flag"] = 0.0
        for horizon, _run, point, lower, upper, disagree, status in fc_rows:
            if horizon == "1m" and status in ("ok", "unstable_disagreement"):
                if point is not None:
                    row["forecast_point_1m"] = float(point)
                if lower is not None and upper is not None:
                    row["forecast_interval_width_1m"] = float(upper) - float(lower)
                if disagree is not None:
                    row["forecast_disagreement_1m"] = float(disagree)

    # --- signal severity features (NaN when severity is unpopulated) ---
    total_scored = conn.execute(
        "SELECT COUNT(*) FROM signal WHERE resource_id = ? AND severity IS NOT NULL",
        (resource_id,),
    ).fetchone()[0]
    if total_scored > 0:
        lo = (cutoff - pd.Timedelta(days=28)).date().isoformat()
        hi = cutoff.date().isoformat()
        agg = conn.execute(
            "SELECT AVG(severity), COUNT(*) FROM signal WHERE resource_id = ? "
            "AND severity IS NOT NULL AND timestamp > ? AND timestamp <= ?",
            (resource_id, lo, hi),
        ).fetchone()
        row["signal_severity_mean_4w"] = float(agg[0]) if agg[0] is not None else _NAN
        row["signal_count_4w"] = float(agg[1])
    # else: leave both NaN — no scored signals means "unknown", not zero.

    # --- structural feature ---
    hhi = conn.execute("SELECT hhi_country FROM resource WHERE resource_id = ?",
                       (resource_id,)).fetchone()
    if hhi and hhi[0] is not None:
        row["hhi_country"] = float(hhi[0])

    return row


def build_training_table(conn) -> pd.DataFrame:
    """Join both label sources (status='ok') against real feature rows.

    A label is a usable REAL training example only if a real feature row can be
    built for it (i.e., real price data exists at-or-before that week -> price_latest
    is not NaN). Returns a DataFrame with FEATURE_COLUMNS + label, label_source,
    resource_id, week_start. Logs the real overlap count split by source.
    """
    labels = conn.execute(
        "SELECT resource_id, week_start, label, source FROM label WHERE status = 'ok'"
    ).fetchall()

    usable, by_source_total, by_source_usable = [], {}, {}
    for resource_id, wk, label, source in labels:
        by_source_total[source] = by_source_total.get(source, 0) + 1
        feats = build_feature_row(resource_id, wk, conn)
        if not np.isnan(feats["price_latest"]):  # real feature genuinely exists
            feats.update({"label": int(label), "label_source": source,
                          "resource_id": resource_id, "week_start": wk})
            usable.append(feats)
            by_source_usable[source] = by_source_usable.get(source, 0) + 1

    logger.info("build_training_table: real usable (feature,label) pairs = %d", len(usable))
    for src in by_source_total:
        logger.info("  %s: %d ok-labels, %d with real feature overlap",
                    src, by_source_total[src], by_source_usable.get(src, 0))

    cols = FEATURE_COLUMNS + ["label", "label_source", "resource_id", "week_start"]
    return pd.DataFrame(usable, columns=cols)
