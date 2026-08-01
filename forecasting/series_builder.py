"""Build the two forecastable input series per resource, at a common weekly cadence.

Both return a DataFrame with columns ['ds' (datetime), 'y' (float)], or an empty
DataFrame when no rows exist. The forecaster decides sufficiency vs MIN_DATA_POINTS.
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Weekly: smooths noisy/gappy daily stock proxies and gives both series one
# frequency so 2w/1m/3m/6m horizons are directly comparable.
RESAMPLE_FREQ = "W"

_EMPTY = pd.DataFrame(columns=["ds", "y"])


def _resample(df: pd.DataFrame, how: str) -> pd.DataFrame:
    """Resample a (ds,y) frame to weekly using mean|sum; drop empty weeks."""
    if df.empty:
        return _EMPTY.copy()
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"])
    if df.empty:
        return _EMPTY.copy()
    s = df.set_index("ds")["y"].sort_index()
    agg = s.resample(RESAMPLE_FREQ).mean() if how == "mean" else s.resample(RESAMPLE_FREQ).sum(min_count=1)
    agg = agg.dropna()
    return agg.reset_index().rename(columns={"ds": "ds", "y": "y"})[["ds", "y"]] \
        if not agg.empty else _EMPTY.copy()


def build_price_series(resource_id: str, conn) -> pd.DataFrame:
    """Weekly-mean stock-proxy price for `resource_id`.

    Scope: unit='USD_per_share_proxy' AND resource_id IS NOT NULL — macro rows
    (resource_id NULL / other units) are never per-resource forecastable.
    """
    rows = conn.execute(
        "SELECT timestamp AS ds, price_usd AS y FROM price_series "
        "WHERE resource_id = ? AND resource_id IS NOT NULL "
        "AND unit = 'USD_per_share_proxy'",
        (resource_id,),
    ).fetchall()
    df = pd.DataFrame(rows, columns=["ds", "y"])
    return _resample(df, how="mean")


def build_demand_intent_series(resource_id: str, conn) -> pd.DataFrame:
    """Weekly severity-weighted demand-intent activity for `resource_id`.

    Formula: y = SUM(severity) over demand_intent_raw signals for this resource
    within each ISO week (each signal contributes its LLM severity). Only
    severity-scored rows count; unscored rows are not activity we can weight.
    """
    rows = conn.execute(
        "SELECT timestamp AS ds, severity AS y FROM signal "
        "WHERE resource_id = ? AND signal_type = 'demand_intent_raw' "
        "AND severity IS NOT NULL",
        (resource_id,),
    ).fetchall()
    df = pd.DataFrame(rows, columns=["ds", "y"])
    return _resample(df, how="sum")
