"""As-of cutoff filtering — the anti-lookahead core of the backtest.

Every function here restricts data to timestamps AT OR BEFORE a cutoff. Filtering
is done in Python against PARSED timestamps (never fragile SQL string comparison),
so it is correct across the project's mixed timestamp formats (GDELT-compact
'20260630T114500Z' and ISO '2026-06-24'). compute_asof_features is deterministic,
so an adversarial test can prove future rows never influence it.
"""
import re
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd

# Reuse the labeling stage's mechanical anomaly logic on cutoff-filtered input —
# no reimplementation of prior-stage query logic.
from labeling.price_derived_labels import compute_price_derived_labels, MIN_BASELINE_WEEKS


def parse_timestamp(ts) -> Optional[datetime]:
    """Parse any of the project's timestamp formats to a naive datetime, or None.
    Handles GDELT-compact ('YYYYMMDDTHHMMSSZ', 'YYYYMMDD'), ISO datetime, ISO date."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=None)
    s = str(ts).strip()
    if not s:
        return None
    m = re.fullmatch(r"(\d{8})T(\d{6})Z?", s)          # 20260630T114500Z
    if m:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    if re.fullmatch(r"\d{8}", s):                        # 20260630
        return datetime.strptime(s, "%Y%m%d")
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        pass
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return None


def is_at_or_before(ts, cutoff) -> bool:
    """True iff `ts` parses to a time at or before `cutoff`. Unparseable -> False
    (excluded), because we must never let an untimestamped row into an as-of view."""
    parsed, cut = parse_timestamp(ts), parse_timestamp(cutoff)
    if parsed is None or cut is None:
        return False
    return parsed <= cut


def filter_by_cutoff(rows: List[tuple], cutoff, ts_index: int = 0) -> List[tuple]:
    """Pure: keep only rows whose element at `ts_index` is at-or-before cutoff."""
    return [r for r in rows if is_at_or_before(r[ts_index], cutoff)]


# --- cutoff-aware retrieval (fetch, then Python-filter by parsed timestamp) --- #

def prices_asof(conn, resource_id, cutoff) -> List[Tuple[str, float]]:
    rows = conn.execute(
        "SELECT timestamp, price_usd FROM price_series "
        "WHERE resource_id = ? AND unit = 'USD_per_share_proxy'", (resource_id,)).fetchall()
    return filter_by_cutoff(rows, cutoff, ts_index=0)


def signals_asof(conn, resource_id, cutoff) -> List[Tuple[str, Optional[float]]]:
    rows = conn.execute(
        "SELECT timestamp, severity FROM signal WHERE resource_id = ?", (resource_id,)).fetchall()
    return filter_by_cutoff(rows, cutoff, ts_index=0)


def labels_asof(conn, resource_id, cutoff) -> List[tuple]:
    rows = conn.execute(
        "SELECT week_start, label, source FROM label WHERE resource_id = ?", (resource_id,)).fetchall()
    return filter_by_cutoff(rows, cutoff, ts_index=0)


def weekly_price_series_asof(conn, resource_id, cutoff) -> pd.DataFrame:
    """Cutoff-filtered weekly-mean proxy price series (ds, y). Empty frame if none."""
    prices = prices_asof(conn, resource_id, cutoff)
    if not prices:
        return pd.DataFrame(columns=["ds", "y"])
    df = pd.DataFrame(prices, columns=["ds", "y"])
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"])
    if df.empty:
        return pd.DataFrame(columns=["ds", "y"])
    w = df.set_index("ds")["y"].sort_index().resample("W").mean().dropna()
    return w.reset_index().rename(columns={"ds": "ds", "y": "y"})[["ds", "y"]]


def _trailing_zscore(weekly: pd.DataFrame) -> Optional[float]:
    """z of the latest weekly price vs the prior MIN_BASELINE_WEEKS (same formula as labeling)."""
    if len(weekly) < MIN_BASELINE_WEEKS + 1:
        return None
    y = pd.to_numeric(weekly["y"], errors="coerce")
    window = y.iloc[-(MIN_BASELINE_WEEKS + 1):-1]
    mean, std = float(window.mean()), float(window.std(ddof=0))
    if std <= 0:
        return float("inf") if float(y.iloc[-1]) > mean else 0.0
    return (float(y.iloc[-1]) - mean) / std


def compute_asof_features(conn, resource_id, cutoff) -> dict:
    """Deterministic as-of feature bundle from cutoff-filtered data ONLY.

    Every value here would change if a post-cutoff row leaked in, which is what
    the adversarial lookahead test exploits to prove no leakage occurs.
    """
    prices = prices_asof(conn, resource_id, cutoff)
    weekly = weekly_price_series_asof(conn, resource_id, cutoff)
    sigs = signals_asof(conn, resource_id, cutoff)
    sev_values = [s for _, s in sigs if s is not None]

    anomaly_label_latest = None
    latest_price = latest_price_date = None
    if len(weekly) > 0:
        latest_price = round(float(weekly["y"].iloc[-1]), 6)  # latest weekly-mean proxy
        # Report the true latest UNDERLYING data date (guaranteed <= cutoff), NOT the
        # weekly bucket-end label (which can fall after the cutoff and look like a leak).
        latest_price_date = max(parse_timestamp(ts) for ts, _ in prices).date().isoformat()
        labels = compute_price_derived_labels(weekly)  # REUSED labeling logic
        if labels and labels[-1]["status"] == "ok":
            anomaly_label_latest = labels[-1]["label"]

    zscore = _trailing_zscore(weekly)
    return {
        "cutoff": str(cutoff),
        "n_price_points": len(weekly),
        "latest_price": latest_price,
        "latest_price_date": latest_price_date,
        "price_zscore_latest": (None if zscore is None else round(zscore, 6)),
        "anomaly_label_latest": anomaly_label_latest,
        "n_signals": len(sigs),
        "n_scored_signals": len(sev_values),
        "mean_severity": (round(sum(sev_values) / len(sev_values), 6) if sev_values else None),
    }
