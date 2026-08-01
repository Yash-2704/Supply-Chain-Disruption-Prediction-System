"""Mechanical price-derived anomaly labels from the weekly price-proxy series.

Rule: a week is label=1 iff its price is > ZSCORE_THRESHOLD std above a trailing
baseline AND it sits within a run of >= MIN_SUSTAINED_WEEKS consecutive elevated
weeks. Weeks without >= MIN_BASELINE_WEEKS of trailing history are
'insufficient_data' — never labeled from too little baseline.

Uses price_series only (via series_builder) — NO dependency on signal.severity,
which may be entirely NULL. (A severity-weighted variant is a natural future
enhancement once LLM scoring has been run against the live database.)

Known limitation: the baseline is the IMMEDIATE trailing window, so this detects
a *ramping* sustained tightening (prices climbing week over week) robustly, but
an instantaneous permanent level-shift is only flagged transiently before the
trailing baseline catches up to the new level — a standard property of
rolling-baseline anomaly detection, acceptable for this weekly, short-history MVP.
"""
import logging
from typing import List, Optional

import pandas as pd

from . import week_start
from forecasting.series_builder import build_price_series

logger = logging.getLogger(__name__)

# Textbook baseline is ~52 weeks; real coverage is only ~19 weekly points, so a
# full-year window is impossible. 8 weeks (~2 months) is the smallest trailing
# window that yields a stable-enough mean/std for a meaningful z-score while
# still leaving a useful number of evaluable weeks. Deliberately short of ideal.
MIN_BASELINE_WEEKS = 8

# ~2 sigma above the trailing mean — a conventional anomaly gate (~top 2.3%
# under normality); high enough to ignore routine week-to-week fluctuation.
ZSCORE_THRESHOLD = 2.0

# A single elevated week is likely noise; requiring 3 consecutive elevated weeks
# (~most of a month at weekly cadence) captures a genuine sustained tightening.
MIN_SUSTAINED_WEEKS = 3


def compute_price_derived_labels(series_df: pd.DataFrame) -> List[dict]:
    """Given a weekly (ds, y) price frame, return per-week label dicts:
    {week_start, label (0/1/None), status, justification}."""
    if series_df is None or len(series_df) == 0:
        return []
    df = series_df.sort_values("ds").reset_index(drop=True)
    y = pd.to_numeric(df["y"], errors="coerce")
    n = len(df)

    results: List[dict] = []
    elevated = {}  # evaluable index -> bool
    zscores = {}
    for i in range(n):
        wk = week_start(df["ds"].iloc[i])
        if i < MIN_BASELINE_WEEKS:
            results.append({"week_start": wk, "label": None, "status": "insufficient_data",
                            "justification": f"< {MIN_BASELINE_WEEKS} trailing weeks of history"})
            continue
        window = y.iloc[i - MIN_BASELINE_WEEKS:i]  # trailing, excludes current week
        mean = float(window.mean())
        std = float(window.std(ddof=0))
        val = float(y.iloc[i])
        if std > 0:
            z = (val - mean) / std
        else:  # perfectly flat baseline: any rise above it is a clear anomaly
            z = float("inf") if val > mean else 0.0
        zscores[i] = z
        elevated[i] = z > ZSCORE_THRESHOLD
        results.append({"week_start": wk, "label": 0, "status": "ok", "justification": ""})

    # Assign label=1 only to elevated weeks within a sustained consecutive run.
    labeled_one = _sustained_indices(elevated, MIN_SUSTAINED_WEEKS)
    for i in range(n):
        if results[i]["status"] != "ok":
            continue
        z = zscores.get(i, 0.0)
        if i in labeled_one:
            results[i]["label"] = 1
            results[i]["justification"] = (
                f"z-score {z:.2f} > {ZSCORE_THRESHOLD} sustained "
                f">= {MIN_SUSTAINED_WEEKS} consecutive weeks")
        else:
            results[i]["label"] = 0
            note = "not elevated" if not elevated.get(i) else "elevated but not sustained"
            results[i]["justification"] = f"z-score {z:.2f}; {note}"
    return results


def _sustained_indices(elevated: dict, min_run: int) -> set:
    """Indices belonging to a maximal run of consecutive elevated evaluable weeks
    of length >= min_run."""
    idxs = sorted(elevated)
    out, run = set(), []
    for i in idxs:
        if elevated[i]:
            if run and i == run[-1] + 1:
                run.append(i)
            else:                       # start a new run, flushing any prior one
                if len(run) >= min_run:
                    out.update(run)
                run = [i]
        else:                           # elevated broken; flush the run
            if len(run) >= min_run:
                out.update(run)
            run = []
    if len(run) >= min_run:
        out.update(run)
    return out


def build_price_derived_for_resource(resource_id: str, conn) -> List[dict]:
    """Fetch the resource's weekly price series and compute its labels."""
    series = build_price_series(resource_id, conn)
    return compute_price_derived_labels(series)
