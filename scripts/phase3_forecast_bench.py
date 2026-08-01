"""Phase 3.2 — honest backtest of Prophet vs Holt-Winters vs Chronos.

For each resource's weekly price_proxy series: hold out the last HOLDOUT weeks,
fit each model on the remainder, forecast HOLDOUT weeks ahead, and compare to the
actual held-out values (MAE + MAPE). This is a real accuracy comparison, not just
reading stored point forecasts.

No lightgbm here, so Chronos (torch) is safe in-process. Run:
    python scripts/phase3_forecast_bench.py [--db PATH] [--holdout 8]
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from forecasting import series_builder, forecaster  # noqa: E402
from forecasting.forecaster import ForecastError  # noqa: E402

warnings.filterwarnings("ignore")


def _mae(pred, actual):
    return float(np.mean(np.abs(np.array(pred) - np.array(actual))))


def _mape(pred, actual):
    a = np.array(actual, dtype=float)
    nz = a != 0
    if not nz.any():
        return float("nan")
    return float(np.mean(np.abs((np.array(pred)[nz] - a[nz]) / a[nz])) * 100)


def _forecast_h(model_fn, train_df, h):
    """Return list of h point forecasts from a forecaster fn, or None on failure.
    Reuses the production fitters by requesting horizons 1..h in weeks."""
    horizons = {f"h{i}": i for i in range(1, h + 1)}
    try:
        out = model_fn(train_df, horizons)
    except ForecastError:
        return None
    # prophet returns {k:{point,...}}, others {k: point}
    pts = []
    for i in range(1, h + 1):
        v = out[f"h{i}"]
        pts.append(v["point"] if isinstance(v, dict) else v)
    return pts


def run(db_path=None, holdout=8):
    conn = db_init.get_connection(db_path)
    resources = [r[0] for r in conn.execute("SELECT resource_id FROM resource ORDER BY resource_id")]
    models = {"prophet": forecaster.run_prophet,
              "holt_winters": forecaster.run_holt_winters,
              "chronos": forecaster.run_chronos}
    print("=" * 74)
    print(f"PHASE 3.2 FORECAST BACKTEST   holdout={holdout} weeks   metric=MAE / MAPE%")
    print("=" * 74)
    agg = {m: [] for m in models}
    for rid in resources:
        df = series_builder.build_price_series(rid, conn)
        n = len(df)
        if n < holdout + 12:
            print(f"\n[{rid}] insufficient ({n} weekly pts) — skipped")
            continue
        train, test = df.iloc[:-holdout], df.iloc[-holdout:]
        actual = test["y"].tolist()
        print(f"\n[{rid}]  train={len(train)}wk  test={holdout}wk  "
              f"actual[{actual[0]:.1f}..{actual[-1]:.1f}]")
        for mname, fn in models.items():
            preds = _forecast_h(fn, train, holdout)
            if preds is None:
                print(f"   {mname:<13} FAILED"); continue
            mae, mape = _mae(preds, actual), _mape(preds, actual)
            agg[mname].append(mape)
            print(f"   {mname:<13} MAE={mae:8.2f}  MAPE={mape:6.1f}%")
    conn.close()
    print("\n" + "-" * 74)
    print("AVERAGE MAPE across resources (lower = better):")
    for m, vals in agg.items():
        if vals:
            print(f"   {m:<13} {np.mean(vals):6.1f}%   (n={len(vals)} resources)")
    print("=" * 74)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--holdout", type=int, default=8)
    args = ap.parse_args()
    run(db_path=args.db, holdout=args.holdout)
