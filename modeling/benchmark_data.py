"""Prepare the labelled dataset + chronological split + scoring timeline for the
4-model benchmark that feeds the merged dashboard.

Pure pandas/sqlite (no torch/lightgbm here) so it is fast and safe to run in the
same process as anything. Heavy model training happens in separate subprocesses
(benchmark_train.py) to avoid the torch+OpenMP segfault on macOS.

Label choice: BOTH sources (curated_episode + price_derived). Rationale — the
curated negatives are all from one 2023 window, so a curated-only chronological
test split would be single-class and degenerate. price_derived labels span every
week with both classes, giving a valid time-based split and realistic (non-inflated)
metrics. The partial price-circularity of price_derived and the homogeneous curated
negatives are surfaced honestly in the dashboard's caveat banner.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from modeling import feature_builder  # noqa: E402
from modeling.feature_builder import FEATURE_COLUMNS, build_feature_row  # noqa: E402
from labeling import week_start  # noqa: E402

META_COLS = ["resource_id", "week_start"]
TEST_RATIO = 0.2


def build_labeled(conn, sources=("curated_episode", "price_derived")) -> pd.DataFrame:
    """All usable (feature,label) rows for the given label sources, deduped to one
    row per (resource, week) — if both sources label the same week, curated wins."""
    tt = feature_builder.build_training_table(conn)
    if tt.empty:
        return tt
    tt = tt[tt["label_source"].isin(sources)].copy()
    # curated_episode preferred over price_derived on the same (resource, week)
    tt["_pri"] = (tt["label_source"] == "curated_episode").astype(int)
    tt = (tt.sort_values("_pri", ascending=False)
            .drop_duplicates(subset=["resource_id", "week_start"], keep="first"))
    keep = FEATURE_COLUMNS + ["label", "label_source"] + META_COLS
    return tt[keep].sort_values("week_start").reset_index(drop=True)


def chronological_split(df: pd.DataFrame, test_ratio=TEST_RATIO):
    """Time-based split: earliest (1-test_ratio) weeks train, latest test. Sorted
    by week_start (not random) — the correct choice for time series."""
    df = df.sort_values("week_start").reset_index(drop=True)
    split = int(len(df) * (1 - test_ratio))
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def build_timeline(conn, start="2023-01-01", end=None) -> pd.DataFrame:
    """Feature rows for EVERY weekly Monday per resource in [start, end], for
    scoring the risk timeline (independent of whether a week is labelled)."""
    resources = [r[0] for r in conn.execute("SELECT resource_id FROM resource ORDER BY resource_id")]
    if end is None:
        end = conn.execute("SELECT MAX(timestamp) FROM price_series "
                           "WHERE unit='USD_per_share_proxy'").fetchone()[0] or "2026-07-01"
    weeks = pd.date_range(start=week_start(start), end=end, freq="W-MON")
    rows = []
    for rid in resources:
        for ws in weeks:
            wk = ws.date().isoformat()
            feats = build_feature_row(rid, wk, conn)
            if not np.isnan(feats["price_latest"]):  # only weeks with real price data
                feats.update({"resource_id": rid, "week_start": wk})
                rows.append(feats)
    cols = FEATURE_COLUMNS + META_COLS
    return pd.DataFrame(rows, columns=cols)


def prepare(work_dir: Path, db_path=None) -> dict:
    """Write train.csv / test.csv / timeline.csv into work_dir; return a summary."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    conn = db_init.get_connection(db_path)
    try:
        labeled = build_labeled(conn)
        if labeled.empty:
            raise SystemExit("No labelled rows — run build_labels.py first.")
        train, test = chronological_split(labeled)
        timeline = build_timeline(conn)
    finally:
        conn.close()

    train.to_csv(work_dir / "train.csv", index=False)
    test.to_csv(work_dir / "test.csv", index=False)
    timeline.to_csv(work_dir / "timeline.csv", index=False)

    summary = {
        "n_labeled": len(labeled), "n_train": len(train), "n_test": len(test),
        "n_timeline": len(timeline),
        "train_pos_rate": round(float(train["label"].mean()), 3),
        "test_pos_rate": round(float(test["label"].mean()), 3),
        "test_classes": sorted(test["label"].unique().tolist()),
        "by_source": labeled["label_source"].value_counts().to_dict(),
    }
    return summary


if __name__ == "__main__":
    work = PROJECT_ROOT / "dashboard_export" / "_work"
    s = prepare(work)
    print("BENCHMARK DATA PREPARED:")
    for k, v in s.items():
        print(f"  {k}: {v}")
