"""Phase 3.1 — HONEST cross-validated evaluation of the fusion model.

The core question: now that Phase 1 landed real overlapping (feature,label) pairs,
can a model actually PREDICT tightening — not just memorise the circular
price_derived labels (which are a deterministic function of the price features
the model also sees)?

So we evaluate on TWO datasets, separately:
  * curated_only : NON-circular curated_episode labels (the real test)
  * all          : curated + price_derived (reported WITH a circularity caveat)

Metric: stratified k-fold CV (no calibration split leakage), reporting ROC-AUC,
accuracy, and Brier score, plus n / positive-rate. One model per process
(`--model lightgbm|tabpfn`) because tabpfn-client imports torch and torch+LightGBM
in one process segfaults on macOS.

Run:  python scripts/phase3_fusion_eval.py --model lightgbm [--db PATH]
      python scripts/phase3_fusion_eval.py --model tabpfn
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from modeling import feature_builder  # noqa: E402
from modeling.feature_builder import FEATURE_COLUMNS  # noqa: E402
from modeling.train_fusion_model import _new_estimator  # noqa: E402

warnings.filterwarnings("ignore")
N_SPLITS = 5
SEED = 42


def _cv_scores(model_type, X, y):
    """Stratified k-fold out-of-fold predictions -> pooled metrics."""
    y = np.asarray(y, dtype=int)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    k = min(N_SPLITS, n_pos, n_neg)
    if k < 2:
        return {"error": f"too few of a class for CV (pos={n_pos}, neg={n_neg})"}
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y):
        est = _new_estimator(model_type, SEED)
        est.fit(X.iloc[tr], y[tr])
        oof[te] = est.predict_proba(X.iloc[te])[:, 1]
    return {
        "n": int(len(y)), "positives": n_pos, "pos_rate": round(n_pos / len(y), 3),
        "cv_folds": k,
        "roc_auc": round(float(roc_auc_score(y, oof)), 3),
        "accuracy": round(float(accuracy_score(y, (oof >= 0.5).astype(int))), 3),
        "brier": round(float(brier_score_loss(y, oof)), 3),
    }


def run(model_type, db_path=None):
    conn = db_init.get_connection(db_path)
    try:
        tt = feature_builder.build_training_table(conn)
    finally:
        conn.close()
    if tt.empty:
        print(json.dumps({"error": "no training rows"})); return

    is_curated = tt["label_source"].str.contains("curated", case=False, na=False)
    datasets = {
        "curated_only (NON-circular — the real test)": tt[is_curated],
        "all (curated + price_derived — CIRCULAR caveat)": tt,
    }

    print("=" * 74)
    print(f"PHASE 3.1 FUSION EVAL   model={model_type}   {N_SPLITS}-fold stratified CV")
    print("=" * 74)
    results = {"model": model_type, "datasets": {}}
    for name, df in datasets.items():
        X = df[FEATURE_COLUMNS]
        y = df["label"]
        scores = _cv_scores(model_type, X, y)
        results["datasets"][name] = scores
        print(f"\n[{name}]")
        if "error" in scores:
            print("  ", scores["error"])
        else:
            print(f"  n={scores['n']}  positives={scores['positives']} "
                  f"(pos_rate={scores['pos_rate']})  folds={scores['cv_folds']}")
            print(f"  ROC-AUC={scores['roc_auc']}  accuracy={scores['accuracy']}  "
                  f"Brier={scores['brier']}")
    print("\n" + "=" * 74)
    print("INTERPRETATION: trust the curated_only ROC-AUC as the honest signal. The")
    print("'all' row will look inflated BECAUSE price_derived labels are a function")
    print("of the price features (circular) — reported only for completeness.")
    print("=" * 74)
    print("JSON " + json.dumps(results))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lightgbm", "tabpfn"], required=True)
    ap.add_argument("--db", default=None)
    args = ap.parse_args()
    run(args.model, db_path=args.db)
