"""Train ONE benchmark model and emit its artefacts. Run per-model in a SEPARATE
process (torch + LightGBM/XGBoost in one process segfault on macOS via libomp).

Usage:  python -m modeling.benchmark_train --model {lightgbm|xgboost|tabpfn|ft} --work DIR

Reads work/{train,test,timeline}.csv; writes into work/:
  metrics_{m}.json      accuracy/precision/recall/f1/roc_auc/pr_auc/train_time
  testproba_{m}.npy     class-1 probabilities on the test set (for ROC/PR/confusion)
  timeline_{m}.csv      resource_id, week_start, risk_score  (full timeline scoring)
  shap_{m}.csv          feature, importance  (tree models only)
  drivers_{m}.csv       resource, feature, contribution, value  (tree models only)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modeling.feature_builder import FEATURE_COLUMNS  # noqa: E402

MODEL_DISPLAY = {"lightgbm": "LightGBM", "xgboost": "XGBoost",
                 "tabpfn": "TabPFN", "ft": "FT-Transformer"}
TREE_MODELS = {"lightgbm", "xgboost"}
SEED = 42


def _fit(model_name, X_train, y_train):
    """Return (fitted_estimator, proba_fn). proba_fn(X)->P(class 1)."""
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(n_estimators=100, random_state=SEED, verbose=-1)
        m.fit(X_train, y_train)
        return m, (lambda X: m.predict_proba(X)[:, 1])
    if model_name == "xgboost":
        from xgboost import XGBClassifier
        pos = int((y_train == 1).sum()); neg = int((y_train == 0).sum())
        m = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                          random_state=SEED, scale_pos_weight=neg / max(pos, 1))
        m.fit(X_train, y_train)
        return m, (lambda X: m.predict_proba(X)[:, 1])
    if model_name == "tabpfn":
        from modeling.train_fusion_model import _new_tabpfn
        m = _new_tabpfn()
        m.fit(X_train, y_train)
        return m, (lambda X: m.predict_proba(X)[:, 1])
    if model_name == "ft":
        from modeling.ft_transformer import FTTransformerModel
        m = FTTransformerModel()
        m.fit(X_train, y_train)
        return m, (lambda X: m.predict_proba1(X))
    raise ValueError(f"unknown model {model_name!r}")


def _metrics(y_true, proba):
    pred = (proba >= 0.5).astype(int)
    out = {"accuracy": accuracy_score(y_true, pred),
           "precision": precision_score(y_true, pred, zero_division=0),
           "recall": recall_score(y_true, pred, zero_division=0),
           "f1_score": f1_score(y_true, pred, zero_division=0)}
    # roc_auc / pr_auc need both classes present
    try:
        out["roc_auc"] = roc_auc_score(y_true, proba)
        out["pr_auc"] = average_precision_score(y_true, proba)
    except ValueError:
        out["roc_auc"] = float("nan"); out["pr_auc"] = float("nan")
    return out


def _tree_shap(model, model_name, X, feature_names):
    """(mean_abs per feature, raw shap matrix) for a tree model, class-1 aware."""
    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):          # [class0, class1]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # (n, features, classes)
        sv = sv[:, :, 1]
    return np.abs(sv).mean(axis=0), sv


def run(model_name, work: Path):
    work = Path(work)
    disp = MODEL_DISPLAY[model_name]
    train = pd.read_csv(work / "train.csv")
    test = pd.read_csv(work / "test.csv")
    timeline = pd.read_csv(work / "timeline.csv")

    Xtr = train[FEATURE_COLUMNS].fillna(0.0).values.astype(float)
    ytr = train["label"].astype(int).values
    Xte = test[FEATURE_COLUMNS].fillna(0.0).values.astype(float)
    yte = test["label"].astype(int).values
    Xtl = timeline[FEATURE_COLUMNS].fillna(0.0).values.astype(float)

    t0 = time.time()
    model, proba_fn = _fit(model_name, Xtr, ytr)
    train_time = time.time() - t0

    test_proba = np.asarray(proba_fn(Xte), dtype=float)
    metrics = _metrics(yte, test_proba)
    metrics.update({"model": disp, "train_time_s": round(train_time, 3)})

    tl_proba = np.asarray(proba_fn(Xtl), dtype=float)
    tl_out = timeline[["resource_id", "week_start"]].copy()
    tl_out["risk_score"] = np.clip(tl_proba, 0.0, 1.0)

    # write per-model artefacts
    (work / f"metrics_{model_name}.json").write_text(json.dumps(metrics, indent=2))
    np.save(work / f"testproba_{model_name}.npy", test_proba)
    tl_out.to_csv(work / f"timeline_{model_name}.csv", index=False)

    if model_name in TREE_MODELS:
        mean_abs, _ = _tree_shap(model, model_name, Xtr, FEATURE_COLUMNS)
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": mean_abs}).to_csv(
            work / f"shap_{model_name}.csv", index=False)
        # per-resource drivers: SHAP on each resource's latest timeline row
        latest = timeline.sort_values("week_start").groupby("resource_id").tail(1)
        Xl = latest[FEATURE_COLUMNS].fillna(0.0).values.astype(float)
        _, sv = _tree_shap(model, model_name, Xl, FEATURE_COLUMNS)
        rows = []
        for i, (_, r) in enumerate(latest.iterrows()):
            for j, feat in enumerate(FEATURE_COLUMNS):
                rows.append({"resource_id": r["resource_id"], "feature": feat,
                             "contribution": float(sv[i, j]),
                             "value": float(r[feat]) if pd.notna(r[feat]) else None})
        pd.DataFrame(rows).to_csv(work / f"drivers_{model_name}.csv", index=False)

    print(f"[{disp}] f1={metrics['f1_score']:.3f} roc_auc={metrics['roc_auc']:.3f} "
          f"pr_auc={metrics['pr_auc']:.3f} time={train_time:.1f}s  (artefacts written)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_DISPLAY))
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    run(a.model, Path(a.work))
