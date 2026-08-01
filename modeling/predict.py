"""Inference + gated SHAP explainability for the fusion model.

Loads the most appropriate model (prefers a 'real' model; falls back to
'synthetic_validation' with an unmistakable flag IN THE RETURN VALUE, not just a
log). SHAP is computed only for a real model that meets the evaluation minimum;
otherwise top_shap_features is None with a stated reason — never a placeholder.
"""
import json
import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from modeling import feature_builder  # noqa: E402
from modeling.feature_builder import FEATURE_COLUMNS  # noqa: E402
from modeling.train_fusion_model import MODELS_DIR, MIN_REAL_EXAMPLES_FOR_EVALUATION  # noqa: E402

logger = logging.getLogger("predict")


def _load_bundle(models_dir: Path, prefer="real", model_type=None):
    """Return (bundle, model_path). Prefer a real model; else synthetic.

    model_type=None loads the default (lightgbm) filenames; pass 'tabpfn' (etc.)
    to load that learner's suffixed file — used to compare learners head-to-head.
    """
    models_dir = Path(models_dir)
    order = ["real", "synthetic_validation"] if prefer == "real" \
        else ["synthetic_validation", "real"]
    suffix = "" if model_type in (None, "lightgbm") else f"_{model_type}"
    for mode in order:
        path = models_dir / f"fusion_model_{mode}{suffix}.pkl"
        if path.exists():
            with open(path, "rb") as fh:
                return pickle.load(fh), path
    return None, None


def _top_shap(model, X, k=5):
    """Top-k features by |SHAP| for the single row X. Returns list of dicts."""
    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):          # older shap: [class0, class1]
        sv = sv[1]
    vals = np.asarray(sv)[0]          # single row
    order = np.argsort(np.abs(vals))[::-1][:k]
    return [{"feature": FEATURE_COLUMNS[i],
             "value": (None if pd.isna(X.iloc[0, i]) else float(X.iloc[0, i])),
             "contribution": float(vals[i])} for i in order]


def predict_resource(resource_id: str, week_start: str, horizon: str,
                     db_path=None, models_dir: Path = MODELS_DIR,
                     prefer="real", model_type=None) -> dict:
    """Predict tightening probability for (resource, week, horizon).

    The returned dict always carries `data_mode` and `is_real_data_model` so a
    caller can programmatically detect a non-real model without reading logs.
    `model_type` selects which learner's saved model to load (None = default
    lightgbm; 'tabpfn' loads the TabPFN model).
    """
    bundle, model_path = _load_bundle(models_dir, prefer=prefer, model_type=model_type)
    if bundle is None:
        return {"error": "no_model_available", "is_real_data_model": False,
                "data_mode": None, "calibrated_probability": None}

    conn = db_init.get_connection(db_path)
    try:
        run_row = conn.execute(
            "SELECT run_id, data_mode, n_examples FROM training_run WHERE model_path = ? "
            "ORDER BY run_id DESC LIMIT 1", (str(model_path),)).fetchone()
        run_id = run_row[0] if run_row else bundle.get("run_id")
        data_mode = (run_row[1] if run_row else bundle["data_mode"])
        n_examples = (run_row[2] if run_row else bundle["n_examples"])
        is_real = (data_mode == "real")

        feats = feature_builder.build_feature_row(resource_id, week_start, conn)
        X = pd.DataFrame([feats])[FEATURE_COLUMNS]

        raw = float(bundle["model"].predict_proba(X)[:, 1][0])
        cal = float(np.clip(bundle["calibrator"].predict([raw])[0], 0.0, 1.0))

        # Gate SHAP: real model, meets evaluation minimum, AND tree-based
        # (TreeExplainer only supports trees; TabPFN is a transformer).
        model_type = bundle.get("model_type", "lightgbm")
        top_shap, shap_reason = None, None
        if not is_real:
            shap_reason = "gated: synthetic_validation model"
        elif model_type != "lightgbm":
            shap_reason = f"gated: non-tree model ({model_type}) — TreeExplainer inapplicable"
        elif n_examples < MIN_REAL_EXAMPLES_FOR_EVALUATION:
            shap_reason = (f"gated: n_examples={n_examples} < "
                           f"MIN_REAL_EXAMPLES_FOR_EVALUATION={MIN_REAL_EXAMPLES_FOR_EVALUATION}")
        else:
            top_shap = _top_shap(bundle["model"], X)

        predicted_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO prediction (run_id, resource_id, week_start, horizon, "
            "raw_score, calibrated_probability, top_shap_features, predicted_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (run_id, resource_id, week_start, horizon, raw, cal,
             json.dumps(top_shap) if top_shap is not None else None, predicted_at),
        )
        conn.commit()

        result = {
            "resource_id": resource_id, "week_start": week_start, "horizon": horizon,
            "raw_score": raw, "calibrated_probability": cal,
            "data_mode": data_mode, "is_real_data_model": is_real,
            "model_type": model_type,
            "run_id": run_id, "n_examples": n_examples,
            "top_shap_features": top_shap, "shap_reason": shap_reason,
        }
        if not is_real:
            result["warning"] = ("PIPELINE-VALIDATION MODEL (synthetic data) — this "
                                 "prediction does NOT reflect real semiconductor patterns.")
        return result
    finally:
        conn.close()


if __name__ == "__main__":
    import sys as _sys
    args = _sys.argv[1:]
    rid = args[0] if args else "dram"
    wk = args[1] if len(args) > 1 else "2026-06-29"
    hz = args[2] if len(args) > 2 else "1m"
    print(json.dumps(predict_resource(rid, wk, hz), indent=2))
