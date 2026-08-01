"""Fusion model training: real-vs-synthetic sufficiency gate, LightGBM +
isotonic calibration, and mandatory provenance metadata.

If enough real, non-trivial overlapping (feature,label) pairs exist, trains a
'real' model. Otherwise trains a clearly-labeled 'synthetic_validation' model
that exercises every code path but must NEVER be mistaken for a real one — the
filename and the training_run row both encode data_mode.
"""
import json
import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from modeling import feature_builder  # noqa: E402
from modeling.feature_builder import FEATURE_COLUMNS  # noqa: E402

MODEL_SCHEMA_PATH = PROJECT_ROOT / "schema_model.sql"
MODELS_DIR = PROJECT_ROOT / "models"

# ~5 examples/feature is already a bare floor for a 10-feature gradient-boosted
# tree that also needs a calibration split; below this we refuse to call a model
# 'real'. (Today's only overlapping labels are also circular price_derived ones,
# reinforcing the refusal.)
MIN_REAL_EXAMPLES_FOR_TRAINING = 50

# A held-out calibration/evaluation metric on fewer than this many examples is
# statistically meaningless; below it we report insufficiency, and SHAP is gated.
MIN_REAL_EXAMPLES_FOR_EVALUATION = 30

RANDOM_SEED = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train_fusion_model")


def init_model_schema(conn):
    conn.executescript(MODEL_SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()


def decide_data_mode(n_real: int) -> str:
    return "real" if n_real >= MIN_REAL_EXAMPLES_FOR_TRAINING else "synthetic_validation"


def generate_synthetic_dataset(n=300, seed=RANDOM_SEED) -> pd.DataFrame:
    """Clearly-labeled SYNTHETIC data for pipeline validation ONLY.

    Feature vectors are sampled from plausible distributions (with injected NaNs
    to exercise missing handling); labels come from a DOCUMENTED logistic rule
    over a few features. This encodes NO real semiconductor semantics.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        price_latest = float(rng.normal(100, 25))
        zscore = float(rng.normal(0, 1.2))
        disagree = float(abs(rng.normal(12, 12)))
        unstable = float(rng.random() < 0.3)
        has_signal = rng.random() < 0.6
        sev_mean = float(rng.uniform(1, 10)) if has_signal else np.nan
        sev_count = float(rng.integers(0, 8)) if has_signal else np.nan
        # DOCUMENTED synthetic labeling rule (NOT real-world semantics):
        logit = 1.3 * zscore + 0.04 * disagree + 0.9 * unstable - 1.1
        prob = 1.0 / (1.0 + np.exp(-logit))
        label = int(rng.random() < prob)
        rows.append({
            "price_latest": price_latest,
            "price_trend_4w": float(rng.normal(0, 6)),
            "price_zscore_8w": zscore,
            "forecast_point_1m": price_latest + float(rng.normal(0, 12)),
            "forecast_interval_width_1m": float(abs(rng.normal(25, 10))),
            "forecast_disagreement_1m": disagree,
            "forecast_unstable_flag": unstable,
            "signal_severity_mean_4w": sev_mean,
            "signal_count_4w": sev_count,
            "hhi_country": float(rng.uniform(0.3, 0.85)),
            "label": label, "label_source": "synthetic",
            "resource_id": None, "week_start": None,
        })
    return pd.DataFrame(rows)


# Selectable fusion learners. LightGBM (default, tree — SHAP-explainable) and
# TabPFN (pretrained tabular transformer that excels in the tiny-data regime the
# real overlap lives in). Both expose the sklearn fit/predict_proba API, so the
# rest of the pipeline (calibration, inference) is model-agnostic; only SHAP is
# tree-specific and is gated to lightgbm in predict.py.
MODEL_TYPES = ("lightgbm", "tabpfn")


def _new_tabpfn():
    """A TabPFN classifier. Prefers the HOSTED client (`tabpfn-client`) when a
    TABPFN_TOKEN is set — Prior Labs runs inference server-side, avoiding the
    local v2 gated-weights + browser-license flow that can't complete headlessly.
    Falls back to the local `tabpfn` package otherwise.

    NOTE: the hosted path sends feature rows to Prior Labs' API. Fine for this
    project's public-data-derived features; flagged so it's a conscious choice.
    Both paths expose the sklearn fit/predict_proba API and handle NaN + small n.
    """
    import os
    token = os.environ.get("TABPFN_TOKEN")
    if token:
        try:
            import tabpfn_client
            tabpfn_client.set_access_token(token)
            return tabpfn_client.TabPFNClassifier()
        except ImportError:
            logger.info("tabpfn-client not installed; trying local tabpfn.")
    from tabpfn import TabPFNClassifier  # local (gated weights) fallback
    return TabPFNClassifier()


def _new_estimator(model_type, seed):
    """Construct an unfitted estimator for `model_type`. Raises on unknown type
    or if the (optional) TabPFN dependency is unavailable."""
    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=seed, n_estimators=100, verbose=-1)
    if model_type == "tabpfn":
        return _new_tabpfn()
    raise ValueError(f"unknown model_type {model_type!r}; expected one of {MODEL_TYPES}")


def _fit_model_and_calibrator(df, seed, model_type="lightgbm"):
    """Fit the chosen learner + isotonic calibration. Returns (model, calibrator)."""
    X, y = df[FEATURE_COLUMNS], df["label"].astype(int)
    # Hold out a calibration split when there's enough data; else reuse train.
    if len(df) >= 20 and y.nunique() > 1:
        X_tr, X_cal, y_tr, y_cal = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y)
    else:
        X_tr, X_cal, y_tr, y_cal = X, X, y, y
    model = _new_estimator(model_type, seed)
    model.fit(X_tr, y_tr)
    raw_cal = model.predict_proba(X_cal)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_cal, y_cal.values)
    return model, calibrator


def _model_filename(data_mode: str, model_type: str) -> str:
    """Default (lightgbm) keeps the original name for backward compatibility;
    other learners get a suffix so both can coexist for head-to-head comparison."""
    return (f"fusion_model_{data_mode}.pkl" if model_type == "lightgbm"
            else f"fusion_model_{data_mode}_{model_type}.pkl")


def train_and_save(df: pd.DataFrame, data_mode: str, conn, seed=RANDOM_SEED,
                   models_dir: Path = MODELS_DIR, notes: str = "",
                   model_type: str = "lightgbm") -> int:
    """Train, calibrate, persist the model + provenance; return run_id."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model, calibrator = _fit_model_and_calibrator(df, seed, model_type=model_type)

    n_examples = len(df)
    n_positive = int(df["label"].sum())
    if data_mode == "real":
        weeks = df["week_start"].dropna()
        date_start = str(weeks.min()) if len(weeks) else None
        date_end = str(weeks.max()) if len(weeks) else None
    else:
        date_start = date_end = None  # synthetic has no real dates

    model_path = models_dir / _model_filename(data_mode, model_type)
    run_at = datetime.now(timezone.utc).isoformat()
    notes = (notes + f" | model_type={model_type}").strip(" |")

    cur = conn.execute(
        "INSERT INTO training_run (run_at, data_mode, n_examples, n_positive, "
        "date_range_start, date_range_end, random_seed, model_path, "
        "calibration_applied, notes) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (run_at, data_mode, n_examples, n_positive, date_start, date_end,
         seed, str(model_path), 1, notes),
    )
    conn.commit()
    run_id = cur.lastrowid

    bundle = {
        "model": model, "calibrator": calibrator,
        "feature_columns": FEATURE_COLUMNS, "data_mode": data_mode,
        "model_type": model_type,
        "random_seed": seed, "n_examples": n_examples, "run_id": run_id,
    }
    with open(model_path, "wb") as fh:
        pickle.dump(bundle, fh)
    return run_id


def run(db_path=None, models_dir: Path = MODELS_DIR, model_type: str = "lightgbm"):
    conn = db_init.get_connection(db_path)
    try:
        init_model_schema(conn)
        real_df = feature_builder.build_training_table(conn)
        n_real = len(real_df)
        mode = decide_data_mode(n_real)

        print("=" * 70)
        print(f"REAL overlapping (feature,label) pairs found: {n_real}")
        if n_real:
            print("  by source:", real_df["label_source"].value_counts().to_dict())
        print(f"MIN_REAL_EXAMPLES_FOR_TRAINING = {MIN_REAL_EXAMPLES_FOR_TRAINING}")
        print(f"MODEL_TYPE = {model_type}")
        print(f"DECISION: data_mode = {mode.upper()}")

        if mode == "real":
            run_id = train_and_save(real_df, "real", conn, models_dir=models_dir,
                                    notes=f"real overlap n={n_real}", model_type=model_type)
            print(f"Trained REAL model ({model_type}) on {n_real} examples. run_id={run_id}")
        else:
            print("!" * 70)
            print("!! SYNTHETIC-VALIDATION RUN — NOT A USABLE PREDICTIVE MODEL !!")
            print(f"!! Real overlap ({n_real}) is below the training floor "
                  f"({MIN_REAL_EXAMPLES_FOR_TRAINING}); the only overlapping real")
            print("!! labels are circular price_derived ones. Pipeline is exercised")
            print("!! end-to-end on synthetic data purely to validate correctness.")
            print("!" * 70)
            syn = generate_synthetic_dataset(seed=RANDOM_SEED)
            run_id = train_and_save(
                syn, "synthetic_validation", conn, models_dir=models_dir,
                notes=(f"pipeline validation only; real overlap={n_real} "
                       f"(all circular price_derived), 0 independent curated overlap"),
                model_type=model_type)
            print(f"Trained SYNTHETIC_VALIDATION model ({model_type}) on {len(syn)} "
                  f"synthetic examples. run_id={run_id}")
        print("=" * 70)
        return mode, n_real
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train the fusion model (LightGBM or TabPFN).")
    ap.add_argument("--model-type", default="lightgbm", choices=MODEL_TYPES, dest="model_type",
                    help="Learner: lightgbm (default, SHAP-explainable) or tabpfn (tiny-data).")
    args = ap.parse_args()
    run(model_type=args.model_type)
