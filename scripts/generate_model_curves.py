"""Generate one training/behaviour graph per benchmarked model for the report.

Reuses the real chronological train/test split already written by the last
benchmark run (dashboard_export/_work/{train,test}.csv) — same rows, same
FEATURE_COLUMNS, same fillna(0.0) convention as modeling/benchmark_train.py.

  XGBoost         -> train/test log-loss vs boosting round (real eval_set)
  LightGBM        -> train/test log-loss vs boosting round (real eval_set)
  TabPFN          -> reliability diagram + predicted-probability histogram
                      (TabPFN has no gradient training loop to plot — this is
                      the closest honest analogue: how well its probabilities
                      are calibrated on the held-out set)
  FT-Transformer  -> real per-epoch train/val BCE loss (early-stopping run)

Each is saved to latex_source/images/model_<name>_curve.png at report width.
Run:  .venv/bin/python scripts/generate_model_curves.py [--model xgboost|lightgbm|tabpfn|ft|all]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modeling.feature_builder import FEATURE_COLUMNS  # noqa: E402

WORK = PROJECT_ROOT / "dashboard_export" / "_work"
OUT_DIR = PROJECT_ROOT / "latex_source" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

# Report palette (matches the dashboard's model colours where practical).
C_TRAIN = "#0284c7"
C_TEST = "#ef4444"
C_BAR = "#4f46e5"


def _load_xy():
    train = pd.read_csv(WORK / "train.csv")
    test = pd.read_csv(WORK / "test.csv")
    Xtr = train[FEATURE_COLUMNS].fillna(0.0).values.astype(float)
    ytr = train["label"].astype(int).values
    Xte = test[FEATURE_COLUMNS].fillna(0.0).values.astype(float)
    yte = test["label"].astype(int).values
    return Xtr, ytr, Xte, yte


def gen_xgboost():
    from xgboost import XGBClassifier
    Xtr, ytr, Xte, yte = _load_xy()
    pos, neg = int((ytr == 1).sum()), int((ytr == 0).sum())
    m = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                      random_state=SEED, scale_pos_weight=neg / max(pos, 1))
    m.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xte, yte)], verbose=False)
    ev = m.evals_result()
    train_ll = ev["validation_0"]["logloss"]
    test_ll = ev["validation_1"]["logloss"]
    rounds = np.arange(1, len(train_ll) + 1)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(rounds, train_ll, color=C_TRAIN, lw=2, label="Train log-loss")
    ax.plot(rounds, test_ll, color=C_TEST, lw=2, label="Test log-loss")
    best = int(np.argmin(test_ll))
    ax.axvline(best + 1, color="#94a3b8", ls="--", lw=1)
    ax.annotate(f"best test round {best+1}", (best + 1, test_ll[best]),
               textcoords="offset points", xytext=(8, 12), fontsize=9, color="#475569")
    ax.set_xlabel("Boosting round"); ax.set_ylabel("Log-loss")
    ax.set_title("XGBoost — training curve (200 boosting rounds)")
    ax.legend(loc="upper right"); ax.grid(alpha=0.25)
    fig.tight_layout()
    out = OUT_DIR / "model_xgboost_curve.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


def gen_lightgbm():
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    Xtr, ytr, Xte, yte = _load_xy()
    evals_result = {}
    m = LGBMClassifier(n_estimators=100, random_state=SEED, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xte, yte)], eval_names=["train", "test"],
         eval_metric="binary_logloss",
         callbacks=[lgb.record_evaluation(evals_result), lgb.log_evaluation(period=0)])
    train_ll = evals_result["train"]["binary_logloss"]
    test_ll = evals_result["test"]["binary_logloss"]
    rounds = np.arange(1, len(train_ll) + 1)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(rounds, train_ll, color=C_TRAIN, lw=2, label="Train log-loss")
    ax.plot(rounds, test_ll, color=C_TEST, lw=2, label="Test log-loss")
    best = int(np.argmin(test_ll))
    ax.axvline(best + 1, color="#94a3b8", ls="--", lw=1)
    ax.annotate(f"best test round {best+1}", (best + 1, test_ll[best]),
               textcoords="offset points", xytext=(8, 12), fontsize=9, color="#475569")
    ax.set_xlabel("Boosting round"); ax.set_ylabel("Log-loss")
    ax.set_title("LightGBM — training curve (100 boosting rounds)")
    ax.legend(loc="upper right"); ax.grid(alpha=0.25)
    fig.tight_layout()
    out = OUT_DIR / "model_lightgbm_curve.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


def gen_tabpfn():
    """TabPFN performs in-context inference from a single forward pass — there
    is no gradient descent and therefore no loss curve to plot. The closest
    honest analogue is how well-calibrated its predicted probabilities are on
    the held-out set, so we plot a reliability diagram alongside the score
    distribution, reusing the REAL saved test-set predictions from the last
    benchmark run (testproba_tabpfn.npy) rather than retraining (a hosted
    TabPFN run takes several minutes and gives point-identical output)."""
    proba_path = WORK / "testproba_tabpfn.npy"
    test = pd.read_csv(WORK / "test.csv")
    yte = test["label"].astype(int).values
    proba = np.load(proba_path)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))

    # Reliability diagram: 5 probability bins (data is small — 138 test rows).
    bins = np.linspace(0, 1, 6)
    bin_idx = np.digitize(proba, bins[1:-1])
    xs, ys, ns = [], [], []
    for b in range(len(bins) - 1):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        xs.append(proba[mask].mean())
        ys.append(yte[mask].mean())
        ns.append(int(mask.sum()))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1, label="Perfect calibration")
    ax.plot(xs, ys, "o-", color=C_BAR, lw=2, markersize=7, label="TabPFN (test set)")
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(6, -10), fontsize=8, color="#475569")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed tightening rate")
    ax.set_title("Reliability diagram"); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.25)

    ax2 = axes[1]
    ax2.hist(proba, bins=15, color=C_BAR, alpha=0.85, edgecolor="white")
    ax2.set_xlabel("Predicted probability"); ax2.set_ylabel("Count (test weeks)")
    ax2.set_title("Predicted-probability distribution")
    ax2.grid(alpha=0.25)

    fig.suptitle("TabPFN — calibration behaviour on the held-out set (no gradient training loop)",
                fontsize=10.5, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "model_tabpfn_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def gen_ft():
    from modeling.ft_transformer import FTTransformerModel, _HAS_FT
    if not _HAS_FT:
        print("torch/tab_transformer_pytorch unavailable — cannot plot a real FT-Transformer curve.")
        return
    Xtr, ytr, Xte, yte = _load_xy()
    m = FTTransformerModel()
    m.fit(Xtr, ytr)
    h = m.history
    if not h["epoch"]:
        print("no epoch history recorded (fallback path?) — skipping.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(h["epoch"], h["train_loss"], color=C_TRAIN, lw=2, marker="o", ms=4, label="Train loss (mean per epoch)")
    ax.plot(h["epoch"], h["val_loss"], color=C_TEST, lw=2, marker="o", ms=4, label="Validation loss")
    best = int(np.argmin(h["val_loss"]))
    ax.axvline(h["epoch"][best], color="#94a3b8", ls="--", lw=1)
    ax.annotate(f"best epoch {h['epoch'][best]}", (h["epoch"][best], h["val_loss"][best]),
               textcoords="offset points", xytext=(8, 12), fontsize=9, color="#475569")
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE loss")
    n_ep = len(h["epoch"])
    stopped = ", early-stopped" if n_ep < 30 else ""
    ax.set_title(f"FT-Transformer — training curve ({n_ep} epochs run{stopped})")
    ax.legend(loc="upper right"); ax.grid(alpha=0.25)
    fig.tight_layout()
    out = OUT_DIR / "model_ft_curve.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


GENERATORS = {"xgboost": gen_xgboost, "lightgbm": gen_lightgbm, "tabpfn": gen_tabpfn, "ft": gen_ft}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(GENERATORS) + ["all"], default="all")
    args = ap.parse_args()
    targets = list(GENERATORS) if args.model == "all" else [args.model]
    for t in targets:
        print(f"\n=== {t} ===")
        GENERATORS[t]()
