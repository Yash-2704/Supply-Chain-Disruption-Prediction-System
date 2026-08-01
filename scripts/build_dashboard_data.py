"""Export adapter — turns our engine's DB + models into the exact CSVs/PNGs the
merged Streamlit dashboard reads. This is the bridge between our SQLite/ML engine
and the (CSV-consuming) dashboard.

Pipeline:
  1. Prepare labelled data + chronological split + scoring timeline (benchmark_data).
  2. Train each of the 4 models in a SEPARATE subprocess (torch+OpenMP segfault guard).
  3. Aggregate:
       dashboard_export/outputs/evaluation_report.csv        (all models' metrics)
       dashboard_export/outputs/{roc,pr,confusion}.png       (from test probabilities)
       dashboard_export/processed/risk_timeline.csv          (primary model, full timeline)
       dashboard_export/processed/risk_scores.csv            (latest week per resource)
       dashboard_export/outputs/shap/feature_importance.csv  (primary tree model)
       dashboard_export/outputs/shap/bar_plot.png
       dashboard_export/processed/resource_drivers.csv       (real per-resource SHAP)
       dashboard_export/processed/{combined,<RES>}_weekly.csv (download conveniences)

Run:  python scripts/build_dashboard_data.py [--skip-train]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from risk_dashboard.config import PROCESSED_DATA_DIR, OUTPUTS_DIR, ID_TO_DISPLAY, RESOURCES  # noqa: E402
from modeling import benchmark_data  # noqa: E402

MODELS = ["lightgbm", "xgboost", "tabpfn", "ft"]
DISPLAY = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "tabpfn": "TabPFN", "ft": "FT-Transformer"}
PRIMARY = "lightgbm"   # drives risk scores + SHAP (tree-based, fast, calibratable)
WORK = PROJECT_ROOT / "dashboard_export" / "_work"
_PALETTE = ["#16a34a", "#0284c7", "#7c3aed", "#d97706", "#ef4444"]


def _train_all():
    for m in MODELS:
        print(f"\n── training {DISPLAY[m]} (subprocess) ──")
        r = subprocess.run([sys.executable, "-m", "modeling.benchmark_train",
                            "--model", m, "--work", str(WORK)],
                           cwd=str(PROJECT_ROOT))
        if r.returncode != 0:
            print(f"  ⚠️  {DISPLAY[m]} failed (exit {r.returncode}) — skipping it.")


def _collect_metrics():
    rows = []
    for m in MODELS:
        p = WORK / f"metrics_{m}.json"
        if p.exists():
            rows.append(json.loads(p.read_text()))
    if not rows:
        raise SystemExit("No model metrics produced — cannot build dashboard data.")
    df = pd.DataFrame(rows)
    cols = ["model", "accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "train_time_s"]
    df = df[[c for c in cols if c in df.columns]].sort_values("f1_score", ascending=False)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUTS_DIR / "evaluation_report.csv", index=False)
    print("\nevaluation_report.csv:\n", df.to_string(index=False))
    return df


def _plots(y_test):
    avail = [(m, WORK / f"testproba_{m}.npy") for m in MODELS if (WORK / f"testproba_{m}.npy").exists()]
    # ROC
    plt.figure(figsize=(7, 6))
    for i, (m, p) in enumerate(avail):
        proba = np.load(p)
        fpr, tpr, _ = roc_curve(y_test, proba)
        from sklearn.metrics import roc_auc_score
        plt.plot(fpr, tpr, color=_PALETTE[i % len(_PALETTE)], lw=2,
                 label=f"{DISPLAY[m]} (AUC={roc_auc_score(y_test, proba):.3f})")
    plt.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Model Comparison"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(OUTPUTS_DIR / "roc_curves.png", dpi=140); plt.close()
    # PR
    plt.figure(figsize=(7, 6))
    for i, (m, p) in enumerate(avail):
        proba = np.load(p)
        prec, rec, _ = precision_recall_curve(y_test, proba)
        from sklearn.metrics import average_precision_score
        plt.plot(rec, prec, color=_PALETTE[i % len(_PALETTE)], lw=2,
                 label=f"{DISPLAY[m]} (AP={average_precision_score(y_test, proba):.3f})")
    plt.axhline(y_test.mean(), ls="--", color="#999", lw=1, label=f"Baseline ({y_test.mean():.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curves"); plt.legend(loc="upper right")
    plt.tight_layout(); plt.savefig(OUTPUTS_DIR / "pr_curves.png", dpi=140); plt.close()
    # Confusion matrices
    n = len(avail)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (m, p) in zip(axes, avail):
        cm = confusion_matrix(y_test, (np.load(p) >= 0.5).astype(int), labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=12)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Risk", "Risk"]); ax.set_yticklabels(["No Risk", "Risk"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(DISPLAY[m])
    fig.suptitle("Confusion Matrices — Model Comparison", fontweight="bold")
    plt.tight_layout(); plt.savefig(OUTPUTS_DIR / "confusion_matrices.png", dpi=140); plt.close()
    print("  wrote roc_curves.png, pr_curves.png, confusion_matrices.png")


def _risk_scores_and_timeline():
    tl = pd.read_csv(WORK / f"timeline_{PRIMARY}.csv")
    tl["resource"] = tl["resource_id"].map(ID_TO_DISPLAY)
    tl = tl.dropna(subset=["resource"])
    out = tl[["week_start", "resource", "risk_score"]].rename(columns={"week_start": "date"})
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROCESSED_DATA_DIR / "risk_timeline.csv", index=False)
    latest = out.sort_values("date").groupby("resource").tail(1)
    latest[["resource", "risk_score"]].to_csv(PROCESSED_DATA_DIR / "risk_scores.csv", index=False)
    print("  wrote risk_timeline.csv + risk_scores.csv (primary:", DISPLAY[PRIMARY], ")")


def _shap_exports():
    shap_dir = OUTPUTS_DIR / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    src = WORK / f"shap_{PRIMARY}.csv"
    if not src.exists():
        print("  (no SHAP for primary model — skipping importance export)"); return
    imp = pd.read_csv(src)
    imp.to_csv(shap_dir / "feature_importance.csv", index=False)
    # bar plot
    imp = imp.sort_values("importance")
    plt.figure(figsize=(8, max(4, len(imp) * 0.35)))
    plt.barh(imp["feature"], imp["importance"], color="#0284c7")
    plt.xlabel("Mean |SHAP value|"); plt.title(f"Feature Importance — {DISPLAY[PRIMARY]}")
    plt.tight_layout(); plt.savefig(shap_dir / "bar_plot.png", dpi=140); plt.close()
    # per-resource drivers
    dsrc = WORK / f"drivers_{PRIMARY}.csv"
    if dsrc.exists():
        d = pd.read_csv(dsrc)
        d["resource"] = d["resource_id"].map(ID_TO_DISPLAY)
        d = d.dropna(subset=["resource"])
        d[["resource", "feature", "contribution", "value"]].to_csv(
            PROCESSED_DATA_DIR / "resource_drivers.csv", index=False)
        print("  wrote feature_importance.csv, bar_plot.png, resource_drivers.csv")


def _download_exports():
    tl = pd.read_csv(WORK / "timeline.csv")
    tl["resource"] = tl["resource_id"].map(ID_TO_DISPLAY)
    tl.to_csv(PROCESSED_DATA_DIR / "combined_weekly.csv", index=False)
    for rid, disp in ID_TO_DISPLAY.items():
        sub = tl[tl["resource_id"] == rid]
        if not sub.empty:
            sub.to_csv(PROCESSED_DATA_DIR / f"{disp}_weekly.csv", index=False)
    print("  wrote combined_weekly.csv + per-resource CSVs")


def run(skip_train=False):
    print("=" * 64)
    print("BUILD DASHBOARD DATA")
    print("=" * 64)
    if not skip_train:
        print("\nStep 1: preparing labelled data + split + timeline …")
        summary = benchmark_data.prepare(WORK)
        print("  ", summary)
        print("\nStep 2: training 4 models (subprocess each) …")
        _train_all()
    print("\nStep 3: aggregating exports …")
    _collect_metrics()
    y_test = pd.read_csv(WORK / "test.csv")["label"].astype(int).values
    _plots(y_test)
    _risk_scores_and_timeline()
    _shap_exports()
    _download_exports()
    print("\n✅ Dashboard data built. Launch with:")
    print("   streamlit run risk_dashboard/app.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true",
                    help="Reuse existing per-model artefacts in _work (aggregate only).")
    args = ap.parse_args()
    run(skip_train=args.skip_train)
