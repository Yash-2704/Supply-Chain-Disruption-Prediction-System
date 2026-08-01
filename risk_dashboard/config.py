"""Configuration for the merged risk dashboard.

The dashboard (ported from the teammate's Streamlit app) is a pure CONSUMER of a
handful of CSVs/PNGs produced by our engine's export adapter
(scripts/build_dashboard_data.py). This module centralises every constant the
dashboard needs, adapted to OUR four semiconductor resources and four models.

Nothing here is fabricated data: DISRUPTION_EVENTS mirror our hand-authored
curated episodes (data/curated_episodes.json). The `severity` field is an
approximate EDITORIAL intensity rating consistent with those episodes already
being hand-authored domain knowledge — it is display-only and never enters a model.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The adapter writes here; the dashboard reads from here.
EXPORT_ROOT = PROJECT_ROOT / "dashboard_export"
PROCESSED_DATA_DIR = EXPORT_ROOT / "processed"
OUTPUTS_DIR = EXPORT_ROOT / "outputs"

# ── Resources ───────────────────────────────────────────────
# Display names shown in the UI; RESOURCE_ID maps them to our DB resource_ids.
RESOURCES = ["DRAM", "NAND", "HBM", "GPU"]
RESOURCE_ID = {"DRAM": "dram", "NAND": "nand", "HBM": "hbm", "GPU": "gpu"}
ID_TO_DISPLAY = {v: k for k, v in RESOURCE_ID.items()}

# Emoji were semantically wrong (🧠 for NAND flash, 🎮 for AI GPUs). v2 replaces
# them with an authored SVG icon set (see app.RESOURCE_SVG) + monogram badges.
RESOURCE_MONOGRAM = {"DRAM": "DR", "NAND": "ND", "HBM": "HB", "GPU": "GP"}
RESOURCE_COLORS = {"DRAM": "#0284c7", "NAND": "#7c3aed", "HBM": "#d97706", "GPU": "#059669"}

# One-line descriptors shown under each resource (domain grounding, not chrome).
RESOURCE_TAGLINE = {
    "DRAM": "Server / DDR5 memory",
    "NAND": "Flash storage",
    "HBM": "High-Bandwidth Memory",
    "GPU": "AI accelerators",
}

# ── Human-readable feature labels + units ───────────────────
# Raw model feature name -> (pretty label, unit/explanation). Fixes "Hhi Country"
# and unexplained columns surfaced in the design critique.
FEATURE_LABELS = {
    "price_latest":                ("Price level", "latest weekly proxy"),
    "price_trend_4w":              ("Price trend (4w)", "4-week slope"),
    "price_zscore_8w":             ("Price z-score (8w)", "vs 8-week baseline"),
    "forecast_point_1m":           ("Forecast (1m)", "1-month point"),
    "forecast_interval_width_1m":  ("Forecast uncertainty (1m)", "interval width"),
    "forecast_disagreement_1m":    ("Forecast disagreement (1m)", "model spread"),
    "forecast_unstable_flag":      ("Forecast unstable flag", "0/1"),
    "signal_severity_mean_4w":     ("News severity (4w)", "LLM 1–10 mean"),
    "signal_count_4w":             ("News volume (4w)", "story count"),
    "hhi_country":                 ("Supplier concentration (HHI)", "0–1 index"),
}


def pretty_feature(name: str) -> str:
    """Human label for a raw feature name; falls back to Title Case."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name][0]
    return name.replace("_", " ").title()


def feature_unit(name: str) -> str:
    """Unit/explanation for a feature, or empty string."""
    return FEATURE_LABELS.get(name, ("", ""))[1]

# ── Models (four benchmarked learners) ──────────────────────
MODEL_COLORS = {
    "LightGBM": "#16a34a",
    "XGBoost": "#0284c7",
    "FT-Transformer": "#7c3aed",
    "TabPFN": "#d97706",
}

# ── Risk thresholds (Safe / Medium / High / Critical) ───────
RISK_THRESHOLDS = {"low": 0.3, "medium": 0.6, "high": 0.8}

# ── Titles ──────────────────────────────────────────────────
DASHBOARD_TITLE = "Supply Chain Risk Intelligence Framework"
DASHBOARD_SUBTITLE = "Case Study: Semiconductor Industry (DRAM · NAND · HBM · GPU)"

# ── Disruption / tightening episodes (from data/curated_episodes.json) ──
# Keys used by the dashboard: start, end, description, resources, severity.
# `severity` is an approximate curated intensity (display-only, see module docstring).
DISRUPTION_EVENTS = [
    {
        "start": "2023-05-01", "end": "2025-06-30",
        "resources": ["GPU"], "severity": 0.90,
        "description": "AI-accelerator allocation constraint",
    },
    {
        "start": "2024-01-01", "end": "2025-12-31",
        "resources": ["HBM"], "severity": 0.95,
        "description": "HBM3/HBM3E sold-out tightening",
    },
    {
        "start": "2024-03-01", "end": "2025-06-30",
        "resources": ["DRAM"], "severity": 0.85,
        "description": "AI-driven DRAM tightening",
    },
    {
        "start": "2024-10-01", "end": "2025-06-30",
        "resources": ["NAND"], "severity": 0.70,
        "description": "NAND supply-cut recovery tightening",
    },
]

# ── Historical negative (non-tightening) periods, for reference ─────
NEGATIVE_PERIODS = [
    {"start": "2023-01-01", "end": "2023-09-30", "resources": ["DRAM"],
     "description": "2023 DRAM oversupply / downturn"},
    {"start": "2023-01-01", "end": "2023-09-30", "resources": ["NAND"],
     "description": "2023 NAND glut"},
]
