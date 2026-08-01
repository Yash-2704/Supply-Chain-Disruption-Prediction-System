"""
Supply Chain Risk Intelligence Framework — Streamlit Dashboard
==============================================================
Premium multi-page dashboard with:
  1. Risk Overview — live risk scores, timeline, key drivers
  2. Model Performance — comparison table, ROC/PR curves, confusion matrices
  3. Explainability — SHAP plots, feature importance, prediction breakdown
  4. Historical Case Studies — annotated disruption timelines

Works fully without trained models using built-in sample data.
"""

import sys
import os
import re
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Import merged-dashboard config (our 4 resources + 4 models) ──
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))
from risk_dashboard_v2.config import (
    RESOURCES, RESOURCE_ID, ID_TO_DISPLAY, RESOURCE_MONOGRAM, RESOURCE_COLORS,
    RESOURCE_TAGLINE, MODEL_COLORS, DISRUPTION_EVENTS, RISK_THRESHOLDS,
    DASHBOARD_TITLE, DASHBOARD_SUBTITLE, PROCESSED_DATA_DIR, OUTPUTS_DIR,
    pretty_feature, feature_unit,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AUTHORED ICON SYSTEM (replaces semantically-wrong emoji)
#  One consistent line style: 24×24 viewBox, stroke=currentColor, 1.75 width.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _svg(body: str, size: int = 24) -> str:
    return (
        f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" '
        f'stroke="currentColor" stroke-width="1.75" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true" focusable="false">{body}</svg>'
    )


# Resource icons — a memory-chip family, differentiated per resource.
_CHIP_PINS = (
    '<path d="M9 3v2M15 3v2M9 19v2M15 19v2M3 9h2M3 15h2M19 9h2M19 15h2"/>'
)
RESOURCE_SVG = {
    # DRAM — DIMM: chip with a single inner die line.
    "DRAM": _svg('<rect x="5" y="5" width="14" height="14" rx="2"/>'
                 '<path d="M9 9h6"/>' + _CHIP_PINS),
    # NAND — stacked storage cells (two inner lines = layers).
    "NAND": _svg('<rect x="5" y="5" width="14" height="14" rx="2"/>'
                 '<path d="M9 10h6M9 14h6"/>' + _CHIP_PINS),
    # HBM — stacked die (3D cube hint) = high-bandwidth stack.
    "HBM": _svg('<rect x="5" y="8" width="12" height="11" rx="1.5"/>'
                '<path d="M7 8l2-3h10l-2 3M17 8l2-3v11l-2 3"/>'),
    # GPU — accelerator board with a fan circle.
    "GPU": _svg('<rect x="3" y="6" width="18" height="12" rx="2"/>'
                '<circle cx="9" cy="12" r="3"/><path d="M15 10h3M15 14h3"/>'),
}

# Navigation icons.
NAV_SVG = {
    "overview":  _svg('<path d="M3 13a9 9 0 0 1 18 0"/><path d="M12 13l4-3"/>'
                      '<circle cx="12" cy="13" r="1"/>'),      # gauge
    "models":    _svg('<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>'),  # bars
    "explain":   _svg('<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/>'),  # lens
    "cases":     _svg('<path d="M4 5a2 2 0 0 1 2-2h11v16H6a2 2 0 0 0-2 2z"/>'
                      '<path d="M17 3v16"/>'),                 # book
}


def resource_badge(res: str, size: int = 36) -> str:
    """A colored square badge holding the resource's authored line icon — the
    accessible stand-in for the old emoji (role=img + aria-label)."""
    color = RESOURCE_COLORS.get(res, "#4f46e5")
    icon = RESOURCE_SVG.get(res, "")
    return (
        f'<span class="res-badge" role="img" aria-label="{res}" '
        f'style="--rc:{color}; width:{size}px; height:{size}px;">{icon}</span>'
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE CONFIG & PREMIUM CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    layout="wide",
    page_title="Supply Chain Risk Intelligence Framework",
    page_icon="🔗",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Roboto+Mono:wght@500;600&display=swap');

/* ── Design tokens ────────────────────────────────── */
:root {
    --bg:        #f4f6fb;
    --surface:   #ffffff;
    --border:    #e5e9f0;
    --ink:       #0f172a;   /* headings                        */
    --body:      #111827;   /* body text (7:1 on bg)           */
    --muted:     #1f2937;   /* secondary text — near-black on light bg */
    --faint:     #94a3b8;   /* DECORATION ONLY — never text    */
    --brand:     #4f46e5;
    --brand-ink: #4338ca;
    --mono:      'Roboto Mono', ui-monospace, SFMono-Regular, monospace;
    --shadow-sm: 0 1px 2px rgba(15,23,42,.06);
    --shadow-md: 0 6px 16px rgba(15,23,42,.08);
    --shadow-lg: 0 16px 38px rgba(15,23,42,.12);
}

/* ── Global ───────────────────────────────────────── */
html, body, .stApp { font-family: 'Inter', sans-serif; }
.stApp {
    background:
        radial-gradient(1100px 560px at 12% -12%, rgba(79,70,229,.05), transparent 60%),
        var(--bg);
    color: var(--body);
}
.block-container { padding-top: 2.2rem; max-width: 1180px; }
.stMarkdown p, .stMarkdown li { color: var(--body); }
[data-testid="stCaptionContainer"], .stCaption, .stCaption p { color: var(--muted) !important; }

/* numerals align in columns */
.tnum, .metric-card .score, .perf-table td, .stat-chip .val { font-variant-numeric: tabular-nums; }

span.material-symbols-rounded, i[class^="st-"], [class*="stIcon"] {
    font-family: 'Material Symbols Rounded' !important;
}

/* Focus visibility (keyboard users) */
a:focus-visible, button:focus-visible, [role="radio"]:focus-visible, summary:focus-visible {
    outline: 3px solid rgba(79,70,229,.55); outline-offset: 2px; border-radius: 8px;
}

/* ── Hero header ──────────────────────────────────── */
.hero {
    position: relative; border-radius: 20px; padding: 30px 34px;
    margin-bottom: 20px; overflow: hidden; box-shadow: var(--shadow-lg);
    background: linear-gradient(118deg, #4338ca 0%, #5b21b6 62%, #6d28d9 100%);
}
.hero::after {
    content: ""; position: absolute; inset: 0; pointer-events: none;
    background: radial-gradient(440px 240px at 90% -30%, rgba(255,255,255,.20), transparent 60%);
}
.hero h1 {
    position: relative; z-index: 1; color: #fff;
    font-size: 31px; font-weight: 800; letter-spacing: -.5px; margin: 0; line-height: 1.14;
}
.hero p { position: relative; z-index: 1; color: rgba(255,255,255,.90); font-size: 15px; margin: 10px 0 0; }
.hero .hero-badges { position: relative; z-index: 1; display: flex; gap: 8px; margin-top: 16px; flex-wrap: wrap; }
.hero .hbadge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 11px; border-radius: 8px; font-size: 12.5px; font-weight: 600;
    background: rgba(255,255,255,.14); color: #fff; border: 1px solid rgba(255,255,255,.22);
}
.hero .hbadge svg { width: 15px; height: 15px; }

/* Semantic page headings (inner pages) */
h1.page-title { font-size: 26px; font-weight: 800; color: var(--ink); letter-spacing: -.4px; margin: 2px 0 2px; }
.page-sub { color: var(--muted); font-size: 14px; margin: 0 0 4px; max-width: 74ch; }

/* ── Callout (honest framing — high contrast, not a faint caption) ── */
.callout {
    display: flex; gap: 12px; align-items: flex-start;
    background: #eef2ff; border: 1px solid #c7d2fe; border-radius: 12px;
    padding: 13px 16px; margin: 8px 0 4px; color: #312e81; font-size: 13.5px; line-height: 1.5;
    max-width: 92ch;
}
.callout svg { flex: none; width: 18px; height: 18px; margin-top: 1px; color: var(--brand); }

/* ── Sidebar ──────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #141d31 55%, #1c2740 100%);
    border-right: 1px solid rgba(148,163,184,.14);
}
section[data-testid="stSidebar"] * { color: #d5dced; }
section[data-testid="stSidebar"] hr { border-color: rgba(148,163,184,.16); }
section[data-testid="stSidebar"] .stRadio label { color: #d5dced !important; font-weight: 500; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    display: flex; align-items: center; gap: 10px; min-height: 44px;
    padding: 8px 12px; margin: 3px 0; border-radius: 10px;
    border: 1px solid transparent; cursor: pointer;
    transition: background .15s ease, border-color .15s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: rgba(99,102,241,.16); border-color: rgba(99,102,241,.38);
}

/* ── Resource badge (authored icon, tinted) ───────── */
.res-badge {
    display: inline-flex; align-items: center; justify-content: center; flex: none;
    border-radius: 10px;
    background: color-mix(in srgb, var(--rc) 13%, #fff);
    color: var(--rc);
    border: 1px solid color-mix(in srgb, var(--rc) 28%, #fff);
}
.res-badge svg { width: 60%; height: 60%; }
.on-dark .res-badge { background: color-mix(in srgb, var(--rc) 26%, transparent); border-color: color-mix(in srgb, var(--rc) 45%, transparent); }

/* ── Metric Cards ─────────────────────────────────── */
.metric-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
    padding: 20px 20px 18px; text-align: center; box-shadow: var(--shadow-sm);
    transition: transform .18s ease, box-shadow .18s ease;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: var(--shadow-md); }
.metric-card .mhead { display: flex; align-items: center; justify-content: center; gap: 10px; margin-bottom: 4px; }
.metric-card .mhead .res-badge { width: 34px; height: 34px; }
.metric-card .mname { font-size: 15px; font-weight: 700; color: var(--ink); letter-spacing: .2px; }
.metric-card .mtag { font-size: 11px; color: var(--muted); margin: 0 0 12px; }
.metric-card .score { font-size: 40px; font-weight: 800; margin: 2px 0; color: var(--ink); letter-spacing: -1px; }
.metric-card .score.nodata { font-size: 26px; color: var(--muted); letter-spacing: 0; }
.metric-card .metric-bar { height: 7px; border-radius: 999px; background: #eef2f7; overflow: hidden; margin: 12px 4px 13px; }
.metric-card .metric-bar > div { height: 100%; width: 100%; border-radius: 999px; transform-origin: left; transform: scaleX(var(--fill, 0)); transition: transform .6s ease; }
.metric-card .asof { font-size: 10.5px; color: var(--muted); margin-top: 10px; }
.metric-card .status {
    font-size: 12.5px; font-weight: 700; padding: 4px 13px; border-radius: 999px;
    display: inline-flex; align-items: center; gap: 6px;
}
.status .dot { width: 8px; height: 8px; border-radius: 999px; background: currentColor; flex: none; }

.risk-safe { color: #15803d !important; } .risk-medium { color: #b45309 !important; }
.risk-high { color: #c2410c !important; } .risk-critical { color: #b91c1c !important; }
.bg-safe { background: #dcfce7; color: #14532d; border: 1px solid #bbf7d0; }
.bg-medium { background: #fef3c7; color: #78350f; border: 1px solid #fde68a; }
.bg-high { background: #ffedd5; color: #7c2d12; border: 1px solid #fed7aa; }
.bg-critical { background: #fee2e2; color: #7f1d1d; border: 1px solid #fecaca; }
.bg-nodata { background: #f1f5f9; color: #1f2937; border: 1px solid #e2e8f0; }

/* ── Section headings (real <h2>) ─────────────────── */
h2.section-header {
    display: flex; align-items: center; gap: 11px;
    font-size: 18px; font-weight: 700; color: var(--ink);
    margin: 34px 0 14px; padding-bottom: 9px; border-bottom: 1px solid var(--border);
}
h2.section-header .shi { display: inline-flex; color: var(--brand); }
h2.section-header .shi svg { width: 19px; height: 19px; }

/* ── Info Banner ──────────────────────────────────── */
.demo-banner {
    background: #fff7ed; border: 1px solid #fed7aa; border-radius: 12px;
    padding: 13px 18px; color: #9a3412; font-size: 14px; margin-bottom: 18px;
}

/* ── Performance Table ────────────────────────────── */
.perf-table { width: 100%; border-collapse: separate; border-spacing: 0; border-radius: 12px; overflow: hidden; border: 1px solid var(--border); box-shadow: var(--shadow-sm); }
.perf-table th { background: #eef2f8; color: #1f2937; padding: 13px 16px; font-weight: 700; font-size: 12px; text-transform: uppercase; letter-spacing: .5px; text-align: left; }
.perf-table td { padding: 12px 16px; border-top: 1px solid var(--border); color: var(--body); font-size: 14px; background: var(--surface); }
.perf-table tr:hover td { background: #f8fafc; }
.perf-table .best-row td { background: #f0fdf4; font-weight: 600; }
.perf-table .bar-cell { position: relative; }
.perf-table .cellbar { display: inline-block; height: 6px; border-radius: 999px; vertical-align: middle; margin-left: 8px; }

/* ── Case study cards ─────────────────────────────── */
.case-card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 22px 24px; margin: 14px 0; line-height: 1.7; color: var(--body); box-shadow: var(--shadow-sm); }
.case-card h3 { color: var(--ink); font-size: 17px; margin: 0 0 10px; display: flex; align-items: center; gap: 9px; }
.case-card p { color: var(--body); margin: 0; }
.case-tag { display: inline-block; padding: 3px 10px; border-radius: 7px; font-size: 12px; font-weight: 700; background: #eef2ff; color: var(--brand-ink); border: 1px solid #c7d2fe; margin-bottom: 12px; }

/* ── Stat chips ───────────────────────────────────── */
.stat-chip { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 12px 18px; box-shadow: var(--shadow-sm); display: flex; align-items: center; gap: 11px; }
.stat-chip .ci { display: inline-flex; color: var(--muted); } .stat-chip .ci svg { width: 18px; height: 18px; }
.stat-chip .lbl { color: var(--muted); font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .6px; }
.stat-chip .val { font-weight: 700; font-size: 15px; color: var(--ink); }

/* ── Driver cards ─────────────────────────────────── */
.driver-card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 15px 17px; margin: 8px 0; box-shadow: var(--shadow-sm); transition: transform .15s ease, box-shadow .15s ease; }
.driver-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }
.driver-card .dname { font-weight: 600; color: var(--ink); font-size: 13.5px; margin-bottom: 8px; display: flex; justify-content: space-between; gap: 8px; }
.driver-card .dunit { color: var(--muted); font-weight: 500; font-size: 11px; }
.driver-card .dtrack { background: #eef2f7; border-radius: 999px; height: 8px; overflow: hidden; }
.driver-card .dfill { height: 100%; width: 100%; border-radius: 999px; transform-origin: left; transform: scaleX(var(--fill, 0)); transition: transform .6s ease; }
.driver-card .dval { color: var(--muted); font-size: 12px; margin-top: 7px; }

/* ── Charts: let value labels breathe (no clipping) ── */
.stPlotlyChart { border-radius: 14px; overflow: visible; background: var(--surface); border: 1px solid var(--border); box-shadow: var(--shadow-sm); padding: 6px 6px 2px; }
[data-testid="stImage"] img { border-radius: 12px; border: 1px solid var(--border); box-shadow: var(--shadow-sm); background: #fff; }

/* ── Expander ─────────────────────────────────────── */
.streamlit-expanderHeader, [data-testid="stExpander"] summary { color: var(--ink) !important; font-weight: 600 !important; min-height: 44px; }
[data-testid="stExpander"] { border-radius: 12px !important; border: 1px solid var(--border) !important; }

/* ── Buttons: 44px targets, brand fill in main ─────── */
.stButton button, .stDownloadButton button { min-height: 44px; border-radius: 10px; font-weight: 600; }
.block-container .stDownloadButton button { background: var(--brand); color: #fff !important; border: 1px solid var(--brand); }
.block-container .stDownloadButton button:hover { background: var(--brand-ink); border-color: var(--brand-ink); }
.block-container .stButton button { border: 1px solid var(--border); }
section[data-testid="stSidebar"] .stDownloadButton button { background: rgba(255,255,255,.07); color: #e8edf7 !important; border: 1px solid rgba(148,163,184,.3); }
section[data-testid="stSidebar"] .stDownloadButton button:hover { background: rgba(99,102,241,.24); border-color: rgba(99,102,241,.6); }

/* ── Data-freshness note ──────────────────────────── */
.stale-note {
    display: flex; align-items: center; gap: 9px;
    background: #fffbeb; border: 1px solid #fde68a; border-radius: 10px;
    padding: 9px 14px; margin: 2px 0 14px; color: #92400e; font-size: 13px; max-width: 92ch;
}
.stale-note svg { flex: none; width: 16px; height: 16px; color: #b45309; }

/* ── Metric-card sparkline + trend delta ──────────── */
.metric-card .spark { margin: 12px 0 2px; display: flex; justify-content: center; }
.metric-card .spark svg { opacity: .9; }
.metric-card .delta { font-size: 11.5px; color: var(--muted); font-weight: 600; }

/* ── Severity tag (non-colour cue in indicator table) ── */
.sev-tag {
    display: inline-block; margin-left: 8px; padding: 1px 8px; border-radius: 999px;
    font-size: 10.5px; font-weight: 700; letter-spacing: .3px; vertical-align: middle;
    background: #fee2e2; color: #991b1b; border: 1px solid #fecaca;
}

/* ── Selectbox → light surface (was dark on light page) ── */
[data-baseweb="select"] > div {
    background: var(--surface) !important; border-color: var(--border) !important;
    border-radius: 10px !important;
}
[data-baseweb="select"] div, [data-baseweb="select"] span, [data-baseweb="select"] input { color: var(--ink) !important; }
[data-baseweb="select"] svg { fill: var(--muted) !important; }
[data-baseweb="popover"] [role="listbox"] { background: var(--surface) !important; border: 1px solid var(--border) !important; }
[data-baseweb="popover"] [role="option"] { color: var(--ink) !important; }
[data-baseweb="popover"] [role="option"]:hover { background: #eef2ff !important; }
/* widget labels legible on the light page */
.block-container [data-testid="stWidgetLabel"] p { color: var(--body) !important; font-weight: 600; }

/* ── Disabled buttons read as unavailable, not blank ── */
.block-container .stButton button:disabled, .block-container .stDownloadButton button:disabled {
    opacity: 1 !important; background: #f8fafc !important; color: #94a3b8 !important;
    border: 1px dashed #cbd5e1 !important;
}

@media (max-width: 640px) {
    .hero { padding: 22px 20px; } .hero h1 { font-size: 24px; }
    .block-container { padding-top: 1.4rem; }
}
</style>
"""

# Small inline icons for section headings / chips (currentColor, one stroke style).
UI_SVG = {
    "scores":  _svg('<rect x="3" y="12" width="4" height="8" rx="1"/><rect x="10" y="7" width="4" height="13" rx="1"/><rect x="17" y="3" width="4" height="17" rx="1"/>'),
    "trend":   _svg('<path d="M3 17l5-5 4 3 7-8"/><path d="M16 7h4v4"/>'),
    "drivers": _svg('<circle cx="12" cy="12" r="9"/><path d="M12 12l4-2"/><circle cx="12" cy="12" r="1.5"/>'),
    "download":_svg('<path d="M12 3v12M7 11l5 5 5-5"/><path d="M4 21h16"/>'),
    "pipeline":_svg('<circle cx="6" cy="6" r="2.4"/><circle cx="18" cy="18" r="2.4"/><path d="M6 8.5V15a3 3 0 0 0 3 3h6"/>'),
    "roc":     _svg('<path d="M4 20V4"/><path d="M4 20h16"/><path d="M4 16C9 16 9 6 20 6"/>'),
    "pr":      _svg('<path d="M4 20V4"/><path d="M4 20h16"/><path d="M4 8c6 0 8 10 16 10"/>'),
    "matrix":  _svg('<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M12 3v18M3 12h18"/>'),
    "info":    _svg('<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>'),
    "calendar":_svg('<rect x="3" y="4" width="18" height="17" rx="2"/><path d="M3 9h18M8 2v4M16 2v4"/>'),
    "severity":_svg('<path d="M12 3l9 16H3z"/><path d="M12 10v4M12 17h.01"/>'),
    "cube":    _svg('<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M12 3v18M4 7.5l8 4.5 8-4.5"/>'),
}


def sec(icon: str, text: str) -> str:
    """A real <h2> section heading with an inline icon (screen-reader navigable)."""
    return f'<h2 class="section-header"><span class="shi">{UI_SVG.get(icon, "")}</span>{text}</h2>'


# Brand mark — interlocked links (supply chain), drawn not emoji.
LOGO_SVG = _svg(
    '<rect x="3" y="8.5" width="10" height="7" rx="3.5"/>'
    '<rect x="11" y="8.5" width="10" height="7" rx="3.5"/>', 34
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DATA LOADING 
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data(ttl=300)
def load_risk_scores():
    """Load real risk scores. Returns (scores_dict, missing_flag).
    missing_flag=True means the file is absent/unreadable — the UI must NOT
    render a confident 'Safe 0.00', but an explicit 'no data' state."""
    try:
        path = PROCESSED_DATA_DIR / "risk_scores.csv"
        if path.exists():
            df = pd.read_csv(path)
            return {row["resource"]: row["risk_score"] for _, row in df.iterrows()}, False
    except Exception:
        pass
    return {}, True


@st.cache_data(ttl=300)
def load_asof():
    """Latest scored week across the timeline, as a display string, or ''."""
    try:
        path = PROCESSED_DATA_DIR / "risk_timeline.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"])
            if not df.empty:
                return df["date"].max().strftime("%d %b %Y")
    except Exception:
        pass
    return ""


@st.cache_data(ttl=300)
def load_metrics():
    """Try to load real model metrics."""
    try:
        # Try evaluation_report.csv first (produced by train_models.py)
        for fname in ["evaluation_report.csv", "model_comparison.csv"]:
            path = OUTPUTS_DIR / fname
            if path.exists():
                df = pd.read_csv(path)
                # Normalize column names for consistency
                rename_map = {
                    "f1_score": "f1",
                    "train_time_s": "training_time",
                    "training_time_s": "training_time",
                }
                df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
                return df, False
    except Exception:
        pass
    return pd.DataFrame(), True


@st.cache_data(ttl=300)
def load_timeline():
    """Load real timeline data. Returns (df, missing_flag)."""
    try:
        path = PROCESSED_DATA_DIR / "risk_timeline.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"])
            return df, False
    except Exception:
        pass
    return pd.DataFrame(), True


@st.cache_data(ttl=300)
def load_shap_data():
    """Try to load real SHAP data."""
    try:
        shap_dir = OUTPUTS_DIR / "shap"
        imp_path = shap_dir / "feature_importance.csv"
        if imp_path.exists():
            imp_df = pd.read_csv(imp_path)
            return (
                imp_df["feature"].tolist(),
                None,  # No raw shap values from file
                imp_df["importance"].values,
            ), False
    except Exception:
        pass
    return ([], None, []), True


@st.cache_data(ttl=300)
def load_resource_drivers():
    """Load real per-resource SHAP drivers (resource, feature, contribution, value).
    Produced by the export adapter. Returns a list of dicts, or [] if absent."""
    try:
        path = PROCESSED_DATA_DIR / "resource_drivers.csv"
        if path.exists():
            df = pd.read_csv(path)
            return df.to_dict("records")
    except Exception:
        pass
    return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPER FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def risk_color_class(score: float) -> tuple:
    """Return (text_class, badge_class, label) for a risk score.
    The label is a word (not just a color), so risk is never conveyed by hue alone."""
    if score < RISK_THRESHOLDS["low"]:
        return "risk-safe", "bg-safe", "Safe"
    elif score < RISK_THRESHOLDS["medium"]:
        return "risk-medium", "bg-medium", "Watch"
    elif score < RISK_THRESHOLDS["high"]:
        return "risk-high", "bg-high", "Elevated"
    else:
        return "risk-critical", "bg-critical", "Critical"


def risk_hex(score: float) -> str:
    """Return hex color for a risk score."""
    if score < RISK_THRESHOLDS["low"]:
        return "#16a34a"
    elif score < RISK_THRESHOLDS["medium"]:
        return "#d97706"
    elif score < RISK_THRESHOLDS["high"]:
        return "#ea580c"
    else:
        return "#dc2626"


_AXIS_GRID = dict(gridcolor="#eef1f6", zerolinecolor="#cbd5e1")

# Generous margins + tabular font so value labels never clip the card edge.
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,1)",
    font=dict(family="Inter, sans-serif", color="#111827", size=12),
    xaxis=dict(**_AXIS_GRID),
    yaxis=dict(**_AXIS_GRID),
    margin=dict(l=64, r=64, t=54, b=48),
    legend=dict(
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#e5e9f0",
        borderwidth=1,
        font=dict(size=12, color="#111827"),
    ),
)


def show_demo_banner(kind: str = "data"):
    """Info banner shown when a required export is missing (real, not fake data)."""
    msg = {
        "data": "No scored data found — the model has not produced "
                "<code>risk_scores.csv</code> yet.",
        "metrics": "No model-benchmark results found "
                   "(<code>evaluation_report.csv</code> is missing).",
        "shap": "No SHAP explanations found — the export adapter has not "
                "written feature attributions yet.",
    }.get(kind, "Required export is missing.")
    st.markdown(
        f'<div class="demo-banner"><strong>Not scored yet.</strong> {msg} '
        'Run <code>python scripts/build_dashboard_data.py</code> to populate it.</div>',
        unsafe_allow_html=True,
    )


# ── Card sparkline + trend reconciliation (P0 fix) ─────────────
WORK_DIR = PROCESSED_DATA_DIR.parent / "_work"

# NumPy 2 renamed trapz -> trapezoid; support both.
_TRAPZ = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)


def weeks_stale(asof_str: str) -> int:
    """Whole weeks between the latest scored date and today (0 if unknown/fresh)."""
    import datetime as _dt
    try:
        d = _dt.datetime.strptime(asof_str, "%d %b %Y").date()
        return max((_dt.date.today() - d).days // 7, 0)
    except Exception:
        return 0


def sparkline_svg(vals, w=112, h=28, color="#4f46e5"):
    """Inline SVG line of a REAL risk series (recent weeks) — content, reconciling
    a card's current score with its trajectory. Not decoration."""
    vals = [float(v) for v in vals if v == v]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    n = len(vals)
    pts = [f"{(i/(n-1))*(w-2)+1:.1f},{h-3-((v-lo)/rng)*(h-6):.1f}" for i, v in enumerate(vals)]
    lx, ly = pts[-1].split(",")
    return (
        f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" fill="none" aria-hidden="true">'
        f'<polyline points="{" ".join(pts)}" stroke="{color}" stroke-width="1.6" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{lx}" cy="{ly}" r="2" fill="{color}"/></svg>'
    )


def card_trend(timeline_df, res, weeks_back=4, n=24):
    """(sparkline_html, delta_text) for a resource from the risk timeline."""
    if timeline_df is None or timeline_df.empty:
        return "", ""
    s = timeline_df[timeline_df["resource"] == res].sort_values("date")
    vals = s["risk_score"].tolist()
    if len(vals) < 2:
        return "", ""
    recent = vals[-n:]
    cur = vals[-1]
    prev = vals[-min(weeks_back + 1, len(vals))]
    spark = sparkline_svg(recent, color=RESOURCE_COLORS.get(res, "#4f46e5"))
    d = cur - prev
    band = lambda v: risk_color_class(v)[2]
    if abs(d) < 0.02:
        delta = "steady · past 4 wks"
    elif band(prev) != band(cur):
        arrow = "↑" if d > 0 else "↓"
        delta = f"{arrow} from {band(prev)} · 4 wks ago"
    else:
        arrow = "↑" if d > 0 else "↓"
        delta = f"{arrow} {abs(d):.2f} · past 4 wks"
    return spark, delta


# ── Native evaluation charts (replace matplotlib PNGs; consistent Plotly) ──
@st.cache_data(ttl=300)
def load_eval_arrays():
    """(y_test, {ModelName: proba}) from dashboard_export/_work, or (None, {})."""
    models = {"lightgbm": "LightGBM", "xgboost": "XGBoost",
              "tabpfn": "TabPFN", "ft": "FT-Transformer"}
    try:
        tp = WORK_DIR / "test.csv"
        if not tp.exists():
            return None, {}
        y = pd.read_csv(tp)["label"].astype(int).values
        out = {}
        for k, disp in models.items():
            p = WORK_DIR / f"testproba_{k}.npy"
            if p.exists():
                arr = np.load(p)
                if len(arr) == len(y):
                    out[disp] = arr
        return (y, out) if out else (None, {})
    except Exception:
        return None, {}


def _roc_points(y, p):
    order = np.argsort(-p)
    ys = y[order]
    P = max(ys.sum(), 1)
    N = max(len(ys) - ys.sum(), 1)
    tpr = np.concatenate([[0.0], np.cumsum(ys) / P])
    fpr = np.concatenate([[0.0], np.cumsum(1 - ys) / N])
    return fpr, tpr, float(_TRAPZ(tpr, fpr))


def _pr_points(y, p):
    order = np.argsort(-p)
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    P = max(ys.sum(), 1)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / P
    ap = float(np.sum(prec * np.diff(np.concatenate([[0.0], rec]))))
    return rec, prec, ap


def _confusion(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return np.array([
        [int(((pred == 0) & (y == 0)).sum()), int(((pred == 1) & (y == 0)).sum())],
        [int(((pred == 0) & (y == 1)).sum()), int(((pred == 1) & (y == 1)).sum())],
    ])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: RISK OVERVIEW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def page_risk_overview():
    """Render the Risk Overview page."""
    # Load data
    scores, missing_scores = load_risk_scores()
    timeline_df, _ = load_timeline()
    asof = load_asof()

    # Hero header — resource pills carry the authored icon set (no emoji).
    pills = "".join(
        f'<span class="hbadge on-dark">{resource_badge(r, 18)}{r}</span>'
        for r in RESOURCES
    )
    st.markdown(
        f'<div class="hero on-dark">'
        f'<h1>{DASHBOARD_TITLE}</h1>'
        f'<p>{DASHBOARD_SUBTITLE}</p>'
        f'<div class="hero-badges">{pills}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if missing_scores:
        show_demo_banner("data")

    # Honest framing — high-contrast callout, not a faint caption.
    st.markdown(
        f'<div class="callout">{UI_SVG["info"]}<div>'
        "<strong>Read these as decision support, not certainty.</strong> Scores are model "
        "estimates from real public signals (prices, news, filings). The model's ability to "
        "separate tightening from non-tightening is still being validated on more diverse "
        "examples, so treat a low score as “no signal yet,” not a guarantee."
        "</div></div>",
        unsafe_allow_html=True,
    )

    # Data-freshness note — the score is only as current as the latest scored week.
    stale = weeks_stale(asof)
    if asof and stale >= 2:
        st.markdown(
            f'<div class="stale-note">{UI_SVG["calendar"]}'
            f'<span>Latest scored week is <strong>{asof}</strong> — about '
            f'{stale} weeks old. Rebuild the data for a current read.</span></div>',
            unsafe_allow_html=True,
        )

    # ── Metric Cards ──────────────────────────────────────────
    st.markdown(sec("scores", "Current Risk Scores"), unsafe_allow_html=True)
    st.caption("Each card shows the latest model score, its recent trajectory, and how it "
               "compares to a month ago — so “current” is read against history, not alone.")
    cols = st.columns(len(RESOURCES))
    for col, res in zip(cols, RESOURCES):
        accent = RESOURCE_COLORS.get(res, "#4f46e5")
        badge = resource_badge(res)
        tagline = RESOURCE_TAGLINE.get(res, "")
        has_score = (not missing_scores) and (res in scores)
        if has_score:
            score = float(scores[res])
            color_cls, bg_cls, label = risk_color_class(score)
            bar_hex = risk_hex(score)
            fill = min(max(score, 0.0), 1.0)   # no artificial floor — 0.00 reads empty
            spark, delta = card_trend(timeline_df, res)
            asof_html = f'<div class="asof">as of {asof}</div>' if asof else ""
            trend_html = (f'<div class="spark">{spark}</div>'
                          f'<div class="delta">{delta}</div>') if spark else ""
            col.markdown(
                f'<div class="metric-card" style="--rc:{accent};">'
                f'  <div class="mhead">{badge}<span class="mname">{res}</span></div>'
                f'  <div class="mtag">{tagline}</div>'
                f'  <div class="score {color_cls}">{score:.2f}</div>'
                f'  <div class="metric-bar"><div style="--fill:{fill:.3f}; background:{bar_hex};"></div></div>'
                f'  <div class="status {bg_cls}"><span class="dot"></span>{label}</div>'
                f'  {trend_html}'
                f'  {asof_html}'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            # Explicit "no data" state — must be visibly different from a real Safe reading.
            col.markdown(
                f'<div class="metric-card" style="--rc:{accent};">'
                f'  <div class="mhead">{badge}<span class="mname">{res}</span></div>'
                f'  <div class="mtag">{tagline}</div>'
                f'  <div class="score nodata">—</div>'
                f'  <div class="metric-bar"><div style="--fill:0;"></div></div>'
                f'  <div class="status bg-nodata"><span class="dot"></span>Not scored</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Risk Timeline ─────────────────────────────────────────
    st.markdown(sec("trend", "Model Score Timeline"), unsafe_allow_html=True)
    st.caption(
        "Weekly model score per resource (raw, uncalibrated 0–1 — it tends to saturate "
        "near 0 or 1 rather than read as a smooth probability). Shaded bands mark "
        "documented tightening episodes; the dashed lines are the score thresholds."
    )

    fig = go.Figure()
    if timeline_df.empty:
        st.warning("Score timeline has not been generated yet — run the export adapter.")
    else:
        for res in RESOURCES:
            res_df = timeline_df[timeline_df["resource"] == res].sort_values("date")
            fig.add_trace(go.Scatter(
                x=res_df["date"], y=res_df["risk_score"],
                mode="lines",
                name=res,
                line=dict(color=RESOURCE_COLORS.get(res, "#888"), width=2.2),
                fill="none",
                hovertemplate=f"<b>{res}</b><br>%{{x|%d %b %Y}} · score %{{y:.3f}}<extra></extra>",
            ))

    # Threshold lines — distinct hues (amber→orange→red) so they read without the label.
    _thr = {"low": ("Watch", "#f59e0b"), "medium": ("Elevated", "#f97316"), "high": ("Critical", "#dc2626")}
    for name, val in RISK_THRESHOLDS.items():
        lbl, tc = _thr.get(name, (name.capitalize(), "#94a3b8"))
        fig.add_hline(
            y=val, line_dash="dash", line_color=tc, line_width=1.2,
            annotation_text=lbl,
            annotation_position="top left",
            annotation_font=dict(color=tc, size=10),
        )

    # Shade disruption episodes; stagger labels across 4 rows so they never overlap.
    for i, evt in enumerate(DISRUPTION_EVENTS):
        fig.add_vrect(
            x0=evt["start"], x1=evt["end"],
            fillcolor="rgba(239,68,68,0.055)", line_width=0, layer="below",
        )
        rc = RESOURCE_COLORS.get(evt["resources"][0], "#ef4444")
        y_row = [1.16, 1.10, 1.04, 0.98][i % 4]
        fig.add_annotation(
            x=evt["start"], y=y_row, text=f'▎{evt["description"]}',
            showarrow=False, xanchor="left", yanchor="bottom",
            font=dict(size=10, color=rc),
            bgcolor="rgba(255,255,255,0.82)", borderpad=2,
        )

    layout_kwargs = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("yaxis", "legend")}
    fig.update_layout(
        **layout_kwargs,
        height=470,
        title_text="",
        xaxis_title="",
        yaxis_title="Model score (uncalibrated, 0–1)",
        yaxis=dict(range=[0, 1.22], **_AXIS_GRID),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.16, xanchor="center", x=0.5,
                    bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=12, color="#111827")),
    )
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})

    # ── Key Risk Drivers ──────────────────────────────────────
    st.markdown(sec("drivers", "Key Risk Drivers"), unsafe_allow_html=True)
    st.caption("Which signals the model weights most, overall. Magnitude only — "
               "per-resource direction is on the Explainability page.")

    shap_data, missing_shap = load_shap_data()
    feature_names, _, importance = shap_data

    if len(importance) == 0:
        show_demo_banner("shap")
    else:
        # Rank by importance, drop features that never contribute (importance ~ 0).
        order = np.argsort(importance)[::-1]
        ranked = [(feature_names[i], float(importance[i])) for i in order
                  if float(importance[i]) > 1e-6][:6]
        top_max = max((v for _, v in ranked), default=1.0)
        driver_cols = st.columns(3)
        for i, (feat, val) in enumerate(ranked):
            with driver_cols[i % 3]:
                fill = min(val / top_max, 1.0) if top_max else 0
                st.markdown(
                    f'<div class="driver-card">'
                    f'<div class="dname"><span>{pretty_feature(feat)}</span>'
                    f'<span class="dunit">{feature_unit(feat)}</span></div>'
                    f'<div class="dtrack"><div class="dfill" style="--fill:{fill:.3f}; '
                    f'background:var(--brand);"></div></div>'
                    f'<div class="dval tnum">Mean impact: {val:.3f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Download Dataset Buttons ──────────────────────────────
    st.markdown(sec("download", "Download Datasets"), unsafe_allow_html=True)
    dl_cols = st.columns(4)
    _raw_path = PROCESSED_DATA_DIR / "combined_weekly.csv"
    _labeled_path = PROCESSED_DATA_DIR / "labeled_dataset.csv"
    _feature_path = PROCESSED_DATA_DIR / "feature_dataset.csv"
    _report_path = OUTPUTS_DIR / "evaluation_report.csv"

    with dl_cols[0]:
        if _raw_path.exists():
            st.download_button("Raw Data", data=_raw_path.read_bytes(), file_name="combined_weekly.csv", mime="text/csv", use_container_width=True)
        else:
            st.button("Raw Data (N/A)", disabled=True, use_container_width=True)
    with dl_cols[1]:
        if _feature_path.exists():
            st.download_button("Feature Dataset", data=_feature_path.read_bytes(), file_name="feature_dataset.csv", mime="text/csv", use_container_width=True)
        else:
            st.button("Feature Dataset (N/A)", disabled=True, use_container_width=True)
    with dl_cols[2]:
        if _labeled_path.exists():
            st.download_button("Labeled Dataset", data=_labeled_path.read_bytes(), file_name="labeled_dataset.csv", mime="text/csv", use_container_width=True)
        else:
            st.button("Labeled Dataset (N/A)", disabled=True, use_container_width=True)
    with dl_cols[3]:
        if _report_path.exists():
            st.download_button("Evaluation Report", data=_report_path.read_bytes(), file_name="evaluation_report.csv", mime="text/csv", use_container_width=True)
        else:
            st.button("Eval Report (N/A)", disabled=True, use_container_width=True)

    # ── Per-resource download ─────────────────────────────────
    with st.expander("Download per-resource weekly data"):
        res_dl_cols = st.columns(4)
        for i, res in enumerate(RESOURCES):
            rpath = PROCESSED_DATA_DIR / f"{res}_weekly.csv"
            with res_dl_cols[i]:
                if rpath.exists():
                    st.download_button(f"{res}", data=rpath.read_bytes(), file_name=f"{res}_weekly.csv", mime="text/csv", use_container_width=True, key=f"dl_{res}")
                else:
                    st.button(f"{res} (N/A)", disabled=True, use_container_width=True, key=f"dl_{res}")

    # ── Pipeline Description ──────────────────────────────────
    with st.expander("How the framework works — data → signals → models"):
        st.markdown("""
        **This build runs on a real multi-signal data engine (DRAM · NAND · HBM · GPU):**

        1. **Data ingestion** — real producer stock prices (yfinance), historical news
           (GDELT + NewsAPI), SEC filings (EDGAR), and memory commodity prices, stored in
           a SQLite database.

        2. **News understanding** — near-duplicate news is clustered, then an LLM extracts a
           1–10 **severity** score per story (news becomes a real numeric signal).

        3. **Feature engineering** — 10 weekly features per resource: price level / trend /
           z-score, forecast point / uncertainty / disagreement, news-severity mean & count,
           and supplier concentration.

        4. **Labelling** — binary risk: `1` if supply tightening is under way within the
           labelling horizon, `0` otherwise, from hand-curated episodes (both classes).

        5. **Model benchmark** — LightGBM, XGBoost, FT-Transformer and TabPFN trained on a
           chronological split; evaluated on Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.

        6. **SHAP explainability** — feature-level attributions so you can see *why* risk is
           elevated for each resource.
        """)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: MODEL PERFORMANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def page_model_performance():
    """Render the Model Performance page."""
    st.markdown(
        '<h1 class="page-title">Model Performance</h1>'
        '<div class="page-sub">Four models benchmarked on a chronological (time-ordered) '
        'train/test split — the split respects time, so no future data leaks into training. '
        'Ranked by F1.</div>',
        unsafe_allow_html=True,
    )

    metrics_df, missing = load_metrics()
    if missing or metrics_df.empty:
        show_demo_banner("metrics")
        return

    # ── Comparison Table ──────────────────────────────────────
    st.markdown(sec("scores", "Performance Metrics"), unsafe_allow_html=True)

    best_f1_model = metrics_df.loc[metrics_df["f1"].idxmax(), "model"]
    _tmax = max(metrics_df.get("training_time", pd.Series([1])).max(), 1)

    header = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Train time"]
    table_html = '<table class="perf-table"><thead><tr>'
    table_html += "".join(f"<th>{c}</th>" for c in header)
    table_html += "</tr></thead><tbody>"
    for _, row in metrics_df.iterrows():
        is_best = row["model"] == best_f1_model
        row_class = ' class="best-row"' if is_best else ""
        mc = MODEL_COLORS.get(row["model"], "#4f46e5")
        star = ' <span style="color:#b45309;font-weight:700;">★ best</span>' if is_best else ""
        table_html += f"<tr{row_class}>"
        table_html += f'<td><strong>{row["model"]}</strong>{star}</td>'
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            val = row.get(metric, 0)
            # inline mini-bar on F1 so the ranking is scannable at a glance
            if metric == "f1":
                w = int(max(min(val, 1), 0) * 54)
                bar = f'<span class="cellbar" style="width:{w}px;background:{mc};"></span>'
                table_html += f'<td class="bar-cell tnum">{val:.3f}{bar}</td>'
            else:
                table_html += f'<td class="tnum">{val:.3f}</td>'
        # readable relative training time (log-fair label, avoids the linear-axis trap)
        t = float(row.get("training_time", 0))
        table_html += f'<td class="tnum">{t:.1f}s</td>'
        table_html += "</tr>"
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)
    st.caption("★ marks the best F1. Train time spans 2s → ~10min across models; "
               "read it from the column, not a bar (the scale is highly skewed).")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROC / PR / Confusion — native Plotly (consistent with the rest) ──
    y_eval, probas = load_eval_arrays()

    if y_eval is None or not probas:
        # Fallback to saved PNGs if the per-model test arrays aren't present.
        _roc_img, _pr_img, _cm_img = (OUTPUTS_DIR / f for f in
                                      ("roc_curves.png", "pr_curves.png", "confusion_matrices.png"))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(sec("roc", "ROC Curves"), unsafe_allow_html=True)
            st.image(str(_roc_img), use_container_width=True) if _roc_img.exists() else st.warning("Not generated yet.")
        with c2:
            st.markdown(sec("pr", "Precision–Recall Curves"), unsafe_allow_html=True)
            st.image(str(_pr_img), use_container_width=True) if _pr_img.exists() else st.warning("Not generated yet.")
        st.markdown(sec("matrix", "Confusion Matrices"), unsafe_allow_html=True)
        st.image(str(_cm_img), use_container_width=True) if _cm_img.exists() else st.warning("Not generated yet.")
        return

    _base = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "legend"}
    _hleg = dict(orientation="h", y=-0.30, x=0.5, xanchor="center",
                 bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=11, color="#111827"))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(sec("roc", "ROC Curves"), unsafe_allow_html=True)
        figr = go.Figure()
        for m, p in probas.items():
            fpr, tpr, auc = _roc_points(y_eval, p)
            figr.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{m} · AUC {auc:.3f}",
                                      line=dict(color=MODEL_COLORS.get(m, "#888"), width=2.2)))
        figr.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                                  line=dict(color="#cbd5e1", dash="dot", width=1)))
        figr.update_layout(**_base, height=360, title_text="",
                           xaxis_title="False positive rate", yaxis_title="True positive rate",
                           legend=_hleg)
        figr.update_xaxes(range=[0, 1]); figr.update_yaxes(range=[0, 1.02])
        st.plotly_chart(figr, width="stretch", theme=None, config={"displayModeBar": False})
        st.caption("Higher and further left is better; the dotted diagonal is chance.")
    with col2:
        st.markdown(sec("pr", "Precision–Recall Curves"), unsafe_allow_html=True)
        figp = go.Figure()
        for m, p in probas.items():
            rec, prec, ap = _pr_points(y_eval, p)
            figp.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"{m} · AP {ap:.3f}",
                                      line=dict(color=MODEL_COLORS.get(m, "#888"), width=2.2)))
        figp.add_hline(y=float(y_eval.mean()), line_dash="dot", line_color="#cbd5e1", line_width=1,
                       annotation_text=f"baseline {y_eval.mean():.2f}", annotation_position="bottom right",
                       annotation_font=dict(color="#6b7280", size=10))
        figp.update_layout(**_base, height=360, title_text="",
                           xaxis_title="Recall", yaxis_title="Precision", legend=_hleg)
        figp.update_xaxes(range=[0, 1]); figp.update_yaxes(range=[0, 1.02])
        st.plotly_chart(figp, width="stretch", theme=None, config={"displayModeBar": False})
        st.caption("Higher curves are better against the positive-class baseline.")

    # ── Confusion Matrices — native heatmaps at threshold 0.5 ──
    st.markdown(sec("matrix", "Confusion Matrices"), unsafe_allow_html=True)
    n = len(probas)
    figc = make_subplots(rows=1, cols=n, subplot_titles=list(probas.keys()), horizontal_spacing=0.055)
    for i, (m, p) in enumerate(probas.items(), 1):
        cm = _confusion(y_eval, p)
        figc.add_trace(go.Heatmap(
            z=cm, x=["No risk", "Risk"], y=["No risk", "Risk"],
            colorscale=[[0, "#f1f5f9"], [1, "#c7d2fe"]], showscale=False,
            text=cm, texttemplate="%{text}", textfont=dict(size=14, color="#111827"),
            xgap=2, ygap=2, hoverinfo="skip"), row=1, col=i)
        figc.update_yaxes(autorange="reversed", row=1, col=i)
    figc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,1)",
                       font=dict(family="Inter, sans-serif", color="#111827", size=11),
                       height=300, margin=dict(l=44, r=20, t=54, b=40))
    figc.update_annotations(font=dict(size=12, color="#111827"))
    st.plotly_chart(figc, width="stretch", theme=None, config={"displayModeBar": False})
    st.caption("Rows = actual (No risk / Risk), columns = predicted, at a 0.5 threshold.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: EXPLAINABILITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def page_explainability():
    """Render the Explainability page."""
    st.markdown(
        '<h1 class="page-title">Model Explainability</h1>'
        '<div class="page-sub">SHAP feature attributions — how much each signal moves the '
        'model, overall and per resource.</div>',
        unsafe_allow_html=True,
    )

    shap_data, missing = load_shap_data()
    feature_names, shap_values, importance = shap_data
    if missing or len(importance) == 0:
        show_demo_banner("shap")
        return

    # ── Feature Importance Bar Chart ──────────────────────────
    st.markdown(sec("scores", "Global Feature Importance"), unsafe_allow_html=True)
    st.caption("Mean absolute SHAP value across the test set — bigger means the feature "
               "swings the risk score more. Features that never contribute are hidden.")

    importance = np.asarray(importance, dtype=float)
    # Drop always-zero features (e.g. inactive forecast_* columns) so they don't pad the chart.
    keep = [i for i in range(len(importance)) if importance[i] > 1e-6]
    keep = sorted(keep, key=lambda i: importance[i])   # ascending for horizontal bars
    sorted_features = [pretty_feature(feature_names[i]) for i in keep]
    sorted_importance = importance[keep]

    _mx = max(sorted_importance) if len(sorted_importance) else 1
    # Solid brand fill; opacity carries magnitude (no gradient-text, stays legible).
    colors = [f"rgba(79,70,229,{0.45 + 0.5 * (v / _mx):.2f})" for v in sorted_importance]

    fig_imp = go.Figure(go.Bar(
        x=sorted_importance, y=sorted_features, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.3f}" for v in sorted_importance],
        textposition="outside", textfont=dict(color="#111827", size=12),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
    ))
    fig_imp.update_layout(
        **PLOTLY_LAYOUT,
        height=max(320, len(sorted_features) * 42),
        title_text="",
        xaxis_title="Mean |SHAP value|",
    )
    fig_imp.update_yaxes(automargin=True)   # never clip prettified feature labels
    st.plotly_chart(fig_imp, width="stretch", theme=None, config={"displayModeBar": False})

    # ── Per-Resource Risk Drivers (REAL SHAP, not simulated) ──
    st.markdown(sec("drivers", "Per-Resource Risk Drivers"), unsafe_allow_html=True)
    st.caption("Signed SHAP contributions for each resource's most recent scored week. "
               "Red bars (right) push risk up; green bars (left) pull it down.")

    selected_resource = st.selectbox("Resource", RESOURCES)

    drivers = load_resource_drivers()
    res_drivers = [d for d in drivers if d.get("resource") == selected_resource]

    signed = True
    if not res_drivers:
        st.info("No per-resource drivers yet — showing global importance (unsigned) instead. "
                "Run the export adapter to generate per-resource SHAP.")
        order_g = [i for i in np.argsort(importance)[::-1] if importance[i] > 1e-6][:8]
        wf_feats = [feature_names[i] for i in order_g]
        wf_vals = [float(importance[i]) for i in order_g]
        signed = False
    else:
        res_drivers = sorted(res_drivers, key=lambda d: abs(d["contribution"]), reverse=True)[:8]
        wf_feats = [d["feature"] for d in res_drivers]
        wf_vals = [float(d["contribution"]) for d in res_drivers]

    badge = resource_badge(selected_resource, 30)
    st.markdown(
        f'<div class="case-card"><h3>{badge} {selected_resource} — top risk drivers</h3>',
        unsafe_allow_html=True,
    )
    driver_parts = []
    for feat, val in list(zip(wf_feats, wf_vals))[:5]:
        if signed:
            direction = "↑ raises" if val > 0 else "↓ lowers"
            color = "#dc2626" if val > 0 else "#15803d"
        else:
            direction, color = "· weight", "#4338ca"
        driver_parts.append(
            f'<span style="color:{color}; font-weight:600; white-space:nowrap; display:inline-block;">'
            f'{pretty_feature(feat)} <span style="font-weight:500;">{direction}</span></span>')
    st.markdown("&nbsp; · &nbsp;".join(driver_parts) + "</div>", unsafe_allow_html=True)

    if signed:
        colors_wf = ["#dc2626" if v > 0 else "#15803d" for v in wf_vals]
    else:
        colors_wf = ["#4f46e5"] * len(wf_vals)
    fig_wf = go.Figure(go.Bar(
        x=wf_vals,
        y=[pretty_feature(f) for f in wf_feats],
        orientation="h",
        marker=dict(color=colors_wf, line=dict(width=0)),
        text=[f"{v:+.3f}" if signed else f"{v:.3f}" for v in wf_vals],
        textposition="auto", textfont=dict(size=12),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Contribution: %{x:+.4f}<extra></extra>",
    ))
    # pad the x-range so 'auto'-placed labels have room on both sides
    _wmax = max((abs(v) for v in wf_vals), default=1.0) or 1.0
    fig_wf.update_layout(
        **PLOTLY_LAYOUT,
        height=max(300, len(wf_feats) * 46),
        title_text="",
        xaxis_title="Contribution to risk (log-odds)" if signed else "Mean |SHAP value|",
        xaxis_range=[-_wmax * 1.35, _wmax * 1.35] if signed else [0, _wmax * 1.25],
    )
    fig_wf.update_yaxes(automargin=True)   # never clip prettified feature labels
    fig_wf.add_vline(x=0, line_color="rgba(15,23,42,0.22)", line_width=1)
    st.plotly_chart(fig_wf, width="stretch", theme=None, config={"displayModeBar": False})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: HISTORICAL CASE STUDIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Case studies map to OUR curated tightening episodes (data/curated_episodes.json).
# event_idx indexes DISRUPTION_EVENTS: 0=GPU, 1=HBM, 2=DRAM, 3=NAND. Narratives are
# our hand-authored domain justifications; the indicators are qualitative drivers we
# actually track (price level/trend/z-score, news severity, supplier concentration) —
# no fabricated inventory/lead-time figures.
CASE_STUDIES = {
    "2024–2025 DRAM AI Tightening": {
        "event_idx": 2,
        "narrative": (
            "Through 2024 into 2025, server and DDR5 DRAM contract prices rose sharply as the "
            "major memory makers (Samsung, SK Hynix, Micron) diverted wafer capacity toward "
            "High-Bandwidth Memory for AI datacenters, tightening the supply of conventional "
            "DRAM. Our engine picks this up as a sustained rise in the producer-price proxy and "
            "its z-score, reinforced by news-severity signals from the extraction stage."
        ),
        "key_features": {
            "Price trend": "Sustained multi-quarter rise",
            "Price z-score": "Elevated vs trailing baseline",
            "News severity (LLM)": "Rising shortage/tightening coverage",
            "Supplier concentration": "High (memory oligopoly)",
            "Primary driver": "HBM capacity diversion + AI datacenter demand",
        },
    },
    "2024–2025 HBM Sold-Out Tightening": {
        "event_idx": 1,
        "narrative": (
            "HBM3/HBM3E was effectively sold out through 2024 into 2025 as AI-accelerator demand "
            "outran the combined HBM capacity of SK Hynix, Samsung and Micron. This is the "
            "sharpest of the memory episodes — advance-booked capacity and multi-quarter lead "
            "commitments — and it is the upstream cause of the parallel DRAM tightening."
        ),
        "key_features": {
            "Price trend": "Strong, sustained",
            "Demand signal": "AI-accelerator pull (sold-out)",
            "News severity (LLM)": "Persistent tightening coverage",
            "Supplier concentration": "Very high (three suppliers)",
            "Primary driver": "AI-accelerator demand outrunning HBM capacity",
        },
    },
    "2023–2025 GPU Allocation Constraint": {
        "event_idx": 0,
        "narrative": (
            "Demand for H100-class AI GPUs vastly exceeded supply from 2023 onward, constrained "
            "by TSMC CoWoS advanced-packaging capacity and HBM availability, forcing multi-quarter "
            "customer allocation. The GPU signal in our engine is driven by the producer-price "
            "proxy (NVIDIA/TSMC) plus demand-intent and news-severity features."
        ),
        "key_features": {
            "Price trend": "Long multi-year rise",
            "Demand signal": "Allocation / order backlog",
            "News severity (LLM)": "Sustained shortage coverage",
            "Bottleneck": "TSMC CoWoS packaging + HBM supply",
            "Primary driver": "AI-accelerator demand vs packaging capacity",
        },
    },
}


def page_case_studies():
    """Render the Historical Case Studies page."""
    st.markdown(
        '<h1 class="page-title">Historical Case Studies</h1>'
        '<div class="page-sub">Documented semiconductor tightening episodes, overlaid on '
        'the model\'s risk timeline.</div>',
        unsafe_allow_html=True,
    )

    selected = st.selectbox("Case study", list(CASE_STUDIES.keys()))

    case = CASE_STUDIES[selected]
    evt = DISRUPTION_EVENTS[case["event_idx"]] if case["event_idx"] < len(DISRUPTION_EVENTS) else None

    if evt:
        res_badges = " ".join(
            f'<span style="display:inline-flex;align-items:center;gap:5px;">'
            f'{resource_badge(r, 20)}{r}</span>' for r in evt["resources"])
        st.markdown(
            f'<div style="display:flex; gap:12px; margin:18px 0; flex-wrap:wrap;">'
            f'<div class="stat-chip"><span class="ci">{UI_SVG["calendar"]}</span><div>'
            f'<div class="lbl">Period</div><div class="val">{evt["start"]} → {evt["end"]}</div></div></div>'
            f'<div class="stat-chip"><span class="ci" style="color:#dc2626;">{UI_SVG["severity"]}</span><div>'
            f'<div class="lbl">Severity</div><div class="val" style="color:#dc2626;">{evt["severity"]:.2f}</div></div></div>'
            f'<div class="stat-chip"><span class="ci">{UI_SVG["cube"]}</span><div>'
            f'<div class="lbl">Resources</div><div class="val">{res_badges}</div></div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Timeline with disruption band ─────────────────────────
    st.markdown(sec("trend", "Disruption Timeline"), unsafe_allow_html=True)

    timeline_df, _ = load_timeline()
    fig = go.Figure()

    if evt:
        affected = evt["resources"]
        if timeline_df.empty:
            st.warning("Risk timeline not generated yet — run the export adapter.")
        else:
            for res in affected:
                res_df = timeline_df[timeline_df["resource"] == res].sort_values("date")
                fig.add_trace(go.Scatter(
                    x=res_df["date"], y=res_df["risk_score"],
                    mode="lines", name=res,
                    line=dict(color=RESOURCE_COLORS.get(res, "#888"), width=2.6),
                    hovertemplate=f"<b>{res}</b><br>%{{x|%d %b %Y}} · risk %{{y:.3f}}<extra></extra>",
                ))

        fig.add_vrect(
            x0=evt["start"], x1=evt["end"],
            fillcolor="rgba(220,38,38,0.09)",
            line=dict(color="rgba(220,38,38,0.45)", width=1.5, dash="dash"),
            annotation_text=evt["description"],
            annotation_position="top left",
            annotation_font=dict(color="#b91c1c", size=12),
        )

    cs_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("yaxis", "legend")}
    fig.update_layout(
        **cs_layout,
        height=420, title_text="", xaxis_title="", yaxis_title="Risk score",
        yaxis=dict(range=[0, 1.12], **_AXIS_GRID),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5,
                    bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=12, color="#111827")),
    )
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})

    # ── Feature Values During Disruption ──────────────────────
    st.markdown(sec("drivers", "Key Indicators During the Episode"), unsafe_allow_html=True)

    feat_table = '<table class="perf-table"><thead><tr><th>Indicator</th><th>Reading</th></tr></thead><tbody>'
    # word-boundary match so "high" doesn't fire inside unrelated words
    _elevated = {"high", "very", "strong", "sustained", "rising", "persistent",
                 "sold-out", "allocation", "backlog", "critical", "crisis", "long", "elevated"}
    for feat, val in case["key_features"].items():
        words = set(re.split(r"[^a-z]+", val.lower()))
        hot = bool(words & _elevated)
        # severity carried by a TAG + color, never hue alone (matches the app's card rule)
        tag = ('<span class="sev-tag">Elevated</span>' if hot else "")
        color = "#b91c1c" if hot else "#111827"
        feat_table += (f'<tr><td>{feat}</td>'
                       f'<td style="color:{color}; font-weight:600;">{val}{tag}</td></tr>')
    feat_table += "</tbody></table>"
    st.markdown(feat_table, unsafe_allow_html=True)
    st.caption("Qualitative drivers we actually track (price, news severity, supplier "
               "concentration) — no fabricated inventory or lead-time figures.")

    # ── Narrative ─────────────────────────────────────────────
    st.markdown(sec("drivers", "Analysis"), unsafe_allow_html=True)
    st.markdown(
        f'<div class="case-card"><span class="case-tag">Case study</span>'
        f'<h3>{selected}</h3><p>{case["narrative"]}</p></div>',
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN — SIDEBAR NAVIGATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    """Main entry point with sidebar navigation."""
    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown(
            f'<div class="on-dark" style="text-align:center; padding:16px 0 8px;">'
            f'<span style="display:inline-flex; color:#c4b5fd;">{LOGO_SVG}</span>'
            f'<div style="font-size:22px; font-weight:800; letter-spacing:3px; color:#e6e9f5; margin-top:8px;">SCRIF</div>'
            f'<div style="font-size:11px; color:#a9b2c4; letter-spacing:2.5px; margin-top:3px;">'
            f'RISK INTELLIGENCE</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["Risk Overview", "Model Performance", "Explainability", "Case Studies"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Resource status rail — score + a WORD (risk never conveyed by colour alone).
        scores, missing = load_risk_scores()
        st.markdown(
            '<div style="font-size:11px; color:#94a3b8; font-weight:700; '
            'text-transform:uppercase; letter-spacing:1.5px; margin-bottom:10px;">'
            'Resource Status</div>',
            unsafe_allow_html=True,
        )
        for res in RESOURCES:
            has = (not missing) and (res in scores)
            if has:
                score = float(scores[res])
                color = risk_hex(score)
                _, _, label = risk_color_class(score)
                right = (f'<span style="color:{color}; font-weight:700; font-size:13px;">{label}</span>'
                         f'<span class="tnum" style="color:#e6e9f5; font-weight:700; font-size:13px; '
                         f'margin-left:8px;">{score:.2f}</span>')
            else:
                right = '<span style="color:#94a3b8; font-size:12px;">Not scored</span>'
            st.markdown(
                f'<div class="on-dark" style="display:flex; justify-content:space-between; '
                f'align-items:center; gap:8px; padding:8px 0; border-bottom:1px solid rgba(148,163,184,.16);">'
                f'<span style="display:inline-flex; align-items:center; gap:8px; color:#d5dced; font-size:13px;">'
                f'<span style="--rc:{RESOURCE_COLORS.get(res)};">{resource_badge(res, 24)}</span>{res}</span>'
                f'<span style="display:inline-flex; align-items:center;">{right}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align:center; color:#94a3b8; font-size:11px; line-height:1.6;">'
            'Supply Chain Risk<br>Intelligence Framework<br>'
            '<span style="color:#8b95a8;">Datasets: Risk Overview → Downloads</span></div>',
            unsafe_allow_html=True,
        )

    # Page routing
    if page == "Risk Overview":
        page_risk_overview()
    elif page == "Model Performance":
        page_model_performance()
    elif page == "Explainability":
        page_explainability()
    elif page == "Case Studies":
        page_case_studies()


if __name__ == "__main__":
    main()
