# Supply Chain Disruption Prediction System

An early-warning system for **semiconductor supply tightening** — DRAM, NAND, High-Bandwidth Memory (HBM) and GPUs. It turns real public signals (prices, news, filings) into a weekly risk score per resource, benchmarks several models to produce it, and presents everything in an interactive Streamlit dashboard.

> **Read the scores as decision support, not certainty.** They are model estimates on real signals; the model's ability to separate tightening from non-tightening is still being validated on more diverse examples (see *Honest limitations* below).

## What it does

For each resource, every week, the engine:

1. **Ingests real public signals** — producer stock/commodity prices (yfinance), historical news (GDELT + NewsAPI), and SEC filings (EDGAR), stored in SQLite.
2. **Turns news into a number** — near-duplicate stories are clustered, then an LLM scores each story 1–10 for tightening **severity**, making news a real numeric feature.
3. **Builds 10 weekly features** per resource — price level / trend / z-score, a price forecast + its uncertainty + model disagreement, news-severity mean & count, and supplier concentration (HHI).
4. **Labels** each week (`1` = tightening under way, `0` = not) from hand-curated episodes.
5. **Trains & benchmarks 4 models** on a **chronological** (time-ordered) train/test split — no future leakage — and explains predictions with **SHAP**.

## Results (held-out chronological test set, 138 weeks)

| Model | F1 | ROC-AUC | Notes |
|---|---|---|---|
| **XGBoost** | **0.71** | **0.87** | Best overall — balanced precision/recall |
| LightGBM | 0.46 | 0.82 | High precision, low recall (conservative) |
| TabPFN | 0.23 | 0.77 | Very conservative; rarely false-alarms |
| FT-Transformer | 0.56 | 0.38 | Underfits at this data size (deep net needs more data) |

Gradient-boosted trees win, which is the expected result for a small, tabular, mixed-signal problem — consistent with the tabular-ML literature.

## The dashboard

Four pages: **Risk Overview** (current scores + timeline), **Model Performance** (benchmark table, ROC/PR/confusion), **Explainability** (global + per-resource SHAP), and **Case Studies** (documented episodes).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env.local   # add your own API keys
python scripts/run_risk_dashboard.py   # http://localhost:8501
```

To regenerate the dashboard's data from the engine: `python scripts/build_dashboard_data.py`.

## Repository layout

- `ingestion/`, `entity_resolution/`, `clustering/`, `llm_extraction/` — the data + news-understanding pipeline
- `forecasting/`, `labeling/`, `modeling/` — features, labels, and the 4-model benchmark
- `risk_dashboard/` — the Streamlit dashboard (the approved build)
- `scripts/` — data builder + launcher · `tests/` — offline tests · `schema*.sql` — database schema

## Honest limitations

The negative (non-tightening) examples are concentrated in the 2023 memory glut, so the chronological split is partly time-confounded. The model **ranks** tightening weeks well (AUC ≈ 0.87) but its raw score is **uncalibrated** and tends to saturate near 0/1 — treat it as "does this week resemble a known tightening window?" rather than a calibrated probability. The clear next step is broadening negatives across more periods, then calibrating the score.

## Configuration

API keys are read from environment variables (see `.env.example`) — never hardcoded. `.env.local` is git-ignored.
