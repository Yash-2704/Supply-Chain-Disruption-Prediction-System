# To-Do: Data Acquisition & Off-the-Shelf Models

_Planning document — nothing here is implemented yet. Created 5 July 2026._

Goal: fix the **data problem** (the fusion model can't be trained because features
and independent labels don't overlap in time) by (Phase 1) acquiring the right
data, and (Phase 2) adding free pre-trained models alongside the existing code,
then (Phase 3) verifying everything on the newly-acquired real data.

Sequencing agreed: **add data solutions + add the models first; verify them all
after more real data is in.** Everything below is additive and must keep the
project's honesty discipline (synthetic/unverified results stay labelled as such).

---

## Phase 1 — Data acquisition (unblocks the fusion model)

- [x] **1.1  Historical producer stock prices, 2023–2025 (yfinance) — HIGHEST PRIORITY, free. DONE.**
  Added `yfinance_client.get_daily_closes_range` (uncapped backfill path, daily cap
  untouched) + `scripts/backfill_historical_prices.py`. Backfilled **MU, NVDA, TSM**
  → every resource now spans 2023-01-03…2026-07-02 (842 rows each). Samsung/SK Hynix
  stay excluded (KRW/`SSNLF` corrupt); MU still covers their memory resources.
  **Result:** 333 dram / 502 hbm / 543 gpu price rows now sit INSIDE their episode
  windows (was 0). Tests: `tests/test_historical_prices.py`.

- [x] **1.2  Free real memory commodity prices (Stanford DAM CSV). DONE.**
  Added `ingestion/dam_price_client.py` + `scripts/ingest_dam_prices.py`. Loaded 636
  real USD/GB rows (DRAM 507 / NAND 122 / HBM 7) under `unit='USD_per_GB'`,
  `source='stanford_dam:<series>'` (series embedded → no month collisions).
  **Deliberately NOT auto-consumed** by the model yet: feature/forecast paths filter
  on `USD_per_share_proxy`, so this lands real data without covertly changing behavior
  or re-introducing price/label circularity (that wiring is a guarded Phase-3 step).
  Tests in `tests/test_historical_prices.py`.

- [x] **1.3  Historical news for 2023–2025 (GDELT DOC window). DONE (backfill idempotent, re-runnable).**
  Extended `gdelt_client.search_articles` with `start_date`/`end_date` (GDELT DOC
  covers 2017→present, no credentials) + `scripts/ingest_historical_news.py` walking
  month-by-month (250/query cap) into `signal` (news_raw), reusing the live tag/dedup
  path. Chose GDELT over FNSPID (multi-GB download) / GDELT-BigQuery (needs GCP creds).
  Tests: `tests/test_historical_news.py`. _(GDELT rate-limits hard, so the full 4-resource
  run is slow; idempotent, safe to re-run to top up coverage.)_

- [x] **1.4  Historical SEC filings, 2023–2025 (edgartools). DONE.**
  Added `sec_edgar_client.get_capex_history` (full date-windowed capex trajectory,
  deduped by period) + `get_filings_in_range` (native EDGAR filing_date range) +
  `scripts/ingest_sec_historical.py`. Ingested **32 capacity_expansion** (Micron/Nvidia
  quarterly capex 2023–2025) + **588 demand_intent_raw** 8-Ks (Micron/Nvidia + MSFT/
  GOOGL/AMZN/META). Samsung/TSMC correctly skipped (foreign filers, no US-GAAP/8-K).
  Tests: `tests/test_sec_historical.py`.

- [x] **1.5  More & better expert labels (incl. NAND + negatives). DONE.**
  Added a NAND tightening episode (2024H2–2025, supply-cut driver noted honestly) and
  a `negative_periods` array (2023 DRAM downturn + 2023 NAND glut → label=0) to
  `data/curated_episodes.json`; extended `curated_labels.expand_episodes` to emit
  label=0 (no lead) with a positive-wins dedup guard. Rebuilt labels: curated set now
  has **both classes** (dram 79×1/40×0, nand 48×1/40×0, gpu 122×1, hbm 113×1) and only
  **2 of 442 curated weeks lack live price coverage** (was ~all, pre-backfill). Tests
  updated in `tests/test_labels.py`.

- **Cross-cutting infra:** added `PRAGMA busy_timeout=30000` to `db_init.get_connection`
  so the concurrent ingest scripts wait for locks instead of erroring.

---

## Phase 2 — Off-the-shelf models (add alongside existing code)

- [x] **2.1  TabPFN — pre-trained tabular classifier for the FUSION stage. DONE & VERIFIED (hosted).**
  Refactored `modeling/train_fusion_model.py` to a selectable learner (`_new_estimator`,
  `--model-type lightgbm|tabpfn`); `model_type` threads into `train_and_save` (suffixed
  filename `fusion_model_<mode>_tabpfn.pkl` so it coexists with LightGBM), the bundle,
  and `training_run.notes`. `modeling/predict.py` loads by `model_type` and **gates SHAP
  off for non-tree models** (TreeExplainer is LightGBM-only).
  **Backend:** TabPFN v2 LOCAL weights are gated behind a browser license-acceptance step
  + a gated HF repo (can't complete headlessly). Resolved by using the **HOSTED
  `tabpfn-client`** authenticated via `TABPFN_TOKEN` (now in `.env.local`) — `_new_tabpfn`
  prefers the hosted client, falls back to local. Verified end-to-end with a real token:
  fit→calibrate→pickle→load→predict all succeed (server-side inference).
  ⚠️ **Note:** hosted path SENDS feature rows to Prior Labs' API (fine for our public-data
  features; flagged). License remains **non-commercial** — a commercial ship needs Prior
  Labs enterprise, or fall back to LightGBM/Chronos+LLM (permissive).

- [x] **2.2  LLM-as-reasoner — zero-label tightening judgment. DONE.**
  `modeling/llm_reasoner.py`: builds the SAME feature vector + retrieves recent scored
  signals as evidence, asks Groq/Mistral/Cerebras (via existing `llm_client`) for a
  tightening probability + rationale. Output is explicitly labelled a JUDGMENT
  (`is_calibrated=False`, `is_real_data_model=False`, `note`); never raises (error dicts
  on no-provider / unparseable / out-of-range). Smoke-tested on the real DB (groq → 0.55
  with a sensible rationale). Tests in `tests/test_phase2_models.py`.

- [x] **2.3  Chronos-Bolt — zero-shot forecasting. DONE (verified end-to-end).**
  `forecaster.run_chronos` (lazy-loaded `amazon/chronos-bolt-small`, Apache-2.0, CPU) as
  an OPTIONAL third model: `schema_forecasting.sql` gains a nullable `chronos_point`
  (idempotent `ALTER TABLE` for existing DBs); `run_forecasting` records it and flags
  Chronos-vs-Prophet divergence, but Chronos failing is **non-fatal** and
  `disagreement_pct` stays Prophet-vs-HW for continuity. Ran on the real DB (dram 1m:
  prophet 826 / HW 1278 / chronos 1126). Tests in `test_forecasting.py` + `test_phase2_models.py`.

- [x] **2.4  FinBERT — local financial sentiment (EXTRA feature only). DONE.**
  `modeling/finbert_sentiment.py` (`ProsusAI/finbert`, local, no quota): 3-class + a signed
  scalar `P(pos)-P(neg)`. Smoke test **empirically confirmed sentiment ≠ severity** — both
  a shortage AND a glut scored "negative" (−0.90 / −0.96) — so it is an ADDITIONAL feature,
  never a severity replacement. Tests in `test_phase2_models.py`.

- [x] **2.5  `facebook/bart-large-mnli` — zero-shot event_type classifier. DONE.**
  `llm_extraction/local_event_classifier.py`: zero-shot into the closed set (reuses
  `validator.EVENT_TYPES`, asserted no-drift), lazy-loaded, graceful fallback. Free
  alternative to the LLM for the `event_type` field ONLY (no severity/entities/reasoning).
  Smoke test: 3/4 correct; the one miss (a demand headline → price_movement @0.47) is a
  live reminder that **Phase 3.3 must validate it against the LLM**. Tests in `test_phase2_models.py`.

---

## Phase 3 — Verification (DONE — see `PHASE3_FINDINGS.md`)

- [x] **3.1  REAL fusion model. DONE — gate flips synthetic→real, BUT discrimination unvalidated.**
  `data_mode=real` now (1112 pairs, run_id=2); `fusion_model_real.pkl` persisted; SHAP
  unlocks. `scripts/phase3_fusion_eval.py` (5-fold CV). **Key finding:** curated-only
  ROC-AUC=1.0 for BOTH LightGBM and TabPFN — a RED FLAG, not a win: all 80 negatives are
  one homogeneous regime (2023 DRAM/NAND glut), so the model just recognises that regime.
  Not memorisation (generalises DRAM→NAND negatives), but NOT validated tightening skill.
  LightGBM chosen (fast, local, SHAP; TabPFN = no edge + hosted latency). → **Refined data
  need: label=0 examples in 2024-2025 and for GPU/HBM.**

- [x] **3.2  Forecast backtest. DONE.** `scripts/phase3_forecast_bench.py` (8wk holdout):
  Holt-Winters 13.9% MAPE (best on trending memory), Chronos 20.1% (best on stable GPU,
  competitive zero-shot), Prophet 35.6%. No model dominates → keep all three + disagreement.

- [x] **3.3  Local classifiers vs LLM. DONE — bart NOT adopted.** bart event_type only
  **33% agreement** with the LLM on 160 real rows (good on price_movement, poor on
  supply_disruption, never "other") → LLM stays the event_type source. FinBERT re-confirmed
  sentiment≠severity → extra feature only.

- [x] **3.4  Calibration & explainability. DONE.** Real model returns isotonic-calibrated
  probs; SHAP correctly activates (real + n≥30), top feature `price_latest`. Deeper
  calibration assessment blocked by the same 3.1 separability issue.

- [x] **3.5  Findings written.** `PHASE3_FINDINGS.md` (this phase); HANDOFF refreshed.

---

## Cross-cutting reminders
- **Pretrained ≠ accurate on our task.** Every added model must be spot-checked on
  semiconductor-tightening data before its output is trusted (Phase 3).
- **Keep provenance labelling.** Real vs synthetic vs judgment must stay visible in
  the DB and dashboard, exactly as now.
- **Licenses:** Chronos (Apache-2.0) and FinBERT/bart (permissive) are commercial-safe;
  **TabPFN weights are non-commercial** — flagged for any future productionisation.
- **Not started:** this is a plan only, per instruction.
