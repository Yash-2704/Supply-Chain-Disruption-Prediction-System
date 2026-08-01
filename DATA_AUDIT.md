# Data Audit — pre-Phase-3

_Run 2026-07-05 against `data/supply_chain.db` (read-only). Audit script:
`scratchpad/data_audit.py`. Two questions: (A) is the data CORRECT, and (B) is
there ENOUGH to use?_

## Verdict

- **(A) Correct:** ✅ Yes — the training-relevant data is clean. Two benign, out-of-window quirks in the (unwired) Stanford DAM series; nothing that touches the model.
- **(B) Enough:** ✅ Yes for the core problem — **440 real, NON-circular (feature,label) pairs with both classes** now exist (was 0). One processing gap remains before news becomes a strong feature (cluster → extract, below).

---

## The headline: the data problem is solved

`feature_builder.build_training_table` now returns **1112 real overlapping
(feature,label) pairs**:

| | count |
|---|---|
| total real pairs | **1112** |
| — non-circular (curated_episode) | **440** |
| — circular (price_derived) | 672 |
| label=0 / label=1 | 654 / 458 |

440 non-circular pairs with both classes is far above the training floor (50) and
eval minimum (30). Before Phase 1 this was **0**. A **real** fusion model (not
`synthetic_validation`) is now trainable — that is exactly what Phase 3.1 will test.

## Feature population in the training table

| feature | non-null | note |
|---|---|---|
| price_latest | 100.0% | strong (yfinance backfill) |
| price_trend_4w | 99.3% | strong |
| price_zscore_8w | 98.6% | strong |
| signal_count_4w | 100.0% | strong |
| hhi_country | 100.0% | structural |
| signal_severity_mean_4w | **32.6%** | improves after extraction (see gap) |
| forecast_point_1m | **0.0%** | structural NaN — see limitation |
| forecast_interval_width_1m | 0.0% | " |
| forecast_disagreement_1m | 0.0% | " |
| forecast_unstable_flag | 0.0% | " |

6/10 features are well-populated; that alone supports a real model.

---

## Correctness checks (PASS unless noted)

**price_series**
- yfinance: 3368 rows, 2023-01-03…2026-07-02, all positive & plausible ($14–$1214 = real share prices). ✅
- No duplicate `(resource,timestamp,source)` rows. ✅
- yfinance coverage inside every episode window ≥ 186 rows. ✅
- DAM in-window prices sane: dram $0.90–6.85/GB, hbm $15–18/GB, nand $0.03–0.06/GB. ✅
- ⚠️ **DAM has 198 rows > $1000/GB** — ALL pre-1988 McCallum historical data (memory really did cost millions–billions/GB then). **Zero are inside 2023-2025.** Inert (DAM not wired to features), but irrelevant clutter.
- ⚠️ **1 future-dated row** — a single HBM DAM projection ($16.50/GB, 2026-09-01). Out of window; feature builders cut off at the target week so it cannot leak.

**signal**
- No orphan `resource_id`; no duplicate `(source_url,resource)`. ✅
- news_raw: 1532+ rows (backfill still finishing), good in-window coverage (dram 147 / hbm 430 / gpu 97 / nand 54). ✅
- capacity_expansion 36 rows, demand_intent_raw 737 rows, 2023-2025. ✅

**label**
- Curated has **both classes**; NAND now present (40×0, 48×1). ✅
- price_derived: 672 ok / 32 insufficient_data. ✅

---

## Gaps / limitations before Phase 3

1. **News is landed but not yet feature-ready (needs 2 pipeline stages).**
   Only **88 of 1617** news rows are marked canonical; extraction scores only
   canonical rows, so the ~1500 backfilled articles are unscored. To turn them
   into `signal_severity_mean_4w` features:
   `run_clustering.py` (entity-resolution/dedup → mark canonical) **then**
   `run_extraction.py` (LLM severity, backlog ≈ 1500 news + 628 demand; idempotent,
   multi-run under free-tier caps).

2. **forecast_* features are structurally NaN for historical rows.** The only
   forecast run is dated 2026-07-05, so no forecast is "as-of" any 2023-2025 week.
   We never made past forecasts, so 4/10 features contribute nothing to historical
   training. Not an error — an inherent property. The model trains on the other 6.

3. **TabPFN (Phase 2/3) needs a token.** Non-blocking for the audit; see
   `TODO_data_and_models.md` §2.1 — accept license at ux.priorlabs.ai, set
   `TABPFN_TOKEN`. All other models are verified.

## Suggested cleanups (optional, low-risk)
- Bound the DAM loader to recent years (e.g. ≥ 2015) and drop future-dated rows, to
  keep `price_series` free of the ancient/projected DAM clutter. (No model impact
  today since DAM isn't wired in.)
