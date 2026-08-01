# Verification Findings — Supply Chain Disruption Prediction System (MVP)

**Audited fresh on 2026-07-04 by an independent QA pass that trusted no prior
summary and ran everything itself.** Headline: **the code is in good structural
health — all 156 automated tests pass right now, and all 14 build stages exist,
integrate, and run end-to-end when invoked as they are meant to be (one process
per stage).** However, this system has **not** produced a single real,
non-synthetic, evidence-backed prediction, and it cannot yet, because three
real-data gaps remain exactly as prior stages honestly reported them: (1) **no
LLM API key has ever been used**, so `signal.severity`/`confidence` are entirely
NULL and the `event_extraction` table has never even been created; (2) the
**`price_series.resource_id` nullability fix was never applied** to the real
database (still `NOT NULL`); and (3) the **only trained model is
`synthetic_validation`**, never a real one. Everything that is *code* is
verified working; everything that requires *real-world data or a credential* is
still pending human action. This is a healthy MVP with a clearly bounded set of
known gaps — it is **not** "done," "production-ready," or a validated predictor,
and nothing below should be read as claiming otherwise.

Real database checksum was `478065930a715b27438a0dc2ced53b16` **before and after
this entire audit** — no verification step mutated real data, real Chroma, or the
real trained model.

---

## 1. Test suite status

Ran the full suite fresh (`verification/run_full_test_suite.sh`):

**156 passed, 0 failed, 0 errors** (5 warnings, all third-party deprecations —
chromadb/asyncio and shap/lightgbm — none from this project's code).

Per-file (freshly observed, not transcribed):

| Test file | Result |
|---|---|
| test_foundation.py | 9 passed |
| test_ingest_news.py | 13 passed |
| test_ingest_sec.py | 10 passed |
| test_ingest_market.py | 11 passed |
| test_entity_resolution.py | 11 passed |
| test_clustering.py | 10 passed |
| test_llm_extraction.py | 19 passed |
| test_forecasting.py | 10 passed |
| test_labels.py | 9 passed |
| test_fusion_model.py | 10 passed |
| test_rag_explanation.py | 13 passed |
| test_dashboard_data.py | 9 passed |
| test_dashboard_formatters.py | 11 passed |
| test_backtest.py | 11 passed |

All 14 stages have a corresponding test file, and every file passes. No failures
to describe.

---

## 2. Schema and data audit

Freshly observed from the real `data/supply_chain.db` via
`verification/schema_audit.py` (PRAGMA table_info on every table + direct counts).

**13 tables present** (not 14 — see NOTE 7 below): capacity_record(0),
explanation(4), facility(0), forecast_result(20), label(390),
news_cluster_membership(11), prediction(2), price_series(360), producer(5),
producer_resource(11), resource(4), signal(395), training_run(1).

Core seed integrity: **4 resources** (dram, hbm, nand, gpu), **5 producers**,
11 producer_resource links. `signal` breakdown: news_raw=11, demand_intent_raw=380,
capacity_expansion=4. `label`: curated_episode=314, price_derived=76. Forecast
statuses: ok=13, insufficient_data=4, unstable_disagreement=3.

### Resolution of every carried-forward "NOTE"

- **NOTE 4 — `price_series.resource_id` nullability fix: NOT APPLIED.**
  Confirmed via `PRAGMA table_info(price_series)`: `resource_id` is still
  `TEXT NOT NULL`. Consequence (unchanged): resource-agnostic World Bank macro
  rows (which need `resource_id = NULL`) still cannot be stored. This is a
  genuine, still-open item — a fix was discussed and a follow-up task was
  spawned in an earlier stage, but it was **never actually applied** to the
  database. See Action Item #2.

- **NOTE 7 — `signal.severity`/`confidence`: ENTIRELY NULL.** Confirmed:
  **0 of 395** signal rows have a non-null `severity`; 0 have non-null
  `confidence`. Corroborating structural finding: the **`event_extraction`
  table does not exist** in the real database. That table is created lazily by
  `scripts/run_extraction.py` on its first run — its absence is direct evidence
  that the LLM extraction stage's live orchestration **has never been executed
  against the real DB** (because no API key was ever set). The extraction *code*
  is fully tested with mocks (19 passing tests); it has simply never processed
  real data. See Action Item #1.

- **NOTE 10 — only `synthetic_validation` model exists.** Confirmed:
  `training_run` holds exactly **1 row**, `data_mode='synthetic_validation'`,
  `n_examples=300`. The saved model on disk is
  `models/fusion_model_synthetic_validation.pkl` (no `..._real.pkl` exists). The
  two `prediction` rows both trace to this synthetic run. See Action Item #4.

Other contentious columns: `label` sources are exactly `{curated_episode,
price_derived}` with statuses `{ok, insufficient_data}`; **nand has 0
curated_episode rows** (as intended — no fabricated episode). `explanation`
statuses are `{insufficient_evidence(2), llm_error(2)}`, all with
`is_real_data_model=0` (no real narratives exist yet — llm_error because no LLM
key). On-disk artifacts all present: `data/chroma/` (11 vectors),
`data/entity_resolution_log.jsonl`, `data/curated_episodes.json`, all six
`schema_*.sql` files, and the model file.

---

## 3. Integration check results

Ran `verification/integration_check.py` against a **fresh temp copy** of the
real DB, each stage as its **own subprocess** with `SUPPLY_CHAIN_DB` pointed at
the copy (this is how the pipeline is actually invoked). Real DB, real Chroma,
and real `models/` were snapshotted and restored; real DB md5 verified identical
afterward.

| Check | Result | Detail |
|---|---|---|
| Entity resolution reprocess + idempotent | **PASS** | 11 news_raw rows; 2nd run byte-identical (no drift) |
| Clustering membership == news_raw count | **PASS** | news_raw=11, membership=11 |
| Forecasting 4 resources × 2 series + valid statuses | **PASS** | 8 (resource,series) combos; statuses ⊆ valid set |
| Labels unique-constraint holds + rows present | **PASS** | total=390, curated=314, duplicate keys=0 |
| Fusion training runs (mode reported honestly) | **PASS** | produced `synthetic_validation`, n_examples=300 |
| Dashboard `get_leaderboard()` one-per-resource | **PASS** | 4 rows for 4 resources, no exception |
| **Isolation: real DB unchanged (md5)** | **PASS** | md5 identical before/after |

Additionally verified first-hand outside the script: the **Streamlit app**
(`dashboard_ui/app.py`) renders all three views (Leaderboard / Resource detail /
Alert feed) with **0 exceptions** via headless AppTest; the **backtest harness**
runs, self-labels as SYNTHETIC-FIXTURE-only, and leaves the real DB md5 unchanged.

### FINDING — in-process chaining of the heavy native stages segfaults

An early version of the integration check imported and ran **every** stage in a
**single long-lived Python process**. It **segfaulted (exit 139)** during fusion
training, immediately after clustering (which loads PyTorch via
sentence-transformers) and forecasting had run in the same process. Root cause:
a native **OpenMP runtime conflict** — PyTorch's bundled libomp and LightGBM's
libomp coexisting in one process. Verified that fusion training runs cleanly in
its own process, and that the test suite (which does not chain torch+LightGBM in
one long-lived process) passes. **This is not a defect in normal operation:** the
pipeline is designed and documented to run as separate `python scripts/X.py`
invocations, which the corrected integration check exercises successfully.
**Action for developers: do not build a single-process orchestrator that chains
the embedding/clustering stage and the LightGBM fusion stage together** — keep
them in separate processes (or resolve the OpenMP duplication explicitly, e.g.
via a single shared OpenMP runtime) before doing so.

---

## 4. Known data-completeness gaps

These are all **real, correct, well-tested code paths that have not yet been run
against real data** due to volume/coverage/credential limits. None is a code bug.

1. **No real LLM severity/confidence scores.** `signal.severity` and
   `signal.confidence` are 100% NULL; `event_extraction` table not yet created.
   The Gemini/Groq extraction code is fully mock-tested but has never processed a
   real article. Downstream effect: the `demand_intent_activity` forecast series
   is always `insufficient_data`, and no real narrative explanations exist
   (all `explanation` rows are `llm_error`/`insufficient_evidence`).

2. **No real (non-synthetic) fusion model.** Real (feature, label) overlap is 44
   pairs, all from `price_derived` labels, and those are circular with the price
   features; 0 independent curated-label overlap (curated episodes are 2023-2025,
   ingested price data is 2026-only). This is below the training threshold, so
   the pipeline correctly refuses and produces a `synthetic_validation` model.

3. **Thin, recent-only news volume.** Only 11 raw news signals (8 canonical after
   dedup), all recently dated. No genuinely historical (2023-2025) news exists in
   the DB.

4. **World Bank macro data unstorable.** Because of NOTE 4, resource-agnostic
   macro rows cannot be inserted into `price_series`; that feed is built and
   parses correctly but has no home in the schema as-is.

5. **Backtest is mechanism-validation only.** The backtest harness explicitly
   used synthetic fixtures (genuine historical news is unobtainable via free-tier
   GDELT/NewsAPI). It validates as-of/isolation machinery, **not** any real
   historical prediction.

---

## 5. What this system can honestly claim today — and what it cannot

**Can claim (genuinely verified):**
- The full 14-stage pipeline exists and every stage's automated tests pass right
  now (156/156).
- The stages integrate end-to-end when run as intended (separate processes):
  ingestion schema → entity resolution → clustering → forecasting → labeling →
  fusion training → dashboard all chain correctly against a database, proven on a
  copy without touching real data.
- The system is rigorously honest about its own limitations by construction:
  synthetic models are labeled synthetic, missing data surfaces as explicit
  statuses/NULLs (never fabricated zeros), LLM citations are validated against
  retrieved evidence, and the backtest refuses to claim real validation.
- Database isolation works: multiple orchestration and backtest runs left the
  real DB byte-for-byte identical.

**Cannot claim (has NOT been done):**
- It has **not** ingested or scored any real LLM-derived severity data — the
  extraction stage has never run against real data (no API key).
- It has **not** trained a real predictive model — only a synthetic-validation
  one exists.
- It has **not** produced a single real, evidence-backed shortage prediction or a
  real narrative explanation.
- It has **not** validated any real historical outcome — the DRAM 2024-25
  "backtest" is a mechanism test on synthetic fixtures, by explicit design.
- It is **not** production-ready and has **not** demonstrated real-world
  predictive value. It is an MVP whose *machinery* is verified and whose
  *real-data proof* is still pending the actions below.

---

## 6. Action items required from the human operator

A person doing only these items (no code changes required for #1, #3, #4;
#2 is a small schema migration) would move the system from
"machinery verified" toward "real prediction demonstrated":

1. **Supply an LLM API key and run extraction.** Set `GEMINI_API_KEY` (free tier;
   optionally `GROQ_API_KEY` for fallback), then run
   `python scripts/run_extraction.py`. This populates `signal.severity`/
   `confidence`, creates the `event_extraction` table, and enables real
   `demand_intent_activity` forecasts and real narrative explanations
   (`python scripts/generate_explanation.py`). *This is the single highest-value
   unblock.*

2. **Apply the `price_series.resource_id` nullability migration.** The fix
   discussed in the market-data stage was never applied (confirmed: still
   `NOT NULL`). A migration must recreate `price_series` with a nullable
   `resource_id` (SQLite cannot drop a NOT NULL constraint in place) and copy
   existing rows, so World Bank macro rows can be stored. Until then, macro
   context is collected but unstorable.

3. **Run a wider / repeated ingestion pass for volume.** Current news coverage is
   ~11 rows, recent-only. Re-running `scripts/ingest_news.py` over time (and with
   a `NEWSAPI_KEY`) accumulates more canonical evidence, which improves
   clustering, extraction, and RAG retrieval quality.

4. **Understand that a real fusion model needs independent, time-overlapping
   labels.** Even after #1, a real model requires ≥ the training threshold of
   non-circular (feature, label) pairs whose feature window overlaps the label
   window. Today's curated labels (2023-2025) do not overlap today's price
   features (2026). This needs either genuinely historical feature ingestion or
   accumulated forward data over time — it is a data-availability constraint, not
   a code fix.

5. **For a genuine historical backtest, obtain a historical news source.** Free-
   tier GDELT (recent-window) and NewsAPI (~30-day free tier) cannot supply real
   2023-2025 news. A paid archive/historical news API would be required to turn
   the backtest from mechanism-validation into a real historical validation.

---

*Verification artifacts: `verification/run_full_test_suite.sh`,
`verification/schema_audit.py`, `verification/integration_check.py`. All were run
against the real database read-only or against isolated temporary copies; the
real database's md5 was identical before and after the entire audit.*
