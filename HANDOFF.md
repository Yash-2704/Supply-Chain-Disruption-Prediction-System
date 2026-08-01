# Handoff — Supply Chain Disruption Prediction System

_Last updated: 2026-07-04. Written for the human operator. Covers the last two
work sessions and exactly what is still needed from you._

---

## TL;DR — where the system is right now

The pipeline is code-complete and **157/157 automated tests pass**. Real data is
now flowing where it used to be empty: real LLM severities, real explanations,
and a much larger real news corpus. The model is still honestly labeled
`synthetic_validation` (a real predictor needs data that no API key can supply —
see "What I still need from you"). Nothing here claims production-readiness.

**Live database snapshot (data/supply_chain.db):**

| Item | Value |
|---|---|
| News articles (`news_raw`) | **86** (was 11) |
| Canonical news events (deduped) | **48** (was 8) |
| News with LLM severity scored | 8 (the rest await an extraction pass) |
| `signal.severity` populated | 50 rows (news + demand) |
| `event_extraction` audit rows | 50 |
| Real explanations (`status=ok`) | dram, hbm |
| Demand-intent forecasts | dram/hbm/nand `ok`, gpu `insufficient` |
| `price_series.resource_id` | **nullable** (migration applied) |
| Trained model | `synthetic_validation` only |

Backups of the DB were taken before each mutating step, in `backups/`.

---

## What was done LAST session (the independent audit — stage 15)

A from-scratch QA pass that trusted no prior claim:
- Built `verification/run_full_test_suite.sh`, `verification/schema_audit.py`,
  `verification/integration_check.py`, and the report `verification/FINDINGS.md`.
- Ran the full suite fresh and audited the real DB table-by-table.
- Confirmed the three known gaps were still open (severity all-NULL, model
  synthetic-only, `price_series.resource_id` fix not applied).
- **Found** that chaining every heavy native stage (PyTorch + LightGBM) in one
  process segfaults — so the integration check runs each stage as its own
  subprocess (how the pipeline is actually invoked). Proved the real DB is
  untouched by verification (md5 identical before/after).

## What was done THIS session

1. **Applied the `price_series.resource_id` nullability migration** — the fix
   that was discussed but never applied. New `scripts/migrate_price_series_nullable.py`
   (in-place, 360 rows preserved), `schema.sql` updated for fresh installs, and
   the three tests that encoded the old NOT-NULL reality updated. World Bank macro
   rows are now storable.
2. **Ran real LLM extraction** against the live DB → populated 50 real severities,
   created the `event_extraction` table. **Finding:** Gemini's free tier for
   `gemini-2.5-flash` is only **5 requests/day** — it exhausted immediately and
   the fallback provider carried the rest.
3. **Generated real explanations** — dram and hbm now have real narratives
   (with the code-enforced synthetic-model disclaimer + validated citations);
   gpu/nand honestly `insufficient_evidence`.
4. **Refreshed forecasting** — the demand-intent series lit up from real severity
   data (3 of 4 resources now `ok`, previously all `insufficient_data`).
5. **Replaced the LLM providers** (per your instructions):
   - **Removed Gemini entirely** and **removed OpenRouter entirely**.
   - Provider stack is now **Groq (primary) → Mistral → Cerebras (fallback)**,
     all OpenAI-compatible via `requests` (no SDKs).
   - **Empirically ranked** them (`verification/llm_provider_quota_test.py`):
     Groq (1000 req/day, fastest, no 429s) > Mistral (50 req/min, reliable) >
     Cerebras (2400 req/day + 1M tok/day but only 5 req/min — biggest daily
     budget, needs pacing). OpenRouter's free pool was exhausted/429 → dropped.
   - **Added multi-key rotation**: set `GROQ_API_KEY_2`, `GROQ_API_KEY_3`,
     `CEREBRAS_API_KEY_2`, etc. and the client automatically rotates to the next
     key on a 429. This is ready for the extra keys you'll supply.
6. **Improved news ingestion** — GDELT client now uses a broader 1-month window,
   pulls up to 250 records, and **retries with backoff** (GDELT rate-limits
   hard). Ran it live: news grew **11 → 86**, canonical events **8 → 48** after
   dedup/clustering. (GDELT still rate-limits most queries per run, so re-running
   accumulates more coverage for the other resources.)

---

## What I still need from YOU to make the system complete

### Keys / credentials
1. **Extra provider keys for rotation** (you offered these). Add them as
   `GROQ_API_KEY_2`, `MISTRAL_API_KEY_2`, `CEREBRAS_API_KEY_2`, … — the client
   already rotates across them automatically. More keys = more daily quota to
   process the demand backlog.
2. **A `NEWSAPI_KEY`** (optional but valuable). GDELT alone is heavily
   rate-limited (only ~1 query/run succeeds). NewsAPI's free tier (~30-day
   window) would add a reliable second news source — the client is already wired
   for it, it just needs the key in the environment.

### A decision (uses your quota — I'll do it on your go-ahead)
3. **Run an extraction pass over the 48 canonical news + demand backlog.** Only
   8 of the 48 canonical news events have severity so far; scoring the rest lights
   up more real explanations and richer demand forecasts. With the new stack this
   runs on Groq/Mistral/Cerebras. I recommend I add simple **Cerebras pacing**
   (respect its 5/min cap) so its large 2400/day budget can clear the ~330-row
   demand backlog cheaply — say the word.

### Things NO credential can fix (structural — for your awareness)
4. **A real (non-synthetic) fusion model.** Blocked by data time-overlap: the
   curated labels are 2023-2025 while the ingested price features are 2026-only,
   so there are no independent overlapping (feature, label) pairs to train on.
   This needs genuinely historical feature ingestion or forward-accumulated data
   over time — not a key.
5. **A genuine historical backtest.** Needs a **paid historical-news archive API**
   (free GDELT/NewsAPI can't reach 2023-25). Until then the backtest harness is
   mechanism-validation only, by explicit design.

---

## Immediate next actions (ready to run on request)
- Add Cerebras pacing + run extraction to score the 48 canonical news and clear
  the demand backlog.
- Re-run news ingestion a few more times to accumulate dram/hbm/nand coverage
  (GDELT rate-limits per run, so volume grows across runs).
- Once real news severities are richer, re-run forecasting + explanations to
  refresh the dashboard.

🔐 **Rotate every API key you have pasted into chat** once you're done — they are
exposed in plaintext.
