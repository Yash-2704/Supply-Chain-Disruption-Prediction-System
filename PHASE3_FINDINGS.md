# Phase 3 — Verification Findings

_Run 2026-07-05, after Phase 1 (data) + Phase 2 (models). Honest results, including
the negative ones. Eval harnesses: `scripts/phase3_fusion_eval.py`,
`scripts/phase3_forecast_bench.py`._

## Headline

The data problem's **first half is solved**: the fusion pipeline now trains a
genuine **`data_mode = real`** model (run_id=2, 1112 real overlapping pairs,
2023–2026) where before Phase 1 it could only produce `synthetic_validation`
(44 circular pairs). SHAP explainability now unlocks for the real model.

But verification also surfaced the **next binding constraint**, and it is NOT
model quality — it is the **negative class**. Read on.

---

## 3.1 Real fusion model — LightGBM vs TabPFN

**Setup:** 5-fold stratified CV, evaluated separately on non-circular curated
labels vs the full set (which includes circular price_derived labels).

| dataset | LightGBM ROC-AUC | TabPFN ROC-AUC |
|---|---|---|
| curated_only (non-circular) | **1.00** | **1.00** |
| all (incl. circular price_derived) | 0.62 | (timed out) |

**A perfect 1.0 is a red flag, not a win.** We investigated it three ways:

1. **Not a single-feature threshold** — no individual feature separates the
   classes (price level, trend, z-score all overlap heavily between classes).
2. **Not autocorrelation leakage** — GroupKFold by (resource, month) still gives
   1.0, so it is not adjacent-week duplication across folds.
3. **It IS a homogeneous-negative artifact.** All 80 negatives come from ONE
   regime: the 2023 DRAM/NAND glut (2022-12→2023-09, only 2 of 4 resources).
   Positives span all resources 2023–2025. The model learns to recognise "the
   2023 glut regime," which is trivially separable — so any model saturates the
   metric (both LightGBM and TabPFN hit 1.0, so the eval can't even tell them
   apart).

**Generalisation probe (the informative bit):** trained on positives + only
*DRAM* negatives, the model classified 100% of *unseen NAND* 2023-glut weeks as
non-tightening (prob ≈ 0.000). So it learned a real, transferable glut signature —
**not pure memorisation** — but it still only proves "recognises the 2023-style
glut," never "distinguishes tightening from non-tightening in 2024–2025 or for
GPU/HBM," because we have **no negatives there to test it.**

**Verdict:** a real model is now trainable and behaves sensibly, but its accuracy
**cannot yet be trusted** as real-world skill. The metric is saturated by a narrow,
homogeneous negative class.

**LightGBM vs TabPFN:** indistinguishable here (both 1.0). LightGBM is the
practical choice — local (ms vs hosted API seconds/fold), SHAP-explainable, and it
keeps features on-machine. TabPFN's tiny-data edge would matter at ~30 examples, not
440; no measurable benefit here, plus it sends features to Prior Labs' API.

### → Refined data need (the real Phase-1-follow-up)
Add **non-tightening (label=0) examples that**: (a) fall in **2024–2025**, not only
2023; (b) cover **GPU and HBM**, not only DRAM/NAND; ideally (c) are temporally
interspersed with positives. Until then, treat the fusion model as a
pipeline-correct, real-data-backed model whose **discrimination is unvalidated**.

## 3.2 Forecasting — Prophet vs Holt-Winters vs Chronos

8-week holdout backtest on the weekly price_proxy series (MAPE, lower better):

| model | avg MAPE | where it wins |
|---|---|---|
| **Holt-Winters** | **13.9%** | trending memory (MU) series |
| Chronos (zero-shot) | 20.1% | stable GPU (NVDA) series |
| Prophet | 35.6% | — (lagged the sharp memory run-up) |

Caveat: DRAM/HBM/NAND share the Micron proxy, so the average weights memory 3×;
really 2 distinct series. **No model dominates** — Holt-Winters best on strong
trends, Chronos best on the stable series and impressively competitive with **zero
fitting**, Prophet weakest on the sharp run-up. This vindicates the "keep all three +
flag disagreement" design. Chronos is a worthwhile free addition; not a silver bullet.

## 3.3 Local classifiers vs the LLM

- **bart-large-mnli event_type: 33% agreement with the LLM** on 160 real rows —
  too low to substitute. Good on `price_movement` (92%), poor on `supply_disruption`
  (21%, confused with demand) and never predicts `other` (0%). **Not adopted** as an
  LLM replacement; the LLM remains the event_type source. (Textbook "pretrained ≠
  accurate on our task.")
- **FinBERT:** re-confirmed sentiment ≠ severity (a shortage and a glut both read
  "negative"). Stays an **extra feature only**, never a severity replacement.

## 3.4 Calibration & explainability

- The real model returns calibrated probabilities (isotonic) and, being `real` +
  n ≥ 30, **SHAP now activates** (correctly gated off before). Top SHAP feature is
  `price_latest` — consistent with, and a caution about, the price-dominated class
  separation from 3.1.
- Meaningful calibration assessment is **blocked by the same 3.1 issue**: with
  near-perfectly separable classes the model is pushed to 0/1, so reliability curves
  aren't informative until harder, more diverse negatives exist.

## Bottom line

| item | status |
|---|---|
| Real-data fusion model trainable | ✅ achieved (was synthetic-only) |
| Model discrimination validated | ❌ blocked by homogeneous negatives |
| Chronos forecaster | ✅ competitive, kept |
| bart as event_type substitute | ❌ 33% agreement, not adopted |
| FinBERT as extra feature | ✅ kept (not as severity) |
| TabPFN | ✅ works (hosted); no edge over LightGBM here |

**Single most valuable next step:** curate diverse **label=0** examples (2024–2025,
GPU/HBM). That — not more models — is what unlocks a trustworthy fusion model.
