"""LLM-as-reasoner: a ZERO-LABEL tightening-judgment path.

When there are still too few real labels to train a trustworthy fusion model,
this asks an LLM (via the existing Groq/Mistral/Cerebras client) to reason over
the SAME feature vector the fusion model would use — plus a few retrieved recent
signals as grounding evidence — and return a tightening probability with a written
rationale.

Honesty rules baked in:
  * Output is a JUDGMENT, not a calibrated statistic — `is_calibrated=False`,
    `is_real_data_model=False`, and a `note` say so. Never present it as the
    trained model's calibrated_probability.
  * Features are shown with real values or the literal "unknown" (NaN), never
    silently zeroed — matching feature_builder's discipline.
  * Never raises: missing provider / parse failure / bad range → an error dict.

This adds no training and no schema; it complements (does not replace) the fusion
model. Callers choose when to use it (e.g. zero valid labels for a resource).
"""
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from modeling import feature_builder  # noqa: E402
from modeling.feature_builder import FEATURE_COLUMNS, _week_bounds  # noqa: E402
from llm_extraction import llm_client  # noqa: E402

logger = logging.getLogger("llm_reasoner")

_NOTE = ("LLM judgment, NOT a calibrated statistic or a trained-model output. "
         "Treat as qualitative reasoning over the feature vector + evidence.")


def _fetch_evidence(conn, resource_id, cutoff, limit=5):
    """A few most-recent severity-scored signal texts at-or-before the cutoff,
    to ground the reasoning. Returns list of short strings."""
    rows = conn.execute(
        "SELECT timestamp, signal_type, severity, raw_text FROM signal "
        "WHERE resource_id = ? AND severity IS NOT NULL AND raw_text IS NOT NULL "
        "AND timestamp <= ? ORDER BY timestamp DESC LIMIT ?",
        (resource_id, cutoff.date().isoformat(), limit)).fetchall()
    out = []
    for ts, stype, sev, text in rows:
        snippet = (text or "").strip().replace("\n", " ")
        if len(snippet) > 220:
            snippet = snippet[:220] + "…"
        out.append(f"[{str(ts)[:10]} {stype} sev={sev}] {snippet}")
    return out


def _format_features(feats: dict) -> str:
    lines = []
    for col in FEATURE_COLUMNS:
        v = feats.get(col)
        shown = "unknown" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4g}" \
            if isinstance(v, float) else str(v)
        lines.append(f"  - {col}: {shown}")
    return "\n".join(lines)


def _build_prompt(resource_name, horizon, feats, evidence):
    ev = "\n".join(f"  - {e}" for e in evidence) if evidence else "  (no scored signals available)"
    return (
        f"You are a semiconductor supply-chain analyst. Assess the probability that "
        f"{resource_name} supply will TIGHTEN (shortage risk rising) over the next "
        f"{horizon}.\n\n"
        f"Feature vector as of the target week (values are real; 'unknown' means no "
        f"data, do NOT assume zero):\n{_format_features(feats)}\n\n"
        f"Recent grounding evidence (may be sparse):\n{ev}\n\n"
        f"Weigh the evidence conservatively; when features are mostly 'unknown', keep "
        f"the probability near 0.5 and say so. Respond with STRICT JSON only:\n"
        f'{{"tightening_probability": <number 0.0-1.0>, "rationale": "<one or two '
        f'sentences citing the specific features/evidence that drove the number>"}}')


def reason_tightening(resource_id: str, week_start: str, horizon: str = "1m",
                      db_path=None, max_evidence: int = 5) -> dict:
    """Return an LLM tightening JUDGMENT for (resource, week, horizon).

    Always includes `method='llm_reasoner'`, `is_calibrated=False`,
    `is_real_data_model=False`, and `note`. On no-provider / parse failure / out-of
    -range probability, returns a dict with an `error` key (probability None)."""
    base = {"resource_id": resource_id, "week_start": week_start, "horizon": horizon,
            "method": "llm_reasoner", "is_calibrated": False,
            "is_real_data_model": False, "note": _NOTE}

    if not llm_client.available_providers():
        return {**base, "error": "no_llm_provider", "tightening_probability": None}

    conn = db_init.get_connection(db_path)
    try:
        name_row = conn.execute("SELECT name FROM resource WHERE resource_id = ?",
                                (resource_id,)).fetchone()
        resource_name = name_row[0] if name_row else resource_id
        _, cutoff = _week_bounds(week_start)
        feats = feature_builder.build_feature_row(resource_id, week_start, conn)
        evidence = _fetch_evidence(conn, resource_id, cutoff, max_evidence)

        prompt = _build_prompt(resource_name, horizon, feats, evidence)
        raw, provider = llm_client.generate_text(prompt)
        if raw is None:
            return {**base, "error": "llm_call_failed", "tightening_probability": None}

        parsed = llm_client._parse_json_object(raw)
        if not isinstance(parsed, dict) or "tightening_probability" not in parsed:
            return {**base, "error": "unparseable_llm_output",
                    "tightening_probability": None, "llm_provider": provider,
                    "raw_response": raw[:500]}
        try:
            prob = float(parsed["tightening_probability"])
        except (TypeError, ValueError):
            return {**base, "error": "non_numeric_probability",
                    "tightening_probability": None, "llm_provider": provider}
        if not (0.0 <= prob <= 1.0):
            return {**base, "error": "probability_out_of_range",
                    "tightening_probability": None, "llm_provider": provider}

        # Present features with unknowns as None (JSON-friendly), not NaN.
        clean_feats = {c: (None if (feats.get(c) is None or
                       (isinstance(feats.get(c), float) and np.isnan(feats[c])))
                       else feats[c]) for c in FEATURE_COLUMNS}
        return {**base,
                "tightening_probability": prob,
                "rationale": str(parsed.get("rationale", "")).strip(),
                "llm_provider": provider,
                "llm_model": llm_client.PROVIDERS.get(provider, {}).get("model"),
                "features_used": clean_feats,
                "evidence_count": len(evidence),
                "reasoned_at": datetime.now(timezone.utc).isoformat()}
    finally:
        conn.close()


if __name__ == "__main__":
    import json as _json
    a = sys.argv[1:]
    rid = a[0] if a else "dram"
    wk = a[1] if len(a) > 1 else "2025-01-06"
    hz = a[2] if len(a) > 2 else "1m"
    print(_json.dumps(reason_tightening(rid, wk, hz), indent=2))
