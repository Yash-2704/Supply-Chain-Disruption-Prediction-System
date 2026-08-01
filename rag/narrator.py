"""Prompt construction, citation validation, and the code-enforced disclaimer.

The LLM is asked to cite only enumerated evidence, but that request is NEVER
trusted: validate_narration cross-checks every returned citation against the
actual retrieved evidence and drops fabricated ones. The disclaimer is applied
in code by apply_disclaimer, never left to the model.
"""
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

MAX_NARRATIVE_WORDS = 120
MAX_EXCERPT_WORDS = 25  # hard cap on how much of any raw_text a citation may quote

# Fixed, hardcoded disclaimer prepended whenever the explanation is NOT backed by
# a real predictive model. Enforced in code (apply_disclaimer), never by the LLM.
SYNTHETIC_DISCLAIMER = (
    "⚠️ NOTE: This explanation is NOT backed by a validated predictive "
    "model — the current fusion model is synthetic-validation-only. Treat the "
    "following as retrieved evidence context, not a forecast."
)


def build_narration_prompt(resource_id: str, evidence_list: List[dict],
                           prediction: Optional[dict]) -> str:
    """Build a JSON-only narration prompt grounded strictly in `evidence_list`."""
    lines = [
        f"You are a supply-chain analyst. Write a short, evidence-grounded note about "
        f"the semiconductor resource \"{resource_id}\".",
        "",
        "You may reference ONLY the enumerated evidence items below. Do NOT introduce "
        "any outside knowledge, and do NOT invent sources or facts not present here.",
        "",
        "EVIDENCE (cite by signal_id):",
    ]
    for e in evidence_list:
        lines.append(f"  [signal_id={e['signal_id']}] ({e.get('source_url','')}) "
                     f"{e.get('raw_text','')}")

    if prediction is not None:
        prob = prediction.get("calibrated_probability")
        lines += ["", "MODEL CONTEXT (for reference only):",
                  f"  calibrated_probability: {prob}"]
        shap = prediction.get("top_shap_features")
        if shap:
            feats = ", ".join(f"{f['feature']}({f['contribution']:+.3f})" for f in shap)
            lines.append(f"  top_shap_features: {feats}")

    lines += [
        "",
        "Return ONLY a valid JSON object, no prose or code fences, with exactly:",
        f'  "narrative": string, at most {MAX_NARRATIVE_WORDS} words',
        f'  "citations": list of objects {{"signal_id": int, "excerpt": string}} '
        f"where excerpt is a quote of at most {MAX_EXCERPT_WORDS} words taken from that "
        "signal_id's evidence text.",
        "Every citation's signal_id MUST be one of the evidence signal_ids above.",
    ]
    return "\n".join(lines)


def _word_count(s: str) -> int:
    return len((s or "").split())


def validate_narration(parsed: dict, evidence_list: List[dict]) -> Optional[dict]:
    """Validate shape and STRIP any citation not backed by real retrieved evidence.

    Returns a cleaned {narrative, citations} dict, or None if the response is
    malformed or NO valid citations survive (a content narrative with no real
    citation is a fabrication risk and is rejected).
    """
    if not isinstance(parsed, dict):
        return None
    narrative = parsed.get("narrative")
    if not isinstance(narrative, str) or not narrative.strip():
        return None
    citations = parsed.get("citations")
    if not isinstance(citations, list):
        return None

    # Truncate an over-long narrative (length is not a trust issue, just a bound).
    words = narrative.split()
    if len(words) > MAX_NARRATIVE_WORDS:
        narrative = " ".join(words[:MAX_NARRATIVE_WORDS])

    valid_ids = {int(e["signal_id"]) for e in evidence_list}
    kept = []
    for c in citations:
        if not isinstance(c, dict):
            logger.warning("Dropping non-dict citation: %r", c)
            continue
        try:
            sid = int(c.get("signal_id"))
        except (TypeError, ValueError):
            logger.warning("Dropping citation with bad signal_id: %r", c)
            continue
        if sid not in valid_ids:  # HALLUCINATION GUARD: not in retrieved evidence
            logger.warning("Dropping fabricated citation signal_id=%s (not retrieved)", sid)
            continue
        excerpt = c.get("excerpt", "")
        if not isinstance(excerpt, str) or _word_count(excerpt) > MAX_EXCERPT_WORDS:
            logger.warning("Dropping citation signal_id=%s: excerpt over %d-word cap",
                           sid, MAX_EXCERPT_WORDS)
            continue
        kept.append({"signal_id": sid, "excerpt": excerpt.strip()})

    if not kept:  # evidence_list is never empty here (guarded upstream)
        logger.warning("No valid citations survived validation; rejecting narration.")
        return None
    return {"narrative": narrative.strip(), "citations": kept}


def apply_disclaimer(narrative: str, is_real_data_model: bool) -> str:
    """Pure: prepend the fixed disclaimer iff the model is NOT real. Unconditionally
    called by the orchestrator so a synthetic/no-model narrative can never ship
    without it."""
    if is_real_data_model:
        return narrative
    return f"{SYNTHETIC_DISCLAIMER}\n\n{narrative}"
