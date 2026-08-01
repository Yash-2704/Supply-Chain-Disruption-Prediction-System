"""Local, free zero-shot classifier for the `event_type` field only.

Uses facebook/bart-large-mnli (permissive MIT license) via transformers'
zero-shot-classification pipeline, running locally with NO API key / quota. It is
a drop-in ALTERNATIVE to the LLM for the single `event_type` field — it does NOT
produce severity, entities, confidence, or reasoning (those still require the LLM
in extract_structured). Good for clearing the low-value demand backlog at zero API
cost, or as a cheap second opinion on the LLM's event_type.

Design mirrors llm_client: lazy singleton, and it NEVER raises to the caller —
if transformers or the model weights are unavailable it logs once and returns
None, so callers can fall back to the LLM path.
"""
import logging
import threading
from typing import Optional, Tuple

from .validator import EVENT_TYPES

logger = logging.getLogger(__name__)

MODEL_NAME = "facebook/bart-large-mnli"

# bart-mnli scores natural-language hypotheses, so map each terse event_type id
# to a descriptive candidate phrase, then map the winner back to the id. Keys MUST
# equal EVENT_TYPES exactly (asserted below) so this can never drift from the
# validator's closed set.
_LABEL_HYPOTHESES = {
    "supply_disruption": "a supply disruption, shortage, or production problem",
    "demand_signal":     "a change in demand or customer buying/ordering",
    "price_movement":    "a change in price",
    "regulatory_policy": "a government regulation, policy, tariff, or export control",
    "other":             "something unrelated to supply, demand, price, or policy",
}
assert set(_LABEL_HYPOTHESES) == set(EVENT_TYPES), \
    "local_event_classifier labels drifted from validator.EVENT_TYPES"

# Ordered so the pipeline's candidate list is deterministic; index-aligned mapping
# back to ids is done via _HYP_TO_ID.
_CANDIDATES = [_LABEL_HYPOTHESES[k] for k in sorted(_LABEL_HYPOTHESES)]
_HYP_TO_ID = {v: k for k, v in _LABEL_HYPOTHESES.items()}

_pipe = None
_pipe_lock = threading.Lock()
_load_failed = False


def _get_pipeline():
    """Lazily build the zero-shot pipeline once. Returns None (logged once) if
    transformers/torch or the weights can't be loaded — callers fall back."""
    global _pipe, _load_failed
    if _pipe is not None or _load_failed:
        return _pipe
    with _pipe_lock:
        if _pipe is not None or _load_failed:
            return _pipe
        try:
            from transformers import pipeline
            _pipe = pipeline("zero-shot-classification", model=MODEL_NAME)
        except Exception as exc:
            _load_failed = True
            logger.warning("local_event_classifier: could not load %s (%s) — "
                           "callers will fall back to the LLM.", MODEL_NAME, exc)
            return None
    return _pipe


def is_available() -> bool:
    """True if the model can be loaded (attempts a lazy load)."""
    return _get_pipeline() is not None


def classify_event_type(text: str) -> Optional[Tuple[str, float]]:
    """Return (event_type, score) — event_type always in EVENT_TYPES — or None.

    None means empty input or the model is unavailable / errored. The returned
    id is guaranteed to be a member of the validator's closed set, so callers may
    use it wherever the LLM's event_type would go.
    """
    if not text or not text.strip():
        return None
    pipe = _get_pipeline()
    if pipe is None:
        return None
    try:
        result = pipe(text, _CANDIDATES, multi_label=False)
    except Exception as exc:
        logger.warning("local_event_classifier: inference failed: %s", exc)
        return None
    # transformers returns labels sorted by score descending.
    top_hyp = result["labels"][0]
    top_score = float(result["scores"][0])
    return _HYP_TO_ID.get(top_hyp, "other"), top_score
