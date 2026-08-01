"""Local, free financial-sentiment scoring (FinBERT) — an EXTRA feature only.

Uses ProsusAI/finbert via transformers, locally, with no API key/quota. Produces
a 3-class sentiment (positive/negative/neutral) plus a signed scalar in [-1, 1]
for use as an ADDITIONAL model feature.

IMPORTANT — sentiment is NOT severity. FinBERT measures the tone of financial
text, which is a DIFFERENT quantity from the 1–10 tightening `severity` the LLM
produces. This module must never overwrite or stand in for severity; it only adds
an orthogonal signal. (A glut and a shortage can both read as "negative" tone.)

Never raises to the caller: if transformers/weights are unavailable it logs once
and returns None, so feature building degrades gracefully (feature stays NaN).
"""
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"

# Signed weight per FinBERT class, to collapse the 3-way distribution into one
# scalar feature: score = P(positive) - P(negative). Neutral contributes 0.
_SIGN = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

_pipe = None
_pipe_lock = threading.Lock()
_load_failed = False


def _get_pipeline():
    global _pipe, _load_failed
    if _pipe is not None or _load_failed:
        return _pipe
    with _pipe_lock:
        if _pipe is not None or _load_failed:
            return _pipe
        try:
            from transformers import pipeline
            # return_all_scores so we can compute the signed scalar from the full
            # distribution rather than only the top label.
            _pipe = pipeline("text-classification", model=MODEL_NAME, top_k=None)
        except Exception as exc:
            _load_failed = True
            logger.warning("finbert_sentiment: could not load %s (%s) — "
                           "sentiment feature will be NaN.", MODEL_NAME, exc)
            return None
    return _pipe


def is_available() -> bool:
    return _get_pipeline() is not None


def score_sentiment(text: str) -> Optional[dict]:
    """Return {label, score, positive, negative, neutral} or None.

    `score` is P(positive) - P(negative) in [-1, 1]; `label` is the argmax class.
    None means empty input or the model is unavailable/errored. This is an EXTRA
    feature — callers must not treat `score` as a tightening severity.
    """
    if not text or not text.strip():
        return None
    pipe = _get_pipeline()
    if pipe is None:
        return None
    try:
        out = pipe(text, truncation=True)
    except Exception as exc:
        logger.warning("finbert_sentiment: inference failed: %s", exc)
        return None
    # transformers may return [[{label,score},...]] (batched) or [{...}].
    dist = out[0] if out and isinstance(out[0], list) else out
    probs = {d["label"].lower(): float(d["score"]) for d in dist}
    signed = probs.get("positive", 0.0) - probs.get("negative", 0.0)
    label = max(probs, key=probs.get) if probs else "neutral"
    return {
        "label": label,
        "score": signed,
        "positive": probs.get("positive", 0.0),
        "negative": probs.get("negative", 0.0),
        "neutral": probs.get("neutral", 0.0),
    }
