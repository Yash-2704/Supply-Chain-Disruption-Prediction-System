"""Three-layer producer entity resolution: exact -> fuzzy -> embedding.

Layers run in order and short-circuit on the first confident match. If none
is confident the honest answer is producer_id=None, method='unresolved' —
never a forced guess. The embedding model is lazy-loaded ONLY when layer 3 is
actually reached, and the embedding function is injectable so tests never
download a real model.
"""
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALIAS_DICT_PATH = PROJECT_ROOT / "data" / "alias_dictionary.json"

# ---- thresholds ------------------------------------------------------------
# PROVISIONAL values — chosen by judgment, NOT tuned against labeled data.
# They should be recalibrated later against real labeled disagreement cases.

# Exact substring/token hit is near-certain but not literally 1.0 (aliases can
# be ambiguous across contexts), so we leave a little headroom.
EXACT_CONFIDENCE = 0.99

# rapidfuzz partial_ratio is 0-100. 88 is high enough that a 1-2 char misspelling
# of a real producer name clears it, but unrelated tokens do not.
FUZZY_THRESHOLD = 88.0

# Only fuzzy-match aliases this long or longer: short tickers/acronyms ("MU",
# "TSM") belong to the exact layer; fuzzing them causes substring false hits.
FUZZY_MIN_ALIAS_LEN = 4

# Cosine similarity (MiniLM). Loosely related prose rarely exceeds ~0.4-0.6, so
# 0.45 is a deliberately conservative last-resort gate biased toward 'unresolved'.
EMBED_THRESHOLD = 0.45


@dataclass(frozen=True)
class ResolutionResult:
    producer_id: Optional[str]
    confidence: float
    method: str  # 'exact_alias' | 'fuzzy' | 'embedding' | 'unresolved'


UNRESOLVED = ResolutionResult(producer_id=None, confidence=0.0, method="unresolved")


# ---- catalog ---------------------------------------------------------------

def load_producer_catalog(alias_dict_path=ALIAS_DICT_PATH) -> List[dict]:
    """Read alias_dictionary.json into [{producer_id, canonical_name, aliases,
    description}]. `description` is a synthesized canonical sentence used only
    by the embedding fallback layer."""
    data = json.loads(Path(alias_dict_path).read_text(encoding="utf-8"))
    catalog = []
    for p in data.get("producers", []):
        aliases = list(dict.fromkeys(p.get("aliases", [])))  # dedupe, keep order
        catalog.append({
            "producer_id": p["producer_id"],
            "canonical_name": p.get("canonical_name", p["producer_id"]),
            "aliases": aliases,
            "description": (f"{p.get('canonical_name', p['producer_id'])}, a "
                            f"semiconductor producer. Also known as "
                            f"{', '.join(aliases)}."),
        })
    return catalog


# ---- layer 1: exact alias --------------------------------------------------

def _word_boundary_match(term: str, text: str) -> bool:
    return re.search(r"\b" + re.escape(term) + r"\b", text, re.IGNORECASE) is not None


def _exact_layer(text: str, catalog) -> Optional[ResolutionResult]:
    """Pick the producer with the LONGEST matched alias (most specific),
    tie-broken by catalog order. None if nothing matches exactly."""
    best_pid = None
    best_len = -1
    for p in catalog:
        longest_hit = max(
            (len(a) for a in p["aliases"] if _word_boundary_match(a, text)),
            default=0,
        )
        if longest_hit > best_len:
            best_len, best_pid = longest_hit, p["producer_id"]
    if best_len > 0:
        return ResolutionResult(best_pid, EXACT_CONFIDENCE, "exact_alias")
    return None


# ---- layer 2: fuzzy --------------------------------------------------------

def _fuzzy_layer(text: str, catalog) -> Optional[ResolutionResult]:
    best_pid, best_score = None, 0.0
    for p in catalog:
        for alias in p["aliases"]:
            if len(alias) < FUZZY_MIN_ALIAS_LEN:
                continue
            score = fuzz.partial_ratio(alias.lower(), text.lower())
            if score > best_score:
                best_score, best_pid = score, p["producer_id"]
    if best_pid is not None and best_score >= FUZZY_THRESHOLD:
        return ResolutionResult(best_pid, round(best_score / 100.0, 4), "fuzzy")
    return None


# ---- layer 3: embedding (lazy, injectable) --------------------------------

_MODEL = None  # module-level cache; populated only on first real embedding call


def _default_embed_fn(texts: List[str]):
    """Lazy-load MiniLM and embed. Imported and instantiated ONLY when the
    embedding layer is actually reached — never at module import time."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformers model all-MiniLM-L6-v2 (first use)...")
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL.encode(texts, normalize_embeddings=True)


# Public alias so other stages (e.g. clustering) reuse the EXACT same lazy
# loader and module-level model cache, keeping all vectors mutually comparable.
# Additive only — does not change any existing behavior.
default_embed_fn = _default_embed_fn


def _cosine(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(a.dot(b) / denom) if denom else 0.0


def _embedding_layer(text, catalog, embed_fn) -> Optional[ResolutionResult]:
    fn = embed_fn or _default_embed_fn
    descriptions = [p["description"] for p in catalog]
    try:
        vectors = fn([text] + descriptions)
    except Exception as exc:  # never crash the caller on a model/embedding failure
        logger.warning("Embedding layer failed, treating as unresolved: %s", exc)
        return None
    text_vec, desc_vecs = vectors[0], vectors[1:]
    best_pid, best_sim = None, 0.0
    for p, dvec in zip(catalog, desc_vecs):
        sim = _cosine(text_vec, dvec)
        if sim > best_sim:
            best_sim, best_pid = sim, p["producer_id"]
    if best_pid is not None and best_sim >= EMBED_THRESHOLD:
        return ResolutionResult(best_pid, round(best_sim, 4), "embedding")
    return None


# ---- orchestration ---------------------------------------------------------

def resolve_producer(text: str, catalog,
                     embed_fn: Optional[Callable] = None) -> ResolutionResult:
    """Resolve the producer for `text` via exact -> fuzzy -> embedding.
    Short-circuits on the first confident layer; returns UNRESOLVED otherwise."""
    if not text or not text.strip():
        return UNRESOLVED

    exact = _exact_layer(text, catalog)
    if exact:
        return exact

    fuzzy = _fuzzy_layer(text, catalog)
    if fuzzy:
        return fuzzy

    embedding = _embedding_layer(text, catalog, embed_fn)
    if embedding:
        return embedding

    return UNRESOLVED
