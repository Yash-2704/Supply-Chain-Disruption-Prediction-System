"""Retrieve canonical news evidence for a resource from the embedded Chroma store.

Reuses the EXACT embedding function from entity_resolution.resolver so query
vectors are comparable to what clustering stored. Filters to is_canonical=1 in
the retrieval path itself — duplicate coverage never reaches the LLM.
"""
import json
import logging
from typing import List, Optional

import numpy as np

from clustering import vector_store
# REUSE the exact shared embedding function — never a second embedding path.
from entity_resolution.resolver import default_embed_fn

logger = logging.getLogger(__name__)

# Cosine floor for "relevant" evidence. Observed on real data: on-topic
# resource->headline matches sit ~0.45-0.52, off-topic ~0.07, so 0.30 cleanly
# separates them. PROVISIONAL — recalibrate against labeled relevance judgments.
MIN_SIMILARITY_THRESHOLD = 0.30

# Cap on evidence items handed to the LLM: keeps context small/focused and
# matches the tiny real evidence pool. Provisional.
MAX_EVIDENCE_ITEMS = 5


def _query_text(conn, resource_id: str) -> str:
    """Resource name + up to 2 aliases as the implicit retrieval query."""
    row = conn.execute("SELECT name, aliases FROM resource WHERE resource_id = ?",
                       (resource_id,)).fetchone()
    if not row:
        return resource_id
    name, aliases_json = row
    aliases = json.loads(aliases_json) if aliases_json else []
    terms = [name] + list(aliases)[:2]
    # dedupe, keep order
    seen, out = set(), []
    for t in terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return " ".join(out)


def _canonical_ids(conn) -> set:
    return {r[0] for r in conn.execute(
        "SELECT signal_id FROM news_cluster_membership WHERE is_canonical = 1")}


def retrieve_canonical_evidence(resource_id: str, conn, collection,
                                top_k: int = MAX_EVIDENCE_ITEMS,
                                min_similarity: float = MIN_SIMILARITY_THRESHOLD,
                                embed_fn=default_embed_fn) -> List[dict]:
    """Return up to `top_k` canonical evidence items above `min_similarity`.

    Each item: {signal_id, source_url, raw_text, similarity, timestamp}.
    Returns [] (never raises) when nothing qualifies.
    """
    total = collection.count()
    if total == 0:
        return []

    qvec = np.asarray(embed_fn([_query_text(conn, resource_id)])[0], dtype=float)
    n_fetch = min(total, max(top_k * 4, top_k))
    try:
        res = collection.query(
            query_embeddings=[qvec.tolist()], n_results=n_fetch,
            where={"resource_id": resource_id},
            include=["embeddings", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.warning("Chroma query failed for %s: %s", resource_id, exc)
        return []

    ids = res.get("ids", [[]])[0]
    embs = res.get("embeddings", [[]])[0]
    if ids is None or len(ids) == 0:
        return []

    canonical = _canonical_ids(conn)  # signal_ids stored as TEXT
    qnorm = np.linalg.norm(qvec) or 1.0
    items = []
    for sid_str, emb in zip(ids, embs):
        if sid_str not in canonical:  # ENFORCE canonical-only in retrieval
            continue
        v = np.asarray(emb, dtype=float)
        sim = float(qvec.dot(v) / (qnorm * (np.linalg.norm(v) or 1.0)))
        if sim < min_similarity:
            continue
        row = conn.execute(
            "SELECT raw_text, source_url, timestamp FROM signal WHERE signal_id = ?",
            (int(sid_str),)).fetchone()
        if not row:
            continue
        items.append({
            "signal_id": int(sid_str),
            "raw_text": row[0] or "",
            "source_url": row[1],
            "timestamp": row[2],
            "similarity": round(sim, 4),
        })

    items.sort(key=lambda x: x["similarity"], reverse=True)
    return items[:top_k]
