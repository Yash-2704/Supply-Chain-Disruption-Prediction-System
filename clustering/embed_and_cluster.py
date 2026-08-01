"""Core clustering logic: embedding-fn reuse, small-N-aware clustering, canonical.

Clustering degrades gracefully on tiny corpora: below HDBSCAN_MIN_ROWS we use a
deterministic near-duplicate connected-components fallback instead of HDBSCAN,
because density-based clustering is unreliable (near-all-noise) on a dozen points.
"""
import logging
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

# Reuse the EXACT embedding loader from the resolver — never a second one.
from entity_resolution.resolver import default_embed_fn

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLUSTERING_SCHEMA_PATH = PROJECT_ROOT / "schema_clustering.sql"

# ---- thresholds (PROVISIONAL — recalibrate against labeled near-dup pairs) --

# Below this many rows, HDBSCAN's density estimate is unreliable and collapses
# most points to noise; use the deterministic near-dup fallback instead.
HDBSCAN_MIN_ROWS = 40

# Cosine similarity at/above which two articles are treated as near-duplicate
# coverage of the same event. MiniLM paraphrases of one story typically exceed
# ~0.8; 0.83 is a conservative starting point, NOT a tuned final value.
NEAR_DUP_THRESHOLD = 0.83

# HDBSCAN params for the large-N path (smallest meaningful duplicate cluster = 2).
HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES = 1


def get_embedding_fn() -> Callable[[List[str]], "np.ndarray"]:
    """Return the resolver's shared lazy-loading embedding function."""
    return default_embed_fn


def init_clustering_schema(conn) -> None:
    """Create news_cluster_membership from schema_clustering.sql (idempotent)."""
    conn.executescript(CLUSTERING_SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()


# ---- clustering ------------------------------------------------------------

def _normalize(embeddings: "np.ndarray") -> "np.ndarray":
    arr = np.asarray(embeddings, dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _connected_components(sim: "np.ndarray", threshold: float) -> List[int]:
    """Union-Find over the near-duplicate similarity graph. Deterministic labels
    (each component labeled by its smallest member index)."""
    n = sim.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)  # attach to smaller index (stable)

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                union(i, j)
    return [find(i) for i in range(n)]


def cluster_embeddings(embeddings, row_count: int) -> Tuple[List[int], str]:
    """Return (labels, approach). approach is 'small_n_threshold' or 'hdbscan'.

    Every row always receives a cluster label (no unassigned rows); rows with no
    near-duplicate neighbor form honest singleton clusters.
    """
    if row_count == 0:
        return [], "small_n_threshold"
    if row_count == 1:
        return [0], "small_n_threshold"

    norm = _normalize(embeddings)

    if row_count < HDBSCAN_MIN_ROWS:
        sim = norm @ norm.T
        labels = _connected_components(sim, NEAR_DUP_THRESHOLD)
        return labels, "small_n_threshold"

    # Large-N path: HDBSCAN on normalized vectors (~cosine); noise -> singletons.
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
    )
    raw = clusterer.fit_predict(norm)
    labels, next_singleton = [], max(int(raw.max()), -1) + 1
    for lab in raw:
        if lab == -1:  # noise point becomes its own singleton cluster
            labels.append(1000 + next_singleton)
            next_singleton += 1
        else:
            labels.append(int(lab))
    return labels, "hdbscan"


# ---- canonical selection ---------------------------------------------------

def select_canonical(cluster_rows: List[dict]) -> str:
    """Return the canonical signal_id (as str) for a cluster.

    Rule: earliest timestamp wins; ties broken by lexicographically smallest
    source_url, then smallest signal_id. Deterministic and stable across runs.
    """
    def sort_key(r):
        return (
            r.get("timestamp") or "",
            r.get("source_url") or "",
            int(r["signal_id"]) if str(r["signal_id"]).isdigit() else str(r["signal_id"]),
        )

    winner = min(cluster_rows, key=sort_key)
    return str(winner["signal_id"])
