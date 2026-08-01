"""Orchestrate news_raw dedup: embed -> cluster -> canonical -> persist + Chroma.

Scope is strictly signal_type='news_raw'. Writes cluster membership to the
additive news_cluster_membership table and populates the local Chroma store.
Idempotent: INSERT OR REPLACE on signal_id + Chroma upsert by id.
Run:  python scripts/run_clustering.py
"""
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from clustering import embed_and_cluster as ec  # noqa: E402
from clustering import vector_store  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_clustering")


def run(db_path=None, embed_fn=None, chroma_client=None):
    embed_fn = embed_fn or ec.get_embedding_fn()
    conn = db_init.get_connection(db_path)
    try:
        ec.init_clustering_schema(conn)

        rows = conn.execute(
            "SELECT signal_id, timestamp, source_url, resource_id, producer_id, raw_text "
            "FROM signal WHERE signal_type = 'news_raw' ORDER BY signal_id"
        ).fetchall()
        row_count = len(rows)
        logger.info("news_raw rows found in database: %d", row_count)

        summary = {"processed": row_count, "clusters": 0, "canonical": 0,
                   "duplicates": 0, "approach": "small_n_threshold"}
        if row_count == 0:
            logger.warning("No news_raw rows to cluster; nothing to do.")
            _print_summary(summary)
            return summary

        cols = ["signal_id", "timestamp", "source_url", "resource_id", "producer_id", "raw_text"]
        records = [dict(zip(cols, r)) for r in rows]

        embeddings = embed_fn([rec["raw_text"] or "" for rec in records])
        labels, approach = ec.cluster_embeddings(embeddings, row_count)
        summary["approach"] = approach

        # Group rows by cluster label, choose canonical per cluster.
        by_label = defaultdict(list)
        for rec, lab in zip(records, labels):
            by_label[lab].append(rec)

        processed_at = datetime.now(timezone.utc).isoformat()
        for lab, members in by_label.items():
            canonical_id = ec.select_canonical(members)
            cluster_id = f"c_{canonical_id}"
            summary["clusters"] += 1
            for rec in members:
                is_canon = 1 if str(rec["signal_id"]) == canonical_id else 0
                summary["canonical"] += is_canon
                summary["duplicates"] += (1 - is_canon)
                conn.execute(
                    "INSERT OR REPLACE INTO news_cluster_membership "
                    "(signal_id, cluster_id, is_canonical, processed_at) VALUES (?, ?, ?, ?)",
                    (str(rec["signal_id"]), cluster_id, is_canon, processed_at),
                )
        conn.commit()

        # Populate Chroma with every embedding (idempotent upsert by signal_id).
        client = chroma_client or vector_store.get_client()
        collection = vector_store.get_collection(client)
        for rec, emb in zip(records, embeddings):
            vector_store.upsert_signal(collection, rec["signal_id"], emb, {
                "resource_id": rec["resource_id"],
                "producer_id": rec["producer_id"],
                "timestamp": rec["timestamp"],
                "source_url": rec["source_url"],
            })

        _print_summary(summary)
        return summary
    finally:
        conn.close()


def _print_summary(s):
    print("=" * 60)
    print(f"news_raw rows processed : {s['processed']}")
    print(f"clustering approach     : {s['approach']}"
          + ("  (HDBSCAN unreliable at this N -> deterministic near-dup fallback)"
             if s["approach"] == "small_n_threshold" else "  (row count >= HDBSCAN threshold)"))
    print(f"clusters formed         : {s['clusters']}")
    print(f"canonical rows          : {s['canonical']}")
    print(f"duplicate rows          : {s['duplicates']}")
    print("=" * 60)


if __name__ == "__main__":
    run()
