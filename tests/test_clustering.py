"""Required, fully-offline tests for the dedup/clustering stage.

No real network, no real model download (stubbed embeddings), no persistent
Chroma on disk (EphemeralClient, in-memory). Run:
    python -m pytest tests/test_clustering.py
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from clustering import embed_and_cluster as ec  # noqa: E402
from clustering import vector_store  # noqa: E402
import scripts.run_clustering as run_clustering  # noqa: E402


# ---- stubbed embeddings ----------------------------------------------------
# Deterministic map from raw_text -> vector, crafted so some texts are
# near-identical (dup coverage) and others are far apart (distinct events).
_VECTORS = {
    "lawsuit A": [1.0, 0.0, 0.0, 0.0],
    "lawsuit A too": [0.98, 0.02, 0.0, 0.0],   # near-dup of lawsuit A
    "lawsuit A also": [0.97, 0.0, 0.03, 0.0],  # near-dup of lawsuit A
    "cxmt deal": [0.0, 1.0, 0.0, 0.0],         # distinct
    "lenovo backlog": [0.0, 0.0, 1.0, 0.0],    # distinct
}


def _stub_embed_fn(texts):
    return np.array([_VECTORS.get(t, [0.0, 0.0, 0.0, 1.0]) for t in texts], dtype=float)


class ClusterEmbeddingsTest(unittest.TestCase):
    def test_near_identical_group_distinct_separate(self):
        texts = ["lawsuit A", "lawsuit A too", "lawsuit A also", "cxmt deal", "lenovo backlog"]
        emb = _stub_embed_fn(texts)
        labels, approach = ec.cluster_embeddings(emb, len(texts))
        self.assertEqual(approach, "small_n_threshold")
        # The three lawsuit texts share a label; the two distinct ones don't.
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[0], labels[2])
        self.assertNotEqual(labels[0], labels[3])
        self.assertNotEqual(labels[0], labels[4])
        self.assertNotEqual(labels[3], labels[4])
        self.assertEqual(len(set(labels)), 3)  # 1 dup-cluster + 2 singletons

    def test_small_n_two_rows_no_crash(self):
        # Fewer rows than any real cluster; must not crash, must be sensible:
        # two unrelated rows -> two singleton clusters.
        emb = _stub_embed_fn(["cxmt deal", "lenovo backlog"])
        labels, approach = ec.cluster_embeddings(emb, 2)
        self.assertEqual(approach, "small_n_threshold")
        self.assertEqual(len(set(labels)), 2)

    def test_single_row(self):
        labels, approach = ec.cluster_embeddings(_stub_embed_fn(["cxmt deal"]), 1)
        self.assertEqual(labels, [0])


class CanonicalTest(unittest.TestCase):
    def test_earliest_timestamp_wins_deterministic(self):
        cluster = [
            {"signal_id": 2, "timestamp": "20260630T100000Z", "source_url": "http://b"},
            {"signal_id": 1, "timestamp": "20260630T090000Z", "source_url": "http://a"},
            {"signal_id": 4, "timestamp": "20260630T100000Z", "source_url": "http://a"},
        ]
        first = ec.select_canonical(cluster)
        second = ec.select_canonical(list(reversed(cluster)))  # order-independent
        self.assertEqual(first, "1")   # earliest timestamp
        self.assertEqual(first, second)

    def test_tie_broken_by_source_url_then_id(self):
        cluster = [
            {"signal_id": 5, "timestamp": "20260630T090000Z", "source_url": "http://z"},
            {"signal_id": 6, "timestamp": "20260630T090000Z", "source_url": "http://a"},
        ]
        self.assertEqual(ec.select_canonical(cluster), "6")  # same time -> smaller url


def _seed_db(db_path):
    db_init.init_db(db_path)
    conn = db_init.get_connection(db_path)
    for rid, name in [("dram", "DRAM"), ("gpu", "GPUs")]:
        conn.execute("INSERT INTO resource (resource_id, name, aliases) VALUES (?, ?, '[]')",
                     (rid, name))
    conn.execute("INSERT INTO producer (producer_id, name) VALUES ('micron','Micron')")

    def ins(sid, stype, ts, url, raw):
        conn.execute(
            "INSERT INTO signal (signal_id, signal_type, resource_id, timestamp, "
            "source_url, raw_text, source_name) VALUES (?, ?, 'dram', ?, ?, ?, 'x')",
            (sid, stype, ts, url, raw))

    ins(1, "news_raw", "20260630T114500Z", "http://a", "lawsuit A")
    ins(2, "news_raw", "20260630T100000Z", "http://b", "lawsuit A too")   # dup of 1
    ins(3, "news_raw", "20260629T114500Z", "http://c", "cxmt deal")       # distinct
    ins(4, "news_raw", "20260629T123000Z", "http://d", "lenovo backlog")  # distinct
    # out-of-scope rows that must never be clustered:
    ins(5, "demand_intent_raw", "20260630T090000Z", "http://e", "8-K item 8.01")
    ins(6, "capacity_expansion", "20260630T093000Z", "http://f", "capex fact")
    conn.commit()
    conn.close()


class OrchestrationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "test.db"
        os.environ["SUPPLY_CHAIN_DB"] = str(self.db_path)
        _seed_db(self.db_path)

    def tearDown(self):
        os.environ.pop("SUPPLY_CHAIN_DB", None)
        self._tmp.cleanup()

    def _run(self):
        # Ephemeral in-memory Chroma; stubbed embeddings — no disk, no download.
        return run_clustering.run(self.db_path, embed_fn=_stub_embed_fn,
                                  chroma_client=vector_store.get_client(ephemeral=True))

    def test_only_news_raw_in_membership(self):
        self._run()
        conn = db_init.get_connection(self.db_path)
        sids = {r[0] for r in conn.execute("SELECT signal_id FROM news_cluster_membership")}
        out_of_scope = conn.execute(
            "SELECT COUNT(*) FROM news_cluster_membership WHERE signal_id IN "
            "(SELECT signal_id FROM signal WHERE signal_type != 'news_raw')").fetchone()[0]
        conn.close()
        self.assertEqual(sids, {"1", "2", "3", "4"})  # only the 4 news_raw rows
        self.assertEqual(out_of_scope, 0)

    def test_every_news_raw_exactly_once(self):
        self._run()
        conn = db_init.get_connection(self.db_path)
        total = conn.execute("SELECT COUNT(*) FROM news_cluster_membership").fetchone()[0]
        dupes = conn.execute(
            "SELECT COUNT(*) FROM (SELECT signal_id FROM news_cluster_membership "
            "GROUP BY signal_id HAVING COUNT(*) > 1)").fetchone()[0]
        news_n = conn.execute(
            "SELECT COUNT(*) FROM signal WHERE signal_type='news_raw'").fetchone()[0]
        conn.close()
        self.assertEqual(total, news_n)
        self.assertEqual(dupes, 0)

    def test_exactly_one_canonical_per_cluster(self):
        self._run()
        conn = db_init.get_connection(self.db_path)
        rows = conn.execute(
            "SELECT cluster_id, SUM(is_canonical) FROM news_cluster_membership "
            "GROUP BY cluster_id").fetchall()
        conn.close()
        for cluster_id, n_canon in rows:
            self.assertEqual(n_canon, 1, f"{cluster_id} must have exactly one canonical")
        # Rows 1 & 2 are dup coverage -> same cluster; canonical is the earlier one (2).
        conn = db_init.get_connection(self.db_path)
        c1 = conn.execute("SELECT cluster_id, is_canonical FROM news_cluster_membership WHERE signal_id='1'").fetchone()
        c2 = conn.execute("SELECT cluster_id, is_canonical FROM news_cluster_membership WHERE signal_id='2'").fetchone()
        conn.close()
        self.assertEqual(c1[0], c2[0])           # same cluster
        self.assertEqual((c1[1], c2[1]), (0, 1)) # row 2 (earlier ts) canonical

    def test_idempotent_second_run(self):
        self._run()
        conn = db_init.get_connection(self.db_path)
        first = conn.execute(
            "SELECT signal_id, cluster_id, is_canonical FROM news_cluster_membership "
            "ORDER BY signal_id").fetchall()
        conn.close()
        self._run()
        conn = db_init.get_connection(self.db_path)
        second = conn.execute(
            "SELECT signal_id, cluster_id, is_canonical FROM news_cluster_membership "
            "ORDER BY signal_id").fetchall()
        count = conn.execute("SELECT COUNT(*) FROM news_cluster_membership").fetchone()[0]
        conn.close()
        self.assertEqual(first, second)  # no reassignment, same canonical
        self.assertEqual(count, 4)       # no duplicate membership rows


class VectorStoreTest(unittest.TestCase):
    def test_upsert_no_duplicate_on_repeat(self):
        import uuid
        client = vector_store.get_client(ephemeral=True)
        # Unique collection name: Chroma's in-memory client shares state across
        # instances in one process, so isolate this test's collection.
        coll = vector_store.get_collection(client, name=f"test_{uuid.uuid4().hex}")
        emb = np.array([0.1, 0.2, 0.3, 0.4])
        meta = {"resource_id": "dram", "producer_id": None,
                "timestamp": "20260630T114500Z", "source_url": "http://a"}
        vector_store.upsert_signal(coll, 1, emb, meta)
        vector_store.upsert_signal(coll, 1, emb, meta)  # same id again
        self.assertEqual(coll.count(), 1)  # not 2
        # None producer_id coerced to '' (Chroma rejects None).
        got = coll.get(ids=["1"])
        self.assertEqual(got["metadatas"][0]["producer_id"], "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
