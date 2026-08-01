"""Required, fully-offline tests for producer entity resolution.

No real network, and NO real embedding-model download: layer 3 is exercised
only via a stubbed embed_fn returning crafted vectors. Run:
    python -m pytest tests/test_entity_resolution.py
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from entity_resolution import resolver, reprocess_signals  # noqa: E402
from entity_resolution.resolver import resolve_producer, load_producer_catalog  # noqa: E402


CATALOG = load_producer_catalog()


def _pid_index(producer_id):
    """Index of a producer in CATALOG (for crafting one-hot embedding vectors)."""
    for i, p in enumerate(CATALOG):
        if p["producer_id"] == producer_id:
            return i
    raise KeyError(producer_id)


def _one_hot_embed_fn(target_pid, sim=0.9):
    """Return an embed_fn where the text aligns with `target_pid`'s description
    vector at ~`sim` cosine and is orthogonal to the rest. First input is the
    text, remaining inputs are the catalog descriptions (resolver's contract)."""
    dim = len(CATALOG) + 1
    target_col = _pid_index(target_pid) + 1  # +1 because col 0 is the text's own axis

    def embed_fn(texts):
        import numpy as np
        vecs = []
        for i, _ in enumerate(texts):
            v = np.zeros(dim)
            if i == 0:  # the text: put `sim` weight on the target axis
                v[0] = (1 - sim ** 2) ** 0.5
                v[target_col] = sim
            else:       # description i-1: unit vector on its own axis
                v[i] = 1.0
            vecs.append(v)
        return np.array(vecs)
    return embed_fn


def _low_sim_embed_fn(texts):
    """Every description is orthogonal to the text -> cosine 0 everywhere."""
    import numpy as np
    dim = len(texts) + 1
    vecs = []
    for i, _ in enumerate(texts):
        v = np.zeros(dim)
        v[i] = 1.0
        vecs.append(v)
    return np.array(vecs)


class ResolverLayerTest(unittest.TestCase):
    def test_layer1_exact_alias(self):
        r = resolve_producer("Nvidia unveils new AI accelerator lineup", CATALOG)
        self.assertEqual(r.producer_id, "nvidia")
        self.assertEqual(r.method, "exact_alias")
        self.assertGreaterEqual(r.confidence, 0.9)

    def test_layer1_picks_most_specific_alias(self):
        # "SK Hynix Inc." (long, specific) should win over any incidental short hit.
        r = resolve_producer("SK Hynix Inc. reported record HBM demand", CATALOG)
        self.assertEqual(r.producer_id, "sk_hynix")
        self.assertEqual(r.method, "exact_alias")

    def test_layer2_fuzzy_catches_misspelling(self):
        # "Micronn Technology" — misspelled, not an exact alias; fuzzy should catch it.
        r = resolve_producer("Analysts say Micronn Technlogy will expand capacity", CATALOG)
        self.assertEqual(r.method, "fuzzy")
        self.assertEqual(r.producer_id, "micron")
        self.assertGreaterEqual(r.confidence, resolver.FUZZY_THRESHOLD / 100.0)

    def test_layer3_embedding_reached_only_after_1_and_2_fail(self):
        # No lexical overlap with any alias -> layers 1 & 2 fail -> stubbed layer 3.
        called = {"n": 0}
        base = _one_hot_embed_fn("samsung", sim=0.9)

        def spy(texts):
            called["n"] += 1
            return base(texts)

        r = resolve_producer("The Suwon-based conglomerate expanded memory output", CATALOG,
                             embed_fn=spy)
        self.assertEqual(called["n"], 1, "embedding fn must be invoked exactly once")
        self.assertEqual(r.method, "embedding")
        self.assertEqual(r.producer_id, "samsung")

    def test_layer3_embedding_not_called_when_exact_matches(self):
        def boom(texts):
            raise AssertionError("embedding must not be reached when exact matches")
        r = resolve_producer("Samsung Electronics posts strong quarter", CATALOG, embed_fn=boom)
        self.assertEqual(r.method, "exact_alias")

    def test_unresolved_when_all_layers_fail(self):
        r = resolve_producer("Local weather disrupts regional agriculture markets",
                             CATALOG, embed_fn=_low_sim_embed_fn)
        self.assertIsNone(r.producer_id)
        self.assertEqual(r.method, "unresolved")
        self.assertEqual(r.confidence, 0.0)


def _seed_signals(db_path):
    db_init.init_db(db_path)
    conn = db_init.get_connection(db_path)
    for rid, name in [("dram", "DRAM"), ("gpu", "GPUs")]:
        conn.execute("INSERT INTO resource (resource_id, name, aliases) VALUES (?, ?, '[]')",
                     (rid, name))
    for pid, name in [("micron", "Micron"), ("nvidia", "Nvidia"), ("samsung", "Samsung"),
                      ("sk_hynix", "SK Hynix"), ("tsmc", "TSMC")]:
        conn.execute("INSERT INTO producer (producer_id, name) VALUES (?, ?)", (pid, name))

    def ins(stype, rid, pid, raw):
        conn.execute(
            "INSERT INTO signal (signal_type, resource_id, producer_id, source_name, "
            "source_url, raw_text) VALUES (?, ?, ?, 'x', ?, ?)",
            (stype, rid, pid, f"http://u/{stype}/{raw[:8]}", raw))

    # news_raw: a NULL that should resolve, a wrong naive guess, and an unresolvable one.
    ins("news_raw", "gpu", None, "Nvidia ships record GPUs this quarter")
    ins("news_raw", "dram", "samsung", "Micron Technology raises DRAM output")  # wrong naive guess
    ins("news_raw", "dram", None, "Regional weather disrupts farming output")   # unresolvable
    # out-of-scope rows that must never be touched:
    ins("demand_intent_raw", "gpu", None, "8-K item 8.01")
    ins("capacity_expansion", "dram", "micron", "capex fact")
    conn.commit()
    conn.close()


class ReprocessTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "test.db"
        self.log_path = Path(self._tmp.name) / "log.jsonl"
        os.environ["SUPPLY_CHAIN_DB"] = str(self.db_path)
        _seed_signals(self.db_path)

    def tearDown(self):
        os.environ.pop("SUPPLY_CHAIN_DB", None)
        self._tmp.cleanup()

    def _run(self):
        # Stub embed_fn so the one unresolvable row stays unresolved (never downloads).
        return reprocess_signals.reprocess(
            self.db_path, catalog=CATALOG, embed_fn=_low_sim_embed_fn, log_path=self.log_path)

    def test_only_news_raw_rows_modified(self):
        before = self._snapshot_out_of_scope()
        self._run()
        after = self._snapshot_out_of_scope()
        self.assertEqual(before, after, "out-of-scope rows must be untouched")

    def _snapshot_out_of_scope(self):
        conn = db_init.get_connection(self.db_path)
        rows = conn.execute(
            "SELECT signal_id, producer_id FROM signal "
            "WHERE signal_type IN ('demand_intent_raw','capacity_expansion') "
            "ORDER BY signal_id").fetchall()
        conn.close()
        return rows

    def test_news_raw_resolution_outcomes(self):
        self._run()
        conn = db_init.get_connection(self.db_path)
        rows = dict(conn.execute(
            "SELECT raw_text, producer_id FROM signal WHERE signal_type='news_raw'").fetchall())
        conn.close()
        self.assertEqual(rows["Nvidia ships record GPUs this quarter"], "nvidia")
        # wrong naive guess 'samsung' corrected to 'micron' via exact match:
        self.assertEqual(rows["Micron Technology raises DRAM output"], "micron")
        # unresolvable row nulled out (naive guess not trusted):
        self.assertIsNone(rows["Regional weather disrupts farming output"])

    def test_valid_slugs_only(self):
        self._run()
        conn = db_init.get_connection(self.db_path)
        pids = {r[0] for r in conn.execute(
            "SELECT DISTINCT producer_id FROM signal "
            "WHERE signal_type='news_raw' AND producer_id IS NOT NULL")}
        conn.close()
        self.assertTrue(pids.issubset({"micron", "nvidia", "samsung", "sk_hynix", "tsmc"}))

    def test_log_one_line_per_news_raw_row_with_fields(self):
        summary = self._run()
        lines = self.log_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 3)  # exactly 3 news_raw rows seeded
        self.assertEqual(summary["processed"], 3)
        required = {"signal_id", "old_producer_id", "new_producer_id",
                    "method", "confidence", "processed_at"}
        for line in lines:
            rec = json.loads(line)  # valid JSON
            self.assertEqual(set(rec.keys()), required)
            self.assertIn(rec["method"],
                          {"exact_alias", "fuzzy", "embedding", "unresolved"})

    def test_idempotent_db_end_state_second_run(self):
        self._run()
        conn = db_init.get_connection(self.db_path)
        first = conn.execute(
            "SELECT signal_id, producer_id FROM signal WHERE signal_type='news_raw' "
            "ORDER BY signal_id").fetchall()
        conn.close()

        second_summary = self._run()
        conn = db_init.get_connection(self.db_path)
        second = conn.execute(
            "SELECT signal_id, producer_id FROM signal WHERE signal_type='news_raw' "
            "ORDER BY signal_id").fetchall()
        conn.close()

        self.assertEqual(first, second, "DB producer_id state must be identical")
        self.assertEqual(second_summary["changed"], 0, "second run must change nothing")
        # Log still appended fresh lines (timestamped) -> 3 + 3 = 6.
        self.assertEqual(len(self.log_path.read_text().strip().splitlines()), 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
