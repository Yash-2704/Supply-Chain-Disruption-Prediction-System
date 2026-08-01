"""Required, fully-offline tests for the RAG explanation stage.

Ephemeral Chroma, stubbed embeddings, mocked LLM calls — no network, no model
download, no persistent store. Run:  python -m pytest tests/test_rag_explanation.py
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from clustering import vector_store  # noqa: E402
from rag import retriever, narrator  # noqa: E402
from rag.narrator import (  # noqa: E402
    apply_disclaimer, validate_narration, SYNTHETIC_DISCLAIMER, MAX_EXCERPT_WORDS,
)
import scripts.generate_explanation as ge  # noqa: E402


# Stub embeddings: map text -> vector. "dram query" aligns with dram evidence.
_VECS = {
    "q_dram": [1.0, 0.0, 0.0],
    "ev_on": [0.98, 0.02, 0.0],    # ~0.999 cosine with q_dram -> above threshold
    "ev_off": [0.0, 1.0, 0.0],     # orthogonal -> below threshold
}


def _stub_embed_query(texts):
    # Retriever only embeds the query; return the dram-query vector.
    return np.array([_VECS["q_dram"]], dtype=float)


def _unique_coll():
    import uuid
    client = vector_store.get_client(ephemeral=True)
    return vector_store.get_collection(client, name=f"t_{uuid.uuid4().hex}")


def _seed_db(path):
    db_init.init_db(path)
    conn = db_init.get_connection(path)
    for rid, name in [("dram", "DRAM"), ("nand", "NAND")]:
        conn.execute("INSERT INTO resource (resource_id, name, aliases) VALUES (?, ?, '[]')",
                     (rid, name))
    conn.executescript((PROJECT_ROOT / "schema_clustering.sql").read_text())
    conn.executescript((PROJECT_ROOT / "schema_model.sql").read_text())

    def sig(sid, rid, raw):
        conn.execute("INSERT INTO signal (signal_id, signal_type, resource_id, raw_text, "
                     "source_url, source_name) VALUES (?, 'news_raw', ?, ?, ?, 'x')",
                     (sid, rid, raw, f"http://u/{sid}"))
    sig(1, "dram", "DRAM prices climb on AI memory demand")   # canonical, on-topic
    sig(2, "dram", "Duplicate coverage of the DRAM price move")  # DUPLICATE
    sig(3, "dram", "Unrelated bakery festival news")           # canonical but off-topic
    for s, canon in [("1", 1), ("2", 0), ("3", 1)]:
        conn.execute("INSERT INTO news_cluster_membership (signal_id, cluster_id, "
                     "is_canonical, processed_at) VALUES (?, ?, ?, 't')",
                     (s, f"c_{s}", canon))
    conn.commit()
    conn.close()


def _seed_chroma(coll):
    # signal 1 (canonical, on-topic), 2 (duplicate, on-topic), 3 (canonical, off-topic)
    coll.upsert(ids=["1"], embeddings=[_VECS["ev_on"]], metadatas=[{"resource_id": "dram"}])
    coll.upsert(ids=["2"], embeddings=[_VECS["ev_on"]], metadatas=[{"resource_id": "dram"}])
    coll.upsert(ids=["3"], embeddings=[_VECS["ev_off"]], metadatas=[{"resource_id": "dram"}])


class RetrieverTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "t.db"
        _seed_db(self.db_path)
        self.conn = db_init.get_connection(self.db_path)
        self.coll = _unique_coll()
        _seed_chroma(self.coll)

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()

    def test_excludes_noncanonical(self):
        ev = retriever.retrieve_canonical_evidence(
            "dram", self.conn, self.coll, embed_fn=_stub_embed_query)
        ids = {e["signal_id"] for e in ev}
        self.assertIn(1, ids)         # canonical on-topic kept
        self.assertNotIn(2, ids)      # duplicate excluded despite high similarity
        self.assertNotIn(3, ids)      # canonical but below similarity threshold

    def test_empty_when_nothing_qualifies(self):
        # Raise the threshold above everything -> empty list, not an error.
        ev = retriever.retrieve_canonical_evidence(
            "dram", self.conn, self.coll, min_similarity=0.999999,
            embed_fn=_stub_embed_query)
        self.assertEqual(ev, [])


class ValidateNarrationTest(unittest.TestCase):
    def _evidence(self):
        return [{"signal_id": 1, "raw_text": "DRAM prices climb on AI memory demand"},
                {"signal_id": 5, "raw_text": "Micron NAND update"}]

    def test_strips_fabricated_citation_keeps_valid(self):
        parsed = {"narrative": "DRAM is tightening.",
                  "citations": [{"signal_id": 1, "excerpt": "DRAM prices climb"},
                                {"signal_id": 999, "excerpt": "invented source"}]}
        out = validate_narration(parsed, self._evidence())
        ids = {c["signal_id"] for c in out["citations"]}
        self.assertEqual(ids, {1})  # 999 (not retrieved) stripped, 1 kept

    def test_all_invalid_citations_returns_none(self):
        parsed = {"narrative": "Looks fine but unfounded.",
                  "citations": [{"signal_id": 888, "excerpt": "nope"},
                                {"signal_id": 999, "excerpt": "also nope"}]}
        self.assertIsNone(validate_narration(parsed, self._evidence()))

    def test_excerpt_word_cap_enforced(self):
        long_excerpt = " ".join(["word"] * (MAX_EXCERPT_WORDS + 5))
        parsed = {"narrative": "DRAM note.",
                  "citations": [{"signal_id": 1, "excerpt": long_excerpt}]}
        # The only citation is over-cap -> stripped -> zero valid -> None.
        self.assertIsNone(validate_narration(parsed, self._evidence()))

    def test_malformed_returns_none(self):
        self.assertIsNone(validate_narration({"citations": []}, self._evidence()))
        self.assertIsNone(validate_narration("nope", self._evidence()))


class ApplyDisclaimerTest(unittest.TestCase):
    def test_disclaimer_added_when_not_real(self):
        out = apply_disclaimer("Body text.", is_real_data_model=False)
        self.assertIn(SYNTHETIC_DISCLAIMER, out)
        self.assertTrue(out.startswith(SYNTHETIC_DISCLAIMER))

    def test_no_disclaimer_when_real(self):
        out = apply_disclaimer("Body text.", is_real_data_model=True)
        self.assertNotIn(SYNTHETIC_DISCLAIMER, out)
        self.assertEqual(out, "Body text.")


_VALID_LLM_JSON = json.dumps({
    "narrative": "DRAM supply is tightening amid AI-driven memory demand.",
    "citations": [{"signal_id": 1, "excerpt": "DRAM prices climb on AI memory demand"}],
})


class OrchestrationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "t.db"
        os.environ["SUPPLY_CHAIN_DB"] = str(self.db_path)
        _seed_db(self.db_path)
        self.coll = _unique_coll()
        _seed_chroma(self.coll)

    def tearDown(self):
        os.environ.pop("SUPPLY_CHAIN_DB", None)
        self._tmp.cleanup()

    def _explain(self, resource_id, prediction_id=None):
        return ge.explain_resource(resource_id, prediction_id, db_path=self.db_path,
                                   collection=self.coll, embed_fn=_stub_embed_query)

    def _seed_prediction(self, data_mode):
        conn = db_init.get_connection(self.db_path)
        cur = conn.execute(
            "INSERT INTO training_run (run_at, data_mode, n_examples, random_seed, "
            "model_path, calibration_applied) VALUES ('t', ?, 300, 42, 'm.pkl', 1)",
            (data_mode,))
        run_id = cur.lastrowid
        cur = conn.execute(
            "INSERT INTO prediction (run_id, resource_id, week_start, horizon, raw_score, "
            "calibrated_probability, predicted_at) VALUES (?, 'dram', 'w', '1m', 0.3, 0.44, 't')",
            (run_id,))
        pid = cur.lastrowid
        conn.commit()
        conn.close()
        return pid

    def _last_row(self):
        conn = db_init.get_connection(self.db_path)
        row = conn.execute(
            "SELECT status, narrative, citations, is_real_data_model, llm_provider, "
            "evidence_count FROM explanation ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        return row

    def test_zero_evidence_writes_insufficient_and_skips_llm(self):
        with mock.patch.object(ge.llm_client, "generate_text") as gt:
            out = self._explain("nand")  # nand has no signals/embeddings at all
        self.assertEqual(out["status"], "insufficient_evidence")
        gt.assert_not_called()  # no LLM call when there is no evidence
        self.assertEqual(self._last_row()[0], "insufficient_evidence")

    def test_evidence_only_mode_disclaimer_present(self):
        with mock.patch.object(ge.llm_client, "generate_text",
                               return_value=(_VALID_LLM_JSON, "cerebras")):
            out = self._explain("dram", prediction_id=None)
        self.assertEqual(out["status"], "ok")
        status, narrative, citations, is_real, provider, ev = self._last_row()
        self.assertEqual(is_real, 0)
        self.assertIn(SYNTHETIC_DISCLAIMER, narrative)  # direct string check
        self.assertEqual(json.loads(citations)[0]["signal_id"], 1)

    def test_synthetic_prediction_still_gets_disclaimer(self):
        pid = self._seed_prediction("synthetic_validation")
        with mock.patch.object(ge.llm_client, "generate_text",
                               return_value=(_VALID_LLM_JSON, "cerebras")):
            self._explain("dram", prediction_id=pid)
        status, narrative, citations, is_real, provider, ev = self._last_row()
        self.assertEqual(status, "ok")
        self.assertEqual(is_real, 0)  # prediction supplied, but it's synthetic
        self.assertIn(SYNTHETIC_DISCLAIMER, narrative)

    def test_fallback_provider_recorded(self):
        # generate_text reports which provider served the narration (e.g. groq).
        with mock.patch.object(ge.llm_client, "generate_text",
                               return_value=(_VALID_LLM_JSON, "groq")):
            out = self._explain("dram")
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["provider"], "groq")
        self.assertEqual(self._last_row()[4], "groq")

    def test_all_invalid_citations_yields_invalid_status_no_ok(self):
        bad = json.dumps({"narrative": "Fabricated.",
                          "citations": [{"signal_id": 777, "excerpt": "ghost"}]})
        with mock.patch.object(ge.llm_client, "generate_text", return_value=(bad, "cerebras")):
            out = self._explain("dram")
        self.assertEqual(out["status"], "invalid_citations")
        self.assertEqual(self._last_row()[0], "invalid_citations")
        self.assertIsNone(self._last_row()[1])  # no narrative persisted


if __name__ == "__main__":
    unittest.main(verbosity=2)
