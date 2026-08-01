"""Required, fully-offline tests for the LLM extraction stage.

No real API calls: both providers' call functions are monkeypatched. Run:
    python -m pytest tests/test_llm_extraction.py
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from llm_extraction import llm_client, validator  # noqa: E402
from llm_extraction.validator import validate_extraction  # noqa: E402
import scripts.run_extraction as run_extraction  # noqa: E402


def _valid_json(severity=5, confidence=0.8, event_type="supply_disruption"):
    return json.dumps({
        "event_type": event_type,
        "severity": severity,
        "confidence": confidence,
        "entities": ["Samsung", "Micron"],
        "reasoning": "Memory makers face a supply lawsuit affecting DRAM.",
    })


# ---- validator (direct, one pass + one fail per field) ---------------------

class ValidatorTest(unittest.TestCase):
    def _base(self, **over):
        d = {"event_type": "other", "severity": 5, "confidence": 0.5,
             "entities": [], "reasoning": "ok"}
        d.update(over)
        return d

    def test_pass(self):
        self.assertTrue(validate_extraction(self._base()))

    def test_event_type(self):
        self.assertTrue(validate_extraction(self._base(event_type="price_movement")))
        self.assertFalse(validate_extraction(self._base(event_type="not_a_type")))

    def test_severity(self):
        self.assertTrue(validate_extraction(self._base(severity=10)))
        self.assertFalse(validate_extraction(self._base(severity=15)))   # out of range
        self.assertFalse(validate_extraction(self._base(severity=0)))
        self.assertFalse(validate_extraction(self._base(severity=5.0)))  # not int
        self.assertFalse(validate_extraction(self._base(severity=True)))  # bool excluded

    def test_confidence(self):
        self.assertTrue(validate_extraction(self._base(confidence=0.0)))
        self.assertTrue(validate_extraction(self._base(confidence=1.0)))
        self.assertFalse(validate_extraction(self._base(confidence=1.5)))
        self.assertFalse(validate_extraction(self._base(confidence="high")))

    def test_entities(self):
        self.assertTrue(validate_extraction(self._base(entities=["A"])))
        self.assertFalse(validate_extraction(self._base(entities="A")))

    def test_reasoning(self):
        self.assertTrue(validate_extraction(self._base(reasoning="because")))
        self.assertFalse(validate_extraction(self._base(reasoning="")))
        self.assertFalse(validate_extraction(self._base(reasoning=None)))

    def test_not_a_dict(self):
        self.assertFalse(validate_extraction("nope"))


# ---- extract_structured: multi-provider fallback + parsing -----------------

# First and second providers in the ranked fallback order (Gemini removed).
_FIRST = llm_client.PROVIDER_ORDER[0]
_SECOND = llm_client.PROVIDER_ORDER[1]


def _provider_side_effect(mapping, default=None):
    """Build a call_provider side_effect returning `mapping[name]` (else default)."""
    def fn(name, prompt):
        return mapping.get(name, default)
    return fn


class ExtractStructuredTest(unittest.TestCase):
    def test_first_provider_success_short_circuits(self):
        with mock.patch.object(llm_client, "call_provider",
                               side_effect=_provider_side_effect({_FIRST: _valid_json()})) as cp:
            out = llm_client.extract_structured("p")
        self.assertEqual(out["_provider"], _FIRST)
        self.assertEqual(out["severity"], 5)
        cp.assert_called_once()  # later providers not called once one succeeds
        self.assertEqual(cp.call_args[0][0], _FIRST)

    def test_fallback_to_next_provider(self):
        # First provider yields nothing; second returns valid output.
        with mock.patch.object(llm_client, "call_provider",
                               side_effect=_provider_side_effect({_SECOND: _valid_json(severity=7)})):
            out = llm_client.extract_structured("p")
        self.assertEqual(out["_provider"], _SECOND)
        self.assertEqual(out["severity"], 7)

    def test_validation_failure_returns_none(self):
        # Every provider returns out-of-range severity -> overall None.
        with mock.patch.object(llm_client, "call_provider",
                               return_value=_valid_json(severity=15)):
            self.assertIsNone(llm_client.extract_structured("p"))

    def test_code_fenced_response_is_stripped_and_parsed(self):
        fenced = "```json\n" + _valid_json(severity=6) + "\n```"
        with mock.patch.object(llm_client, "call_provider",
                               side_effect=_provider_side_effect({_FIRST: fenced})) as cp:
            out = llm_client.extract_structured("p")
        self.assertEqual(out["severity"], 6)          # fences stripped, parsed
        self.assertEqual(out["_provider"], _FIRST)
        cp.assert_called_once()

    def test_leading_prose_is_stripped(self):
        messy = "Here is the JSON you requested:\n" + _valid_json(severity=4)
        with mock.patch.object(llm_client, "call_provider",
                               side_effect=_provider_side_effect({_FIRST: messy})):
            out = llm_client.extract_structured("p")
        self.assertEqual(out["severity"], 4)


# ---- orchestration (temp DB) ----------------------------------------------

def _seed_db(db_path):
    db_init.init_db(db_path)
    conn = db_init.get_connection(db_path)
    conn.execute("INSERT INTO resource (resource_id, name, aliases) VALUES ('dram','DRAM','[]')")
    conn.execute("INSERT INTO resource (resource_id, name, aliases) VALUES ('gpu','GPUs','[]')")
    conn.execute("INSERT INTO producer (producer_id, name) VALUES ('micron','Micron')")
    conn.executescript(
        (PROJECT_ROOT / "schema_clustering.sql").read_text(encoding="utf-8"))

    def ins(sid, stype, raw):
        conn.execute(
            "INSERT INTO signal (signal_id, signal_type, resource_id, raw_text, source_name) "
            "VALUES (?, ?, 'dram', ?, 'x')", (sid, stype, raw))

    ins(1, "news_raw", "Samsung and Micron sued over DRAM supply")   # canonical
    ins(2, "news_raw", "Duplicate coverage of the DRAM lawsuit")     # duplicate!
    ins(3, "news_raw", "CXMT secures DRAM supply deal")              # canonical
    ins(4, "demand_intent_raw", "8-K item 2.02 earnings")            # demand
    ins(5, "capacity_expansion", "capex 19.6B")                      # must be ignored
    # cluster membership: 1 canonical, 2 duplicate (of 1), 3 canonical singleton.
    for sid, cid, canon in [("1", "c_1", 1), ("2", "c_1", 0), ("3", "c_3", 1)]:
        conn.execute("INSERT INTO news_cluster_membership "
                     "(signal_id, cluster_id, is_canonical, processed_at) VALUES (?,?,?,?)",
                     (sid, cid, canon, "t"))
    conn.commit()
    conn.close()


class OrchestrationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "test.db"
        os.environ["SUPPLY_CHAIN_DB"] = str(self.db_path)
        # A configured provider key so the run() guard passes; calls are mocked.
        os.environ[llm_client.PROVIDERS[_FIRST]["env"]] = "test-key"
        _seed_db(self.db_path)

    def tearDown(self):
        os.environ.pop("SUPPLY_CHAIN_DB", None)
        os.environ.pop(llm_client.PROVIDERS[_FIRST]["env"], None)
        self._tmp.cleanup()

    def test_valid_first_provider_sets_severity_and_audit_row(self):
        with mock.patch.object(llm_client, "call_provider",
                               side_effect=_provider_side_effect({_FIRST: _valid_json(severity=8)})):
            run_extraction.run(self.db_path)
        conn = db_init.get_connection(self.db_path)
        sev = conn.execute("SELECT severity, confidence FROM signal WHERE signal_id=1").fetchone()
        prov = conn.execute("SELECT llm_provider FROM event_extraction WHERE signal_id=1").fetchone()
        conn.close()
        self.assertEqual(sev[0], 8)
        self.assertTrue(0.0 <= sev[1] <= 1.0)
        self.assertEqual(prov[0], _FIRST)

    def test_fallback_provider_recorded(self):
        with mock.patch.object(llm_client, "call_provider",
                               side_effect=_provider_side_effect({_SECOND: _valid_json(severity=6)})):
            run_extraction.run(self.db_path)
        conn = db_init.get_connection(self.db_path)
        prov = conn.execute("SELECT llm_provider FROM event_extraction WHERE signal_id=1").fetchone()
        conn.close()
        self.assertEqual(prov[0], _SECOND)

    def test_duplicate_and_capacity_never_selected(self):
        with mock.patch.object(llm_client, "call_provider",
                               side_effect=_provider_side_effect({_FIRST: _valid_json()})):
            run_extraction.run(self.db_path)
        conn = db_init.get_connection(self.db_path)
        # signal 2 is a duplicate (is_canonical=0), signal 5 is capacity_expansion.
        for sid in (2, 5):
            sev = conn.execute("SELECT severity FROM signal WHERE signal_id=?", (sid,)).fetchone()[0]
            ee = conn.execute("SELECT COUNT(*) FROM event_extraction WHERE signal_id=?", (sid,)).fetchone()[0]
            self.assertIsNone(sev, f"signal {sid} must not be scored")
            self.assertEqual(ee, 0, f"signal {sid} must have no extraction row")
        conn.close()

    def test_middle_failure_does_not_stop_batch(self):
        # 3 eligible rows (news 1, news 3, demand 4). Make the CXMT row (signal 3)
        # fail ALL providers; 1 and 4 must still succeed.
        def per_call(name, prompt):
            return None if "CXMT" in prompt else _valid_json(severity=5)
        with mock.patch.object(llm_client, "call_provider", side_effect=per_call):
            run_extraction.run(self.db_path)
        conn = db_init.get_connection(self.db_path)
        scored = {r[0] for r in conn.execute(
            "SELECT signal_id FROM signal WHERE severity IS NOT NULL")}
        conn.close()
        self.assertIn(1, scored)
        self.assertIn(4, scored)
        self.assertNotIn(3, scored)  # failed all providers, left NULL

    def test_idempotent_no_recall_for_scored_rows(self):
        cp = mock.Mock(side_effect=_provider_side_effect({_FIRST: _valid_json(severity=5)}))
        with mock.patch.object(llm_client, "call_provider", cp):
            run_extraction.run(self.db_path)
            first_calls = cp.call_count
            run_extraction.run(self.db_path)  # second run: all scored already
            second_calls = cp.call_count
        self.assertGreater(first_calls, 0)
        self.assertEqual(first_calls, second_calls, "no LLM calls for already-scored rows")

    def test_no_extraction_row_without_severity(self):
        with mock.patch.object(llm_client, "call_provider",
                               side_effect=_provider_side_effect({_FIRST: _valid_json()})):
            run_extraction.run(self.db_path)
        conn = db_init.get_connection(self.db_path)
        orphans = conn.execute(
            "SELECT COUNT(*) FROM signal s JOIN event_extraction e ON e.signal_id=s.signal_id "
            "WHERE s.severity IS NULL").fetchone()[0]
        conn.close()
        self.assertEqual(orphans, 0)


class NoProviderTest(unittest.TestCase):
    def test_no_keys_returns_nonzero_no_crash(self):
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "t.db"
            _seed_db(db_path)
            with mock.patch.dict(os.environ, {}, clear=True):
                os.environ["SUPPLY_CHAIN_DB"] = str(db_path)
                rc = run_extraction.run(db_path)
        self.assertEqual(rc, 1)  # clean non-zero exit, no exception


if __name__ == "__main__":
    unittest.main(verbosity=2)
