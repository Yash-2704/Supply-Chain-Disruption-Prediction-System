"""Required, fully-offline tests for the SEC EDGAR ingestion pipeline.

No real EDGAR calls: the sec_edgar_client's `Company` and `set_identity` are
mocked, and insertion runs against a temp DB seeded with the real schema +
producers + producer_resource. Run: python -m pytest tests/test_ingest_sec.py
"""
import json
import logging
import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from ingestion import sec_edgar_client as sec  # noqa: E402
import scripts.ingest_sec_filings as ingest_sec  # noqa: E402


# ---- fakes mimicking edgartools objects (metadata only) --------------------

class FakeFact:
    def __init__(self, value, fy, fp, period_end, tag, dimensioned=False, unit="USD"):
        self.numeric_value = value
        self.fiscal_year = fy
        self.fiscal_period = fp
        self.period_end = period_end
        self.filing_date = period_end
        self.is_dimensioned = dimensioned
        self.unit = unit


class FakeQuery:
    def __init__(self, facts_by_concept):
        self._by = facts_by_concept
        self._concept = None

    def by_concept(self, tag):
        self._concept = tag
        return self

    def execute(self):
        return list(self._by.get(self._concept, []))


class FakeFacts:
    def __init__(self, facts_by_concept):
        self._by = facts_by_concept

    def query(self):
        return FakeQuery(self._by)


class FakeFiling:
    def __init__(self, accession, description):
        self.form = "8-K"
        self.filing_date = date(2026, 6, 1)
        self.accession_no = accession
        self.url = f"https://sec.gov/{accession}-index.html"
        self.filing_url = self.url
        self.primary_doc_description = description
        self.items = "8.01"


class FakeFilings:
    def __init__(self, filings):
        self._f = filings

    def __len__(self):
        return len(self._f)

    def __getitem__(self, i):
        return self._f[i]


# One short 8-K description reused across tests — deliberately tiny to prove we
# never store a full document body.
SHORT_8K_DESC = "Current report"

MICRON_CAPEX = {"us-gaap:PaymentsToAcquirePropertyPlantAndEquipment":
                [FakeFact(19602000000.0, 2026, "Q3", date(2026, 5, 28),
                          "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment")]}


def _fake_company_factory(capex_map=None, filings=None):
    """Return a fake Company class whose instances serve canned facts/filings."""
    class _FakeCompany:
        def __init__(self, ticker):
            self.ticker = ticker

        def get_facts(self):
            return FakeFacts(capex_map or {})

        def get_filings(self, form=None):
            return FakeFilings(filings or [])
    return _FakeCompany


def _seed_db(db_path):
    db_init.init_db(db_path)
    conn = db_init.get_connection(db_path)
    resources = [("dram", "DRAM"), ("hbm", "HBM"), ("nand", "NAND"), ("gpu", "GPUs")]
    for rid, name in resources:
        conn.execute("INSERT INTO resource (resource_id, name, aliases) VALUES (?, ?, '[]')",
                     (rid, name))
    producers = [("micron", "Micron"), ("nvidia", "Nvidia"), ("samsung", "Samsung"),
                 ("sk_hynix", "SK Hynix"), ("tsmc", "TSMC")]
    for pid, name in producers:
        conn.execute("INSERT INTO producer (producer_id, name) VALUES (?, ?)", (pid, name))
    pr = [("micron", "dram"), ("micron", "hbm"), ("micron", "nand"),
          ("nvidia", "gpu"), ("tsmc", "gpu"),
          ("samsung", "dram"), ("samsung", "hbm"), ("samsung", "nand"),
          ("sk_hynix", "dram"), ("sk_hynix", "hbm"), ("sk_hynix", "nand")]
    for pid, rid in pr:
        conn.execute("INSERT INTO producer_resource (producer_id, resource_id) VALUES (?, ?)",
                     (pid, rid))
    conn.commit()
    conn.close()


class ClientTest(unittest.TestCase):
    def setUp(self):
        sec._identity_set = False  # reset identity memoization between tests

    def test_capex_picks_latest_non_dimensioned(self):
        facts = {"us-gaap:PaymentsToAcquirePropertyPlantAndEquipment": [
            FakeFact(100.0, 2024, "FY", date(2024, 1, 1), "t"),
            FakeFact(999.0, 2020, "Q1", date(2020, 1, 1), "t", dimensioned=True),  # ignored
            FakeFact(200.0, 2026, "Q3", date(2026, 5, 28), "t"),  # newest
        ]}
        with mock.patch.object(sec, "Company", _fake_company_factory(capex_map=facts)), \
             mock.patch.object(sec, "set_identity"):
            out = sec.get_latest_capex("MU")
        self.assertEqual(out["value"], 200.0)
        self.assertEqual(out["fiscal_year"], 2026)

    def test_capex_none_when_no_concept(self):
        with mock.patch.object(sec, "Company", _fake_company_factory(capex_map={})), \
             mock.patch.object(sec, "set_identity"):
            self.assertIsNone(sec.get_latest_capex("TSM"))

    def test_recent_filings_metadata_only(self):
        filings = [FakeFiling("0001-26-1", SHORT_8K_DESC)]
        with mock.patch.object(sec, "Company", _fake_company_factory(filings=filings)), \
             mock.patch.object(sec, "set_identity"):
            out = sec.get_recent_filings("MSFT", "8-K", 20)
        self.assertEqual(len(out), 1)
        self.assertEqual(set(out[0].keys()),
                         {"form_type", "filing_date", "accession_no", "url", "description", "items"})
        self.assertEqual(out[0]["accession_no"], "0001-26-1")

    def test_missing_identity_env_logs_warning_no_crash(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(sec, "set_identity") as mock_set:
            with self.assertLogs("ingestion.sec_edgar_client", level="WARNING") as cm:
                sec.ensure_identity()
            self.assertTrue(any("SEC_EDGAR_USER_AGENT_EMAIL" in m for m in cm.output))
            # identity still set (with the placeholder) rather than crashing
            mock_set.assert_called_once()


class OrchestrationTest(unittest.TestCase):
    def setUp(self):
        sec._identity_set = False
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "test.db"
        os.environ["SUPPLY_CHAIN_DB"] = str(self.db_path)
        _seed_db(self.db_path)

    def tearDown(self):
        os.environ.pop("SUPPLY_CHAIN_DB", None)
        self._tmp.cleanup()

    def _run(self):
        """Run the real orchestrator with capex only for MU and 8-Ks for everyone."""
        micron_filings = [FakeFiling("0001-26-MU", SHORT_8K_DESC)]
        msft_filings = [FakeFiling("0009-26-MSFT", SHORT_8K_DESC)]

        def company(ticker):
            capex = MICRON_CAPEX if ticker == "MU" else {}
            if ticker == "MU":
                filings = micron_filings
            elif ticker == "MSFT":
                filings = msft_filings
            else:
                filings = []  # NVDA/TSM/foreign/other hyperscalers -> no filings here
            return _fake_company_factory(capex_map=capex, filings=filings)(ticker)

        with mock.patch.object(sec, "Company", side_effect=company), \
             mock.patch.object(sec, "set_identity"), \
             mock.patch.object(sec, "_throttle"), \
             mock.patch.dict(os.environ, {"SEC_EDGAR_USER_AGENT_EMAIL": "real@example.com"}):
            ingest_sec.run(self.db_path)

    def _conn(self):
        return db_init.get_connection(self.db_path)

    def test_capacity_row_producer_and_resource_attribution(self):
        self._run()
        conn = self._conn()
        rows = conn.execute(
            "SELECT resource_id, producer_id, value, unit FROM signal "
            "WHERE signal_type='capacity_expansion' ORDER BY resource_id").fetchall()
        conn.close()
        # Micron makes dram, hbm, nand -> exactly those three, producer_id='micron'.
        self.assertEqual({r[0] for r in rows}, {"dram", "hbm", "nand"})
        for rid, pid, value, unit in rows:
            self.assertEqual(pid, "micron")
            self.assertEqual(value, 19602000000.0)
            self.assertEqual(unit, "USD")

    def test_demand_filer_row_null_producer_and_mapped_resources(self):
        self._run()
        conn = self._conn()
        rows = conn.execute(
            "SELECT resource_id, producer_id FROM signal "
            "WHERE signal_type='demand_intent_raw' AND source_name='Microsoft'").fetchall()
        conn.close()
        self.assertTrue(rows)
        for rid, pid in rows:
            self.assertIsNone(pid)  # hard rule: hyperscaler producer_id is NULL
        # Microsoft mapping is gpu,hbm,dram,nand — must match the file, not be derived.
        expected = set(json.loads(
            (PROJECT_ROOT / "data" / "demand_resource_mapping.json").read_text()
        )["mappings"][0]["resource_ids"])
        self.assertEqual({r[0] for r in rows}, expected)

    def test_producer_8k_has_populated_producer_id(self):
        self._run()
        conn = self._conn()
        rows = conn.execute(
            "SELECT DISTINCT producer_id FROM signal "
            "WHERE signal_type='demand_intent_raw' AND source_name='Micron'").fetchall()
        conn.close()
        self.assertEqual([r[0] for r in rows], ["micron"])

    def test_idempotent_second_run(self):
        self._run()
        conn = self._conn()
        first = conn.execute("SELECT COUNT(*) FROM signal").fetchone()[0]
        conn.close()
        self.assertGreater(first, 0)
        self._run()
        conn = self._conn()
        second = conn.execute("SELECT COUNT(*) FROM signal").fetchone()[0]
        conn.close()
        self.assertEqual(first, second)

    def test_severity_and_confidence_always_null(self):
        self._run()
        conn = self._conn()
        bad = conn.execute(
            "SELECT COUNT(*) FROM signal WHERE severity IS NOT NULL OR confidence IS NOT NULL"
        ).fetchone()[0]
        conn.close()
        self.assertEqual(bad, 0)

    def test_raw_text_is_short_metadata_not_full_body(self):
        self._run()
        conn = self._conn()
        raws = [r[0] for r in conn.execute(
            "SELECT raw_text FROM signal WHERE signal_type='demand_intent_raw'").fetchall()]
        conn.close()
        self.assertTrue(raws)
        for txt in raws:
            # Metadata-only rows are tiny; a stored document body would be huge.
            self.assertLess(len(txt), 120, f"raw_text looks like a body: {txt!r}")
            self.assertIn(SHORT_8K_DESC, txt)


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    unittest.main(verbosity=2)
