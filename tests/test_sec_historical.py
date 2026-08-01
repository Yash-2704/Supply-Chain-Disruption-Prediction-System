"""Offline tests for the Phase-1 historical SEC additions:
  * sec_edgar_client.get_capex_history     (all facts, date-windowed, deduped)
  * sec_edgar_client.get_filings_in_range  (filing_date range passthrough)

No EDGAR network: Company/set_identity mocked. Run:
    python -m pytest tests/test_sec_historical.py
"""
import sys
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion import sec_edgar_client as sec  # noqa: E402

CAPEX_TAG = "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment"


# ---- minimal edgartools fakes (metadata only) ------------------------------

class FakeFact:
    def __init__(self, value, fy, fp, period_end, tag, dimensioned=False, unit="USD"):
        self.numeric_value = value
        self.fiscal_year = fy
        self.fiscal_period = fp
        self.period_end = period_end
        self.filing_date = period_end
        self.is_dimensioned = dimensioned
        self.unit = unit


class _FakeQuery:
    def __init__(self, by):
        self._by = by
        self._concept = None

    def by_concept(self, tag):
        self._concept = tag
        return self

    def execute(self):
        return list(self._by.get(self._concept, []))


class FakeFacts:
    def __init__(self, by):
        self._by = by

    def query(self):
        return _FakeQuery(self._by)


class FakeFilings:
    def __init__(self, filings):
        self._f = filings

    def __len__(self):
        return len(self._f)

    def __getitem__(self, i):
        return self._f[i]


class _FakeCompany:
    def __init__(self, facts=None, filings=None, captured=None):
        self._facts = facts
        self._filings = filings
        self._captured = captured if captured is not None else {}

    def get_facts(self):
        return self._facts

    def get_filings(self, form=None, filing_date=None):
        self._captured["form"] = form
        self._captured["filing_date"] = filing_date
        return self._filings


class GetCapexHistoryTest(unittest.TestCase):
    def _facts(self):
        return FakeFacts({CAPEX_TAG: [
            FakeFact(1e9, 2023, "Q1", date(2023, 3, 31), CAPEX_TAG),
            FakeFact(2e9, 2024, "Q1", date(2024, 3, 31), CAPEX_TAG),
            FakeFact(3e9, 2025, "Q1", date(2025, 3, 31), CAPEX_TAG),
            FakeFact(9e9, 2026, "Q1", date(2026, 3, 31), CAPEX_TAG),  # outside window
        ]})

    def test_returns_all_facts_sorted(self):
        with mock.patch.object(sec, "Company", return_value=_FakeCompany(facts=self._facts())), \
             mock.patch.object(sec, "set_identity"):
            sec._identity_set = False
            out = sec.get_capex_history("MU")
        self.assertEqual([r["period_end"] for r in out],
                         ["2023-03-31", "2024-03-31", "2025-03-31", "2026-03-31"])

    def test_date_window_filters(self):
        with mock.patch.object(sec, "Company", return_value=_FakeCompany(facts=self._facts())), \
             mock.patch.object(sec, "set_identity"):
            sec._identity_set = False
            out = sec.get_capex_history("MU", start_date="2023-01-01", end_date="2025-12-31")
        self.assertEqual(len(out), 3)              # 2026 excluded
        self.assertTrue(all("2023" <= r["period_end"][:4] <= "2025" for r in out))

    def test_dimensioned_and_nonusd_skipped(self):
        facts = FakeFacts({CAPEX_TAG: [
            FakeFact(1e9, 2024, "Q1", date(2024, 3, 31), CAPEX_TAG, dimensioned=True),
            FakeFact(2e9, 2024, "Q2", date(2024, 6, 30), CAPEX_TAG, unit="EUR"),
            FakeFact(3e9, 2024, "Q3", date(2024, 9, 30), CAPEX_TAG),  # only this one qualifies
        ]})
        with mock.patch.object(sec, "Company", return_value=_FakeCompany(facts=facts)), \
             mock.patch.object(sec, "set_identity"):
            sec._identity_set = False
            out = sec.get_capex_history("MU")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["period_end"], "2024-09-30")

    def test_no_facts_returns_empty(self):
        with mock.patch.object(sec, "Company", return_value=_FakeCompany(facts=None)), \
             mock.patch.object(sec, "set_identity"):
            sec._identity_set = False
            out = sec.get_capex_history("MU")
        self.assertEqual(out, [])


class _RangeFiling:
    def __init__(self, d):
        self.form = "8-K"
        self.filing_date = d
        self.accession_no = f"acc-{d.isoformat()}"
        self.url = f"https://sec.gov/{d.isoformat()}-index.html"
        self.filing_url = self.url
        self.primary_doc_description = "Results of Operations"
        self.items = "2.02"


class GetFilingsInRangeTest(unittest.TestCase):
    def test_passes_range_and_normalizes(self):
        captured = {}
        filings = FakeFilings([_RangeFiling(date(2024, 5, 1)), _RangeFiling(date(2024, 8, 1))])
        with mock.patch.object(sec, "Company",
                               return_value=_FakeCompany(filings=filings, captured=captured)), \
             mock.patch.object(sec, "set_identity"):
            sec._identity_set = False
            out = sec.get_filings_in_range("MU", "8-K", "2023-01-01", "2025-12-31")
        self.assertEqual(captured["filing_date"], "2023-01-01:2025-12-31")
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["filing_date"], "2024-05-01")
        self.assertEqual(set(out[0]), {"form_type", "filing_date", "accession_no",
                                       "url", "description", "items"})

    def test_open_ended_range(self):
        captured = {}
        with mock.patch.object(sec, "Company",
                               return_value=_FakeCompany(filings=FakeFilings([]), captured=captured)), \
             mock.patch.object(sec, "set_identity"):
            sec._identity_set = False
            sec.get_filings_in_range("MU", "8-K", start_date="2023-01-01")
        self.assertEqual(captured["filing_date"], "2023-01-01:")

    def test_error_returns_empty(self):
        class _Boom:
            def get_filings(self, **k):
                raise RuntimeError("edgar down")
        with mock.patch.object(sec, "Company", return_value=_Boom()), \
             mock.patch.object(sec, "set_identity"):
            sec._identity_set = False
            with self.assertLogs("ingestion.sec_edgar_client", level="WARNING"):
                out = sec.get_filings_in_range("MU", "8-K", "2023-01-01", "2025-12-31")
        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
