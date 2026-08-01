"""Offline tests for the Phase-1 historical news backfill:
  * gdelt_client.search_articles  start_date/end_date window path
  * scripts.ingest_historical_news._month_windows  month walker

No network: GdeltDoc.article_search is mocked. Run:
    python -m pytest tests/test_historical_news.py
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion import gdelt_client  # noqa: E402
import scripts.ingest_historical_news as hist  # noqa: E402


class MonthWindowsTest(unittest.TestCase):
    def test_single_month(self):
        self.assertEqual(list(hist._month_windows("2024-05", "2024-05")),
                         [("2024-05-01", "2024-06-01")])

    def test_crosses_year_boundary(self):
        got = list(hist._month_windows("2023-11", "2024-02"))
        self.assertEqual(got, [
            ("2023-11-01", "2023-12-01"),
            ("2023-12-01", "2024-01-01"),
            ("2024-01-01", "2024-02-01"),
            ("2024-02-01", "2024-03-01"),
        ])

    def test_full_2023_2025_is_36_months(self):
        self.assertEqual(len(list(hist._month_windows("2023-01", "2025-12"))), 36)


class GdeltHistoricalWindowTest(unittest.TestCase):
    def test_start_end_dates_passed_to_filters(self):
        captured = {}

        class _FakeDoc:
            def article_search(self, filters):
                # gdeltdoc stores the query string; assert the date window is in it.
                captured["query"] = getattr(filters, "query_string", "") or str(filters.__dict__)
                return pd.DataFrame([{
                    "title": "Micron DRAM fab", "url": "https://x.com/1", "domain": "x.com",
                    "seendate": "20240509T220000Z", "sourcecountry": "US", "language": "English"}])

        with mock.patch.object(gdelt_client, "GdeltDoc", _FakeDoc):
            out = gdelt_client.search_articles(
                ["DRAM"], max_records=50, start_date="2024-05-01", end_date="2024-06-01")
        self.assertEqual(len(out), 1)
        # gdeltdoc renders dates as compact startdatetime/enddatetime params.
        self.assertIn("startdatetime=20240501", captured["query"])
        self.assertIn("enddatetime=20240601", captured["query"])

    def test_timespan_used_when_no_dates(self):
        captured = {}

        class _FakeDoc:
            def article_search(self, filters):
                captured["query"] = getattr(filters, "query_string", "") or str(filters.__dict__)
                return pd.DataFrame()

        with mock.patch.object(gdelt_client, "GdeltDoc", _FakeDoc):
            gdelt_client.search_articles(["DRAM"], max_records=50, timespan="1m")
        # No explicit dates -> no start/end window params, timespan present instead.
        self.assertNotIn("startdatetime", captured["query"])
        self.assertIn("timespan", captured["query"].lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
