"""Required, fully-offline tests for the news ingestion pipeline (unittest).

No real network calls: GDELT's article_search and requests.get are both mocked.
Insertion tests run against a temp SQLite DB seeded with the real schema + the
4 resources. Run:  python -m pytest tests/test_ingest_news.py   (or unittest).
"""
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from ingestion import gdelt_client, newsapi_client, tagger  # noqa: E402
import scripts.ingest_news as ingest_news  # noqa: E402

# ---- realistic mocked API payloads -----------------------------------------

MOCK_GDELT_DF = pd.DataFrame([
    {"title": "DRAM prices climb amid tight supply", "url": "https://ex.com/dram-1",
     "domain": "ex.com", "seendate": "20240115T120000Z",
     "sourcecountry": "United States", "language": "English"},
    {"title": "Nvidia HBM and GPU demand surges for AI", "url": "https://ex.com/hbm-gpu-1",
     "domain": "ex.com", "seendate": "20240116T090000Z",
     "sourcecountry": "United States", "language": "English"},
    {"title": "Local bakery wins county fair ribbon", "url": "https://ex.com/noise-1",
     "domain": "ex.com", "seendate": "20240116T100000Z",
     "sourcecountry": "United States", "language": "English"},
])

MOCK_NEWSAPI_JSON = {
    "status": "ok",
    "totalResults": 1,
    "articles": [
        {"source": {"name": "TechWire"}, "title": "NAND flash oversupply hits Micron",
         "description": "NAND prices fall as inventories build.",
         "url": "https://news.com/nand-1", "publishedAt": "2024-01-17T08:00:00Z"},
    ],
}


def _small_resource_catalog():
    return [
        ("dram", ["DRAM", "Dynamic RAM"]),
        ("hbm", ["HBM", "High Bandwidth Memory"]),
        ("nand", ["NAND", "NAND flash"]),
        ("gpu", ["GPUs", "GPU"]),
    ]


def _small_producer_catalog():
    return [("nvidia", ["Nvidia", "NVIDIA"]), ("micron", ["Micron"])]


class GdeltClientTest(unittest.TestCase):
    @mock.patch("ingestion.gdelt_client.GdeltDoc")
    def test_normalizes_response(self, mock_gdelt):
        mock_gdelt.return_value.article_search.return_value = MOCK_GDELT_DF
        out = gdelt_client.search_articles(["DRAM"], max_records=100)
        self.assertEqual(len(out), 3)
        self.assertEqual(set(out[0].keys()),
                         {"title", "url", "domain", "seendate", "sourcecountry", "language"})
        self.assertEqual(out[0]["url"], "https://ex.com/dram-1")
        self.assertEqual(out[0]["sourcecountry"], "United States")

    @mock.patch("ingestion.gdelt_client.GdeltDoc")
    def test_empty_dataframe_returns_empty_list(self, mock_gdelt):
        mock_gdelt.return_value.article_search.return_value = pd.DataFrame()
        self.assertEqual(gdelt_client.search_articles(["DRAM"]), [])

    @mock.patch("ingestion.gdelt_client.GdeltDoc")
    def test_network_error_is_swallowed(self, mock_gdelt):
        mock_gdelt.return_value.article_search.side_effect = ConnectionError("boom")
        # max_retries=1 avoids the real backoff sleeps in this offline test.
        self.assertEqual(gdelt_client.search_articles(["DRAM"], max_retries=1), [])

    @mock.patch("ingestion.gdelt_client.time.sleep", lambda *_: None)
    @mock.patch("ingestion.gdelt_client.GdeltDoc")
    def test_retries_then_succeeds(self, mock_gdelt):
        # First attempt rate-limited, second returns data — retry recovers it.
        mock_gdelt.return_value.article_search.side_effect = [RuntimeError("429"), MOCK_GDELT_DF]
        out = gdelt_client.search_articles(["DRAM"], max_retries=3)
        self.assertEqual(len(out), 3)


class NewsApiClientTest(unittest.TestCase):
    @mock.patch.dict(os.environ, {"NEWSAPI_KEY": "test-key"})
    @mock.patch("ingestion.newsapi_client.requests.get")
    def test_normalizes_response(self, mock_get):
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.json.return_value = MOCK_NEWSAPI_JSON
        out = newsapi_client.search_articles("NAND", max_page_size=50)
        self.assertEqual(len(out), 1)
        self.assertEqual(set(out[0].keys()),
                         {"title", "url", "source_name", "description", "publishedAt", "language"})
        self.assertEqual(out[0]["source_name"], "TechWire")
        self.assertEqual(out[0]["url"], "https://news.com/nand-1")

    def test_missing_key_returns_empty_list_not_exception(self):
        # Explicitly unset the key for this test.
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertNotIn("NEWSAPI_KEY", os.environ)
            with mock.patch("ingestion.newsapi_client.requests.get") as mock_get:
                out = newsapi_client.search_articles("NAND")
                self.assertEqual(out, [])
                mock_get.assert_not_called()  # never even attempted a request

    @mock.patch.dict(os.environ, {"NEWSAPI_KEY": "test-key"})
    @mock.patch("ingestion.newsapi_client.requests.get")
    def test_request_failure_returns_empty_list(self, mock_get):
        mock_get.side_effect = Exception("timeout")
        self.assertEqual(newsapi_client.search_articles("NAND"), [])


class TaggerTest(unittest.TestCase):
    def setUp(self):
        self.rcat = _small_resource_catalog()
        self.pcat = _small_producer_catalog()

    def test_dram_tags_to_dram(self):
        rids, _ = tagger.tag_article("DRAM prices climb", None, self.rcat, self.pcat)
        self.assertEqual(rids, ["dram"])

    def test_hbm_and_gpu_tags_to_both(self):
        rids, pid = tagger.tag_article(
            "Nvidia HBM and GPU demand surges", None, self.rcat, self.pcat)
        self.assertIn("hbm", rids)
        self.assertIn("gpu", rids)
        self.assertEqual(len(rids), 2)
        self.assertEqual(pid, "nvidia")  # producer tagged too

    def test_unrelated_article_tags_to_nothing(self):
        rids, pid = tagger.tag_article(
            "Local bakery wins county fair ribbon", None, self.rcat, self.pcat)
        self.assertEqual(rids, [])
        self.assertIsNone(pid)

    def test_word_boundary_avoids_substring_false_positive(self):
        # "aluminum" contains "mu" (Micron ticker) but must NOT match.
        pcat = [("micron", ["MU"])]
        _, pid = tagger.tag_article("aluminum smelter update", None, self.rcat, pcat)
        self.assertIsNone(pid)


class IngestionInsertTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "test.db"
        os.environ["SUPPLY_CHAIN_DB"] = str(self.db_path)
        db_init.init_db(self.db_path)
        self._seed_resources()

    def tearDown(self):
        os.environ.pop("SUPPLY_CHAIN_DB", None)
        self._tmp.cleanup()

    def _seed_resources(self):
        conn = db_init.get_connection(self.db_path)
        rows = [
            ("dram", "DRAM", ["DRAM", "Dynamic RAM"]),
            ("hbm", "HBM", ["HBM", "High Bandwidth Memory"]),
            ("nand", "NAND", ["NAND", "NAND flash"]),
            ("gpu", "GPUs", ["GPU", "GPUs"]),
        ]
        for rid, name, aliases in rows:
            conn.execute(
                "INSERT INTO resource (resource_id, name, aliases) VALUES (?, ?, ?)",
                (rid, name, json.dumps(aliases)),
            )
        # Producer referenced by producer tagging (FK on signal.producer_id).
        conn.execute("INSERT INTO producer (producer_id, name) VALUES ('nvidia', 'Nvidia')")
        conn.execute("INSERT INTO producer (producer_id, name) VALUES ('micron', 'Micron')")
        conn.commit()
        conn.close()

    def _run_pipeline(self):
        """Run the real run() with both APIs mocked; NEWSAPI_KEY present."""
        with mock.patch("ingestion.gdelt_client.GdeltDoc") as mock_gdelt, \
             mock.patch("ingestion.newsapi_client.requests.get") as mock_get, \
             mock.patch.dict(os.environ, {"NEWSAPI_KEY": "test-key"}):
            mock_gdelt.return_value.article_search.return_value = MOCK_GDELT_DF
            mock_get.return_value.raise_for_status = lambda: None
            mock_get.return_value.json.return_value = MOCK_NEWSAPI_JSON
            ingest_news.run(self.db_path)

    def test_inserts_then_idempotent_second_run(self):
        self._run_pipeline()
        conn = db_init.get_connection(self.db_path)
        first_count = conn.execute(
            "SELECT COUNT(*) FROM signal WHERE signal_type='news_raw'").fetchone()[0]
        conn.close()
        # DRAM(1) + HBM+GPU(2 from one article) + NAND(1 from newsapi, x4 resources'
        # queries but deduped by url+resource) = 4 distinct (url,resource) rows.
        self.assertEqual(first_count, 4)

        # Second identical run must insert zero new rows.
        self._run_pipeline()
        conn = db_init.get_connection(self.db_path)
        second_count = conn.execute(
            "SELECT COUNT(*) FROM signal WHERE signal_type='news_raw'").fetchone()[0]
        conn.close()
        self.assertEqual(second_count, first_count)

    def test_only_valid_resource_slugs_never_null(self):
        self._run_pipeline()
        conn = db_init.get_connection(self.db_path)
        rids = {r[0] for r in conn.execute("SELECT DISTINCT resource_id FROM signal")}
        conn.close()
        self.assertTrue(rids.issubset({"dram", "hbm", "nand", "gpu"}))
        self.assertNotIn(None, rids)

    def test_scoring_fields_always_null(self):
        self._run_pipeline()
        conn = db_init.get_connection(self.db_path)
        rows = conn.execute(
            "SELECT value, unit, severity, confidence FROM signal "
            "WHERE signal_type='news_raw'").fetchall()
        conn.close()
        self.assertTrue(rows)  # there is at least one row
        for value, unit, severity, confidence in rows:
            self.assertIsNone(value)
            self.assertIsNone(severity)
            self.assertIsNone(confidence)
            self.assertIsNone(unit)


if __name__ == "__main__":
    unittest.main(verbosity=2)
