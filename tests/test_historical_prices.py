"""Offline tests for the Phase-1 historical price additions:
  * yfinance_client.get_daily_closes_range  (uncapped backfill path)
  * ingestion.dam_price_client              (real USD/GB memory prices)
  * scripts.ingest_dam_prices               (idempotent insert, isolation)

No network: yfinance Ticker and the DAM HTTP GET are mocked. Inserts run against
a temp DB seeded with the real schema. Run:
    python -m pytest tests/test_historical_prices.py
"""
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from ingestion import yfinance_client, dam_price_client  # noqa: E402
import scripts.ingest_dam_prices as dam_ingest  # noqa: E402


class FakeTicker:
    def __init__(self, history_df, currency="USD"):
        self._h = history_df
        self.fast_info = {"currency": currency}

    def history(self, period=None, interval=None, start=None, end=None):
        return self._h


def _range_history(n_days=800, currency="USD"):
    idx = pd.to_datetime([date(2023, 1, 3) + timedelta(days=i) for i in range(n_days)])
    return pd.DataFrame({"Close": [100.0 + i * 0.1 for i in range(n_days)]}, index=idx)


class GetDailyClosesRangeTest(unittest.TestCase):
    def test_range_not_capped_at_90_days(self):
        # The backfill path must return the FULL range, not clip to 90 like the
        # daily path — that cap is exactly what blocked historical features.
        with mock.patch.object(yfinance_client.yf, "Ticker",
                               return_value=FakeTicker(_range_history(800))):
            out = yfinance_client.get_daily_closes_range("MU", "2023-01-01", "2026-01-01")
        self.assertEqual(len(out), 800)
        self.assertEqual(set(out[0]), {"date", "close_price"})

    def test_range_rejects_non_usd(self):
        with mock.patch.object(yfinance_client.yf, "Ticker",
                               return_value=FakeTicker(_range_history(10), currency="KRW")):
            with self.assertLogs("ingestion.yfinance_client", level="WARNING"):
                out = yfinance_client.get_daily_closes_range("005930.KS", "2023-01-01", "2026-01-01")
        self.assertEqual(out, [])

    def test_range_empty_history_returns_empty(self):
        with mock.patch.object(yfinance_client.yf, "Ticker",
                               return_value=FakeTicker(pd.DataFrame())):
            with self.assertLogs("ingestion.yfinance_client", level="WARNING"):
                out = yfinance_client.get_daily_closes_range("BAD", "2023-01-01", "2026-01-01")
        self.assertEqual(out, [])


_FAKE_CSV = (
    "date,category,series,metric,value,unit,source\n"
    "2024-05-01,DRAM,DRAM cheapest (Keepa),usd_per_gb,2.50,USD/GB,keepa\n"
    "2024-05-01,DRAM,DDR5 (Keepa),usd_per_gb,3.10,USD/GB,keepa\n"      # same month, diff series
    "2024-05-01,NAND,NAND cheapest (Keepa),usd_per_gb,0.05,USD/GB,keepa\n"
    "2024-05-01,HBM,HBM $/GB,usd_per_gb,25.0,USD/GB,est\n"
    "2024-05-01,DRAM,market share,percent_share,40,% of component cost,x\n"  # not a price -> skip
    "2024-06-01,GPU,GPU price,usd_per_gb,9.9,USD/GB,x\n"                # unmapped category -> skip
    "2024-06-01,NAND,NAND cheapest (Keepa),usd_per_gb,,USD/GB,keepa\n"  # blank value -> skip
)


class _FakeResp:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


class DamPriceClientTest(unittest.TestCase):
    def test_parses_only_price_rows_of_mapped_resources(self):
        with mock.patch.object(dam_price_client.requests, "get",
                               return_value=_FakeResp(_FAKE_CSV)):
            rows = dam_price_client.get_memory_prices()
        # 4 valid usd_per_gb rows for DRAM(x2)/NAND/HBM; percent_share, GPU, blank dropped.
        self.assertEqual(len(rows), 4)
        self.assertEqual({r["resource_id"] for r in rows}, {"dram", "nand", "hbm"})
        self.assertTrue(all(r["unit"] == "USD_per_GB" for r in rows))
        self.assertTrue(all(r["source"].startswith("stanford_dam:") for r in rows))
        # Two DRAM series in the same month get DISTINCT sources (no silent loss).
        dram_sources = {r["source"] for r in rows if r["resource_id"] == "dram"}
        self.assertEqual(len(dram_sources), 2)

    def test_fetch_failure_returns_empty(self):
        with mock.patch.object(dam_price_client.requests, "get",
                               side_effect=Exception("network down")):
            with self.assertLogs("ingestion.dam_price_client", level="WARNING"):
                rows = dam_price_client.get_memory_prices()
        self.assertEqual(rows, [])

    def test_recency_filter_drops_ancient_and_future(self):
        # Ancient (pre-MIN_YEAR) and future-dated rows must be excluded; only the
        # in-window row survives. (DAM's real CSV reaches back to 1957 and carries
        # forward projections — neither is a usable price feature.)
        csv_text = (
            "date,category,series,metric,value,unit,source\n"
            "1990-01-01,DRAM,McCallum DRAM (historical),usd_per_gb,50000.0,USD/GB,mccallum\n"
            "2100-01-01,HBM,HBM $/GB,usd_per_gb,20.0,USD/GB,projection\n"
            "2024-05-01,NAND,NAND cheapest (Keepa),usd_per_gb,0.05,USD/GB,keepa\n"
        )
        with mock.patch.object(dam_price_client.requests, "get",
                               return_value=_FakeResp(csv_text)):
            rows = dam_price_client.get_memory_prices()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["resource_id"], "nand")
        self.assertEqual(rows[0]["date"], "2024-05-01")


def _seed_resources(db_path):
    db_init.init_db(db_path)
    conn = db_init.get_connection(db_path)
    for rid, name in [("dram", "DRAM"), ("hbm", "HBM"), ("nand", "NAND"), ("gpu", "GPUs")]:
        conn.execute("INSERT INTO resource (resource_id, name, aliases) VALUES (?, ?, '[]')",
                     (rid, name))
    conn.commit()
    conn.close()


class DamIngestTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "test.db"
        os.environ["SUPPLY_CHAIN_DB"] = str(self.db_path)
        _seed_resources(self.db_path)

    def tearDown(self):
        os.environ.pop("SUPPLY_CHAIN_DB", None)
        self._tmp.cleanup()

    def _run(self):
        with mock.patch.object(dam_price_client.requests, "get",
                               return_value=_FakeResp(_FAKE_CSV)):
            dam_ingest.run(self.db_path)

    def test_inserts_usd_per_gb_rows(self):
        self._run()
        conn = db_init.get_connection(self.db_path)
        n = conn.execute("SELECT COUNT(*) FROM price_series WHERE unit='USD_per_GB'").fetchone()[0]
        conn.close()
        self.assertEqual(n, 4)

    def test_does_not_touch_stock_proxy_path(self):
        # DAM rows must NOT be counted as USD_per_share_proxy (feature/forecast scope).
        self._run()
        conn = db_init.get_connection(self.db_path)
        proxy = conn.execute(
            "SELECT COUNT(*) FROM price_series WHERE unit='USD_per_share_proxy'").fetchone()[0]
        conn.close()
        self.assertEqual(proxy, 0)

    def test_idempotent_second_run(self):
        self._run()
        conn = db_init.get_connection(self.db_path)
        first = conn.execute("SELECT COUNT(*) FROM price_series").fetchone()[0]
        conn.close()
        self._run()
        conn = db_init.get_connection(self.db_path)
        second = conn.execute("SELECT COUNT(*) FROM price_series").fetchone()[0]
        conn.close()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main(verbosity=2)
