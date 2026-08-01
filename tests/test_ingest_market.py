"""Required, fully-offline tests for the market-data ingestion pipeline.

No real network: yfinance's Ticker and the macro HTTP session are mocked.
Inserts run against a temp DB seeded with the real schema. Run:
    python -m pytest tests/test_ingest_market.py
"""
import json
import logging
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
from ingestion import yfinance_client, macro_price_client  # noqa: E402
import scripts.ingest_market_data as market  # noqa: E402


# ---- fakes mimicking yfinance ---------------------------------------------

def _fake_history(n_days=5, currency="USD", with_nan=False):
    idx = pd.to_datetime([date(2026, 6, 1) + timedelta(days=i) for i in range(n_days)])
    closes = [100.0 + i for i in range(n_days)]
    if with_nan:
        closes[-1] = float("nan")
    return pd.DataFrame({"Close": closes}, index=idx)


class FakeTicker:
    def __init__(self, history_df, currency="USD"):
        self._h = history_df
        self.fast_info = {"currency": currency}

    def history(self, period=None, interval=None):
        return self._h


def _seed_db(db_path):
    db_init.init_db(db_path)
    conn = db_init.get_connection(db_path)
    for rid, name in [("dram", "DRAM"), ("hbm", "HBM"), ("nand", "NAND"), ("gpu", "GPUs")]:
        conn.execute("INSERT INTO resource (resource_id, name, aliases) VALUES (?, ?, '[]')",
                     (rid, name))
    for pid, name in [("micron", "Micron"), ("nvidia", "Nvidia"), ("samsung", "Samsung"),
                      ("sk_hynix", "SK Hynix"), ("tsmc", "TSMC")]:
        conn.execute("INSERT INTO producer (producer_id, name) VALUES (?, ?)", (pid, name))
    for pid, rid in [("micron", "dram"), ("micron", "hbm"), ("micron", "nand"),
                     ("nvidia", "gpu"), ("tsmc", "gpu"),
                     ("samsung", "dram"), ("samsung", "hbm"), ("samsung", "nand"),
                     ("sk_hynix", "dram"), ("sk_hynix", "hbm"), ("sk_hynix", "nand")]:
        conn.execute("INSERT INTO producer_resource (producer_id, resource_id) VALUES (?, ?)",
                     (pid, rid))
    conn.commit()
    conn.close()


class YFinanceClientTest(unittest.TestCase):
    def test_valid_ticker_normalizes(self):
        with mock.patch.object(yfinance_client.yf, "Ticker",
                               return_value=FakeTicker(_fake_history(5))):
            out = yfinance_client.get_daily_closes("MU", 90)
        self.assertEqual(len(out), 5)
        self.assertEqual(set(out[0].keys()), {"date", "close_price"})
        self.assertEqual(out[0]["close_price"], 100.0)

    def test_delisted_ticker_returns_empty_and_warns(self):
        with mock.patch.object(yfinance_client.yf, "Ticker",
                               return_value=FakeTicker(pd.DataFrame())):
            with self.assertLogs("ingestion.yfinance_client", level="WARNING") as cm:
                out = yfinance_client.get_daily_closes("BADTICKER", 90)
        self.assertEqual(out, [])
        self.assertTrue(any("BADTICKER" in m for m in cm.output))

    def test_non_usd_currency_rejected(self):
        with mock.patch.object(yfinance_client.yf, "Ticker",
                               return_value=FakeTicker(_fake_history(5), currency="KRW")):
            with self.assertLogs("ingestion.yfinance_client", level="WARNING"):
                out = yfinance_client.get_daily_closes("000660.KS", 90)
        self.assertEqual(out, [])

    def test_nan_closes_skipped(self):
        with mock.patch.object(yfinance_client.yf, "Ticker",
                               return_value=FakeTicker(_fake_history(5, with_nan=True))):
            out = yfinance_client.get_daily_closes("MU", 90)
        self.assertEqual(len(out), 4)  # last NaN dropped


class MacroResourceIdNullableTest(unittest.TestCase):
    """After the nullability migration, price_series.resource_id allows NULL and
    macro rows are stored (resource-agnostic) instead of being blocked."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "test.db"
        _seed_db(self.db_path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_price_series_resource_id_is_nullable(self):
        conn = db_init.get_connection(self.db_path)
        self.assertTrue(market._resource_id_nullable(conn))
        conn.close()

    def test_null_resource_id_macro_insert_succeeds(self):
        # A resource-agnostic macro row (resource_id=NULL) is now accepted.
        conn = db_init.get_connection(self.db_path)
        conn.execute(
            "INSERT INTO price_series (resource_id, timestamp, price_usd, unit, source) "
            "VALUES (NULL, '2024-12-01', 80.5, 'Crude oil ($/bbl)', 'world_bank_pink_sheet')")
        conn.commit()
        n = conn.execute("SELECT COUNT(*) FROM price_series WHERE resource_id IS NULL").fetchone()[0]
        conn.close()
        self.assertEqual(n, 1)

    def test_orchestrator_inserts_macro_rows(self):
        os.environ["SUPPLY_CHAIN_DB"] = str(self.db_path)
        try:
            fake_rows = [{"date": "2024-12-01", "commodity_name": "Crude oil, Brent",
                          "price": 74.0, "unit": "($/bbl)"}]
            with mock.patch.object(macro_price_client, "get_latest_prices",
                                   return_value=fake_rows), \
                 mock.patch.object(yfinance_client, "get_daily_closes", return_value=[]):
                market.run(self.db_path)
            # The macro row IS persisted now (resource_id NULL), not blocked.
            conn = db_init.get_connection(self.db_path)
            n = conn.execute("SELECT COUNT(*) FROM price_series "
                             "WHERE resource_id IS NULL AND source='world_bank_pink_sheet'").fetchone()[0]
            conn.close()
            self.assertEqual(n, 1)
        finally:
            os.environ.pop("SUPPLY_CHAIN_DB", None)


class OrchestrationStockTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "test.db"
        os.environ["SUPPLY_CHAIN_DB"] = str(self.db_path)
        _seed_db(self.db_path)

    def tearDown(self):
        os.environ.pop("SUPPLY_CHAIN_DB", None)
        self._tmp.cleanup()

    def _run(self, closes_by_ticker):
        """Run orchestrator with yfinance mocked per-ticker; macro fetch empty."""
        def fake_closes(ticker, lookback_days=90, require_currency="USD"):
            return list(closes_by_ticker.get(ticker, []))
        with mock.patch.object(yfinance_client, "get_daily_closes", side_effect=fake_closes), \
             mock.patch.object(macro_price_client, "get_latest_prices", return_value=[]):
            market.run(self.db_path)

    def _bars(self, n=3):
        return [{"date": f"2026-06-0{i+1}", "close_price": 100.0 + i} for i in range(n)]

    def test_valid_ticker_inserts_proxy_rows_with_correct_unit(self):
        # Only NVDA (gpu) resolves; tsmc also makes gpu but its ticker returns [].
        self._run({"NVDA": self._bars(3)})
        conn = db_init.get_connection(self.db_path)
        rows = conn.execute(
            "SELECT resource_id, unit, source, price_usd FROM price_series").fetchall()
        conn.close()
        self.assertEqual(len(rows), 3)  # 3 days x 1 resource (gpu)
        for rid, unit, source, price in rows:
            self.assertEqual(rid, "gpu")
            self.assertEqual(unit, "USD_per_share_proxy")  # never a commodity-like unit
            self.assertEqual(source, "yfinance")

    def test_delisted_ticker_produces_zero_rows_no_crash(self):
        # sk_hynix has only a KRW ticker in real life; here all tickers return [].
        self._run({})  # nothing resolves
        conn = db_init.get_connection(self.db_path)
        n = conn.execute("SELECT COUNT(*) FROM price_series").fetchone()[0]
        conn.close()
        self.assertEqual(n, 0)

    def test_no_proxy_row_is_ever_null_resource(self):
        self._run({"MU": self._bars(2)})
        conn = db_init.get_connection(self.db_path)
        bad = conn.execute(
            "SELECT COUNT(*) FROM price_series "
            "WHERE unit='USD_per_share_proxy' AND resource_id IS NULL").fetchone()[0]
        conn.close()
        self.assertEqual(bad, 0)

    def test_idempotent_second_run(self):
        self._run({"MU": self._bars(3)})
        conn = db_init.get_connection(self.db_path)
        first = conn.execute("SELECT COUNT(*) FROM price_series").fetchone()[0]
        conn.close()
        self.assertGreater(first, 0)
        self._run({"MU": self._bars(3)})  # identical data again
        conn = db_init.get_connection(self.db_path)
        second = conn.execute("SELECT COUNT(*) FROM price_series").fetchone()[0]
        conn.close()
        self.assertEqual(first, second)


if __name__ == "__main__":
    logging.disable(logging.NOTSET)
    unittest.main(verbosity=2)
