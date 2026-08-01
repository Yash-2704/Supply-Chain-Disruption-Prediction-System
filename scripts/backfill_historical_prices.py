"""One-off historical backfill of daily producer stock closes, 2023–2025.

WHY THIS EXISTS
---------------
The daily market ingest (scripts/ingest_market_data.py) is capped at a 90-day
lookback, so price_series only ever held recent (2026) closes. The curated
tightening episodes, however, live in 2023–2025:

    dram  2024-03 .. 2025-06      hbm  2024-01 .. 2025-12      gpu  2023-05 .. 2025-06

With no price history in those windows, the fusion model had no REAL,
NON-CIRCULAR (feature, label) pairs to train on. This script backfills the gap
so producer-stock features finally overlap the independent labels.

WHAT IT DOES
------------
Fetches daily USD closes over an explicit date range for the three USD-listed
producers whose history yfinance returns cleanly:

    MU   -> dram, hbm, nand      (Micron)
    NVDA -> gpu                  (Nvidia)
    TSM  -> gpu                  (TSMC)

Samsung (005930.KS = KRW, SSNLF = corrupt history) and SK Hynix (000660.KS =
KRW) are intentionally excluded — the USD-proxy semantics forbid loading a
KRW quote as USD. Their resources (dram/hbm/nand) are still covered by MU.

Rows are written with the SAME source ('yfinance') and unit
('USD_per_share_proxy') and the SAME dedup key (resource_id, timestamp, source)
as the daily ingest, so:
  * re-running is idempotent (existing rows are skipped), and
  * the forecasting / feature stages consume the extended series with no change.

Run:  python scripts/backfill_historical_prices.py [--start 2023-01-01] [--end 2026-01-01]
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from ingestion import yfinance_client  # noqa: E402

# yfinance `end` is EXCLUSIVE, so the default reaches through 2025-12-31.
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2026-01-01"

# Only USD-listed producers with clean yfinance history. Each ticker maps to the
# resources its producer makes (mirrors producer_resource); MU covers memory,
# NVDA + TSM both proxy GPUs.
BACKFILL_TICKERS = {
    "MU":   ["dram", "hbm", "nand"],
    "NVDA": ["gpu"],
    "TSM":  ["gpu"],
}


def _existing_keys(conn):
    """Existing (resource_id, timestamp, source) keys for idempotent re-runs."""
    return {(rid, ts, src) for rid, ts, src in conn.execute(
        "SELECT resource_id, timestamp, source FROM price_series")}


def _known_resources(conn):
    return {rid for (rid,) in conn.execute("SELECT resource_id FROM resource")}


def run(db_path=None, start=DEFAULT_START, end=DEFAULT_END):
    conn = db_init.get_connection(db_path)
    try:
        existing = _existing_keys(conn)
        known = _known_resources(conn)
        print("=" * 64)
        print(f"HISTORICAL PRICE BACKFILL  range=[{start} .. {end})  source=yfinance")
        print("=" * 64)

        grand_inserted = 0
        for ticker, resource_ids in BACKFILL_TICKERS.items():
            # Guard against a resource id drift vs the schema.
            targets = [r for r in resource_ids if r in known]
            missing = [r for r in resource_ids if r not in known]
            if missing:
                print(f"[{ticker:>5}] WARNING: unknown resource(s) {missing} skipped "
                      f"(not in resource table).")
            if not targets:
                print(f"[{ticker:>5}] no valid target resources -> skipped")
                continue

            closes = yfinance_client.get_daily_closes_range(ticker, start, end)
            if not closes:
                print(f"[{ticker:>5}] no USD price data for [{start}..{end}) -> skipped "
                      f"(non-USD listing or empty history)")
                continue

            inserted = skipped = 0
            for bar in closes:
                ts = bar["date"]
                for rid in targets:
                    key = (rid, ts, "yfinance")
                    if key in existing:
                        skipped += 1
                        continue
                    conn.execute(
                        "INSERT INTO price_series (resource_id, timestamp, price_usd, unit, source) "
                        "VALUES (?, ?, ?, 'USD_per_share_proxy', 'yfinance')",
                        (rid, ts, bar["close_price"]),
                    )
                    existing.add(key)
                    inserted += 1
            conn.commit()
            grand_inserted += inserted
            span = f"{closes[0]['date']}..{closes[-1]['date']}"
            print(f"[{ticker:>5}] days={len(closes):4d} span={span} "
                  f"resources={targets} inserted={inserted:4d} skipped_dup={skipped:4d}")

        print("=" * 64)
        print(f"TOTAL inserted this run: {grand_inserted}")
        _report_coverage(conn)
        print("=" * 64)
    finally:
        conn.close()


def _report_coverage(conn):
    """Show, per resource, the yfinance date span now available — the whole point
    of the backfill is that these now span the 2023–2025 episode windows."""
    print("\nyfinance price coverage per resource (post-backfill):")
    rows = conn.execute(
        "SELECT resource_id, COUNT(*), MIN(timestamp), MAX(timestamp) "
        "FROM price_series WHERE source='yfinance' GROUP BY resource_id ORDER BY resource_id"
    ).fetchall()
    for rid, n, lo, hi in rows:
        print(f"  {str(rid):>6}: {n:5d} rows  {lo} .. {hi}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Backfill 2023–2025 daily producer closes.")
    ap.add_argument("--start", default=DEFAULT_START, help=f"ISO start (default {DEFAULT_START}).")
    ap.add_argument("--end", default=DEFAULT_END,
                    help=f"ISO end, EXCLUSIVE (default {DEFAULT_END}).")
    args = ap.parse_args()
    run(start=args.start, end=args.end)
