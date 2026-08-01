"""One-off HISTORICAL SEC ingest, 2023–2025, into the signal table.

WHY
---
The live SEC ingest (scripts/ingest_sec_filings.py) pulls only the LATEST capex
fact and the most-recent 8-Ks, so its signals sit in 2026 — outside the curated
2023–2025 episode windows. This script pulls the historical capex TRAJECTORY and
the 8-Ks filed within the window, so capacity/demand signals finally overlap the
independent labels.

WHAT
----
Part A — capacity_expansion: every historical capex fact (get_capex_history) per
producer in [start, end], one row per (fact, resource). doc_id is keyed by
fiscal_year/period, so each period is a distinct, deduped row.

Part B — demand_intent_raw: 8-K metadata filed in [start, end]
(get_filings_in_range) for producers AND demand filers, one row per (filing,
resource). Reuses the exact insert/dedup path from the live script.

Idempotent via (source_url/doc_id, resource_id). Safe to overlap the live ingest.

Run:  python scripts/ingest_sec_historical.py [--start 2023-01-01] [--end 2025-12-31]
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from ingestion import sec_edgar_client as sec  # noqa: E402
from scripts.ingest_sec_filings import (  # noqa: E402  (reuse live-ingest internals)
    load_producers, load_demand_filers, _existing_pairs, _insert_filings,
)

DEFAULT_START = "2023-01-01"
DEFAULT_END = "2025-12-31"
EIGHTK_CAP = 200   # far higher than the live cap; a 3-year window has many 8-Ks


def ingest_capacity_history(conn, producers, existing_pairs, ingested_at, start, end):
    print("\n--- PART A: capacity_expansion (historical capex trajectory) ---")
    for pid, name, ticker, resource_ids in producers:
        if not ticker:
            print(f"[{pid:>8}] no US-listable ticker -> skipped")
            continue
        facts = sec.get_capex_history(ticker, start_date=start, end_date=end)
        if not facts:
            print(f"[{pid:>8} {ticker:>5}] no historical capex in window -> skipped")
            continue
        inserted = skipped = 0
        for capex in facts:
            doc_id = (f"sec-xbrl:{ticker}:{capex['xbrl_tag']}:"
                      f"{capex['fiscal_year']}:{capex['fiscal_period']}")
            raw_text = f"{capex['xbrl_tag']} {capex['fiscal_year']} {capex['fiscal_period']}"
            for rid in resource_ids:
                if (doc_id, rid) in existing_pairs:
                    skipped += 1
                    continue
                conn.execute(
                    """
                    INSERT INTO signal (
                        timestamp, signal_type, resource_id, producer_id, country,
                        value, unit, severity, confidence,
                        source_name, source_url, raw_text, ingested_at
                    ) VALUES (?, 'capacity_expansion', ?, ?, NULL, ?, 'USD', NULL, NULL, ?, ?, ?, ?)
                    """,
                    (capex["period_end"], rid, pid, capex["value"],
                     "SEC EDGAR XBRL", doc_id, raw_text, ingested_at),
                )
                existing_pairs.add((doc_id, rid))
                inserted += 1
        conn.commit()
        span = f"{facts[0]['period_end']}..{facts[-1]['period_end']}"
        print(f"[{pid:>8} {ticker:>5}] capex_facts={len(facts):2d} span={span} "
              f"resources={len(resource_ids)} inserted={inserted:3d} skipped_dup={skipped:3d}")


def ingest_demand_history(conn, producers, demand_filers, existing_pairs,
                          ingested_at, start, end):
    print("\n--- PART B: demand_intent_raw (historical 8-K metadata) ---")
    for pid, name, ticker, resource_ids in producers:
        if not ticker:
            print(f"[{pid:>18}] no US-listable ticker -> skipped")
            continue
        filings = sec.get_filings_in_range(ticker, "8-K", start, end, max_count=EIGHTK_CAP)
        fetched, inserted, skipped = _insert_filings(
            conn, name, pid, resource_ids, filings, existing_pairs, ingested_at)
        conn.commit()
        print(f"[producer {name:>10} {ticker:>5}] fetched={fetched:3d} "
              f"inserted={inserted:3d} skipped_dup={skipped:3d}")

    for name, ticker, resource_ids in demand_filers:
        filings = sec.get_filings_in_range(ticker, "8-K", start, end, max_count=EIGHTK_CAP)
        fetched, inserted, skipped = _insert_filings(
            conn, name, None, resource_ids, filings, existing_pairs, ingested_at)
        conn.commit()
        print(f"[demand   {name:>10} {ticker:>5}] fetched={fetched:3d} "
              f"inserted={inserted:3d} skipped_dup={skipped:3d} (producer_id=NULL)")


def run(db_path=None, start=DEFAULT_START, end=DEFAULT_END):
    conn = db_init.get_connection(db_path)
    try:
        producers = load_producers(conn)
        demand_filers = load_demand_filers()
        existing_pairs = _existing_pairs(conn)
        ingested_at = datetime.now(timezone.utc).isoformat()
        print("=" * 64)
        print(f"HISTORICAL SEC INGEST  window=[{start} .. {end}]")
        print("=" * 64)
        ingest_capacity_history(conn, producers, existing_pairs, ingested_at, start, end)
        ingest_demand_history(conn, producers, demand_filers, existing_pairs,
                              ingested_at, start, end)
        _report_coverage(conn)
        print("=" * 64)
    finally:
        conn.close()


def _report_coverage(conn):
    print("\nSEC signal coverage inside 2023–2025:")
    rows = conn.execute(
        "SELECT signal_type, COUNT(*), MIN(timestamp), MAX(timestamp) FROM signal "
        "WHERE signal_type IN ('capacity_expansion','demand_intent_raw') "
        "AND timestamp >= '2023' AND timestamp < '2026' GROUP BY signal_type").fetchall()
    for st, n, lo, hi in rows:
        print(f"  {st:>18}: {n:4d} rows  {str(lo)[:10]} .. {str(hi)[:10]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Historical SEC capex + 8-K ingest, 2023–2025.")
    ap.add_argument("--start", default=DEFAULT_START, help=f"ISO start (default {DEFAULT_START}).")
    ap.add_argument("--end", default=DEFAULT_END, help=f"ISO end (default {DEFAULT_END}).")
    args = ap.parse_args()
    run(start=args.start, end=args.end)
