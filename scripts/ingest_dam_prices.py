"""Ingest REAL memory commodity prices (Stanford DAM) into price_series.

Adds actual USD/GB DRAM/NAND/HBM prices — a resource-level commodity signal,
distinct from the yfinance stock proxy. Stored with:
    unit   = 'USD_per_GB'
    source = 'stanford_dam:<series>'   (series embedded so nothing collides)
Dedup key (resource_id, timestamp, source) makes re-runs idempotent.

IMPORTANT — not silently wired into the model (yet), by design
--------------------------------------------------------------
The current feature builder and forecaster filter on
`unit = 'USD_per_share_proxy'`, so these USD_per_GB rows are STORED but NOT
automatically consumed. That is deliberate: it lands real data without covertly
changing model behavior, and — because price_derived labels are built from the
same price-series path — it avoids re-introducing price/label circularity.
Wiring the real commodity price in as its own feature is a separate, guarded
Phase-3 step.

Run:  python scripts/ingest_dam_prices.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from ingestion import dam_price_client  # noqa: E402


def _existing_keys(conn):
    return {(rid, ts, src) for rid, ts, src in conn.execute(
        "SELECT resource_id, timestamp, source FROM price_series")}


def _known_resources(conn):
    return {rid for (rid,) in conn.execute("SELECT resource_id FROM resource")}


def run(db_path=None):
    conn = db_init.get_connection(db_path)
    try:
        rows = dam_price_client.get_memory_prices()
        print("=" * 64)
        print(f"STANFORD DAM commodity prices  fetched={len(rows)} USD/GB rows")
        print("=" * 64)
        if not rows:
            print("nothing to insert (fetch/parse returned no rows).")
            return

        existing = _existing_keys(conn)
        known = _known_resources(conn)
        inserted = skipped = unknown = 0
        by_resource = {}
        for r in rows:
            rid = r["resource_id"]
            if rid not in known:
                unknown += 1
                continue
            key = (rid, r["date"], r["source"])
            if key in existing:
                skipped += 1
                continue
            conn.execute(
                "INSERT INTO price_series (resource_id, timestamp, price_usd, unit, source) "
                "VALUES (?, ?, ?, ?, ?)",
                (rid, r["date"], r["price_usd"], r["unit"], r["source"]),
            )
            existing.add(key)
            inserted += 1
            by_resource[rid] = by_resource.get(rid, 0) + 1
        conn.commit()

        print(f"inserted={inserted}  skipped_dup={skipped}  unknown_resource={unknown}")
        if by_resource:
            print("inserted by resource:", dict(sorted(by_resource.items())))
        _report_coverage(conn)
        print("=" * 64)
    finally:
        conn.close()


def _report_coverage(conn):
    print("\nStanford DAM (USD_per_GB) coverage per resource:")
    rows = conn.execute(
        "SELECT resource_id, COUNT(*), MIN(timestamp), MAX(timestamp) "
        "FROM price_series WHERE unit='USD_per_GB' GROUP BY resource_id ORDER BY resource_id"
    ).fetchall()
    for rid, n, lo, hi in rows:
        print(f"  {str(rid):>6}: {n:4d} rows  {lo} .. {hi}")


if __name__ == "__main__":
    run()
