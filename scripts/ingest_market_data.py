"""Orchestrate market-data ingestion into the `price_series` table, two parts.

Part A — producer stock proxies: daily USD closes per producer (yfinance), one
row per resource that producer makes (via producer_resource), with
unit='USD_per_share_proxy', source='yfinance'. A stock price is a weak
market-sentiment proxy for the producer's resources — NOT a resource price.

Part B — macro context: monthly World Bank Pink Sheet commodity prices, which
are resource-agnostic and therefore require resource_id = NULL.

Dedup key: (resource_id, timestamp, source), persisted == checked.
Run:  python scripts/ingest_market_data.py
"""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from ingestion import yfinance_client  # noqa: E402
# NOTE: `macro_price_client` is imported lazily inside ingest_macro_context() so
# that the price-only fast path (--no-macro, used by the dashboard's live fetch)
# does not pay the cost of importing `requests`/`openpyxl` — a large chunk of the
# cold-start import time on a slow disk.

ALIAS_DICT_PATH = PROJECT_ROOT / "data" / "alias_dictionary.json"
LOOKBACK_DAYS = 90
MACRO_MONTHS = 3

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ingest_market")


# ---- helpers ---------------------------------------------------------------

def load_producers(conn):
    """[(producer_id, name, [tickers], [resource_ids])] via producer_resource."""
    aliases = json.loads(ALIAS_DICT_PATH.read_text(encoding="utf-8"))
    tickers_by_id = {p["producer_id"]: p.get("tickers", []) for p in aliases["producers"]}
    res_by_producer = {}
    for pid, rid in conn.execute("SELECT producer_id, resource_id FROM producer_resource"):
        res_by_producer.setdefault(pid, []).append(rid)
    producers = []
    for pid, name in conn.execute("SELECT producer_id, name FROM producer ORDER BY producer_id"):
        producers.append((pid, name, tickers_by_id.get(pid, []),
                          sorted(res_by_producer.get(pid, []))))
    return producers


def _existing_keys(conn):
    """Load existing (resource_id, timestamp, source) dedup keys."""
    return {(rid, ts, src) for rid, ts, src in conn.execute(
        "SELECT resource_id, timestamp, source FROM price_series")}


def _resource_id_nullable(conn) -> bool:
    """True if price_series.resource_id permits NULL (PRAGMA notnull == 0)."""
    for cid, name, ctype, notnull, dflt, pk in conn.execute("PRAGMA table_info(price_series)"):
        if name == "resource_id":
            return notnull == 0
    return False


# ---- Part A: producer stock proxies ---------------------------------------

def ingest_stock_proxies(conn, producers, existing_keys, ingested_at):
    print("\n--- PART A: producer stock proxy (price_series, USD_per_share_proxy) ---")
    for pid, name, tickers, resource_ids in producers:
        used_ticker = closes = None
        for t in tickers:
            data = yfinance_client.get_daily_closes(t, LOOKBACK_DAYS)
            if data:
                used_ticker, closes = t, data
                break
        if not closes:
            print(f"[{pid:>8}] no USD price data from tickers {tickers} -> skipped")
            continue

        inserted = skipped = 0
        for bar in closes:
            ts = bar["date"]
            for rid in resource_ids:
                key = (rid, ts, "yfinance")
                if key in existing_keys:
                    skipped += 1
                    continue
                conn.execute(
                    "INSERT INTO price_series (resource_id, timestamp, price_usd, unit, source) "
                    "VALUES (?, ?, ?, 'USD_per_share_proxy', 'yfinance')",
                    (rid, ts, bar["close_price"]),
                )
                existing_keys.add(key)
                inserted += 1
        conn.commit()
        print(f"[{pid:>8} {used_ticker:>9}] days={len(closes):2d} "
              f"resources={len(resource_ids)} inserted={inserted:3d} skipped_dup={skipped:3d}")


# ---- Part B: macro context (schema-conflict aware) -------------------------

def ingest_macro_context(conn, existing_keys, ingested_at):
    print("\n--- PART B: macro commodity context (World Bank Pink Sheet) ---")
    from ingestion import macro_price_client  # lazy: see import note at top
    rows = macro_price_client.get_latest_prices(months_back=MACRO_MONTHS)
    source = macro_price_client.SOURCE_NAME
    print(f"[macro {source}] fetched={len(rows)} monthly rows from World Bank Pink Sheet")

    if not rows:
        logger.warning("macro: no rows fetched (WB URL may have rotated to 404); "
                       "nothing to insert this run.")
        return

    if not _resource_id_nullable(conn):
        # LOUD, deliberate flag. Macro data is resource-agnostic and MUST use
        # resource_id = NULL, but price_series.resource_id is NOT NULL and the
        # schema is frozen for this task. A sentinel resource_id is explicitly
        # forbidden. So we refuse to persist and surface the conflict.
        logger.error(
            "SCHEMA CONFLICT: %d World Bank macro rows were fetched and parsed "
            "successfully, but price_series.resource_id is declared NOT NULL. "
            "Macro commodity data is resource-agnostic (requires resource_id=NULL) "
            "and MUST NOT be force-mapped to a semiconductor resource or given a "
            "sentinel id. schema.sql may not be modified in this task. "
            "RESOLUTION NEEDED: make price_series.resource_id nullable (drop NOT "
            "NULL + adjust the FK) in a schema-migration task, then re-run. "
            "Skipping macro insertion.", len(rows))
        print(f"[macro {source}] inserted=0 skipped=0 -> BLOCKED by schema conflict "
              f"(price_series.resource_id is NOT NULL); see logged ERROR above.")
        return

    # Reached only if the schema is later fixed to allow NULL resource_id.
    latest_stored = conn.execute(
        "SELECT MAX(timestamp) FROM price_series WHERE source = ?", (source,)
    ).fetchone()[0]
    inserted = skipped = 0
    for r in rows:
        ts = r["date"]
        if latest_stored and ts <= latest_stored:
            skipped += 1
            continue
        key = (None, ts, source)
        if key in existing_keys:
            skipped += 1
            continue
        conn.execute(
            "INSERT INTO price_series (resource_id, timestamp, price_usd, unit, source) "
            "VALUES (NULL, ?, ?, ?, ?)",
            (ts, r["price"], f"{r['commodity_name']} {r['unit']}", source),
        )
        existing_keys.add(key)
        inserted += 1
    conn.commit()
    print(f"[macro {source}] inserted={inserted} skipped={skipped}")


def run(db_path=None, include_macro=True):
    """Ingest market data. Part A (producer stock proxies) always runs; it is the
    only part scoring uses and the only part that actually inserts rows. Part B
    (World Bank macro context) is optional: it is slow (large .xlsx download +
    parse) and currently inserts nothing (blocked by the resource_id NOT NULL
    schema), so the dashboard's live fetch runs with include_macro=False to stay
    well inside its time budget."""
    conn = db_init.get_connection(db_path)
    try:
        producers = load_producers(conn)
        existing_keys = _existing_keys(conn)
        ingested_at = datetime.now(timezone.utc).isoformat()
        print("=" * 64)
        ingest_stock_proxies(conn, producers, existing_keys, ingested_at)
        if include_macro:
            ingest_macro_context(conn, existing_keys, ingested_at)
        else:
            print("\n--- PART B: macro commodity context — skipped (--no-macro) ---")
        print("=" * 64)
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-macro", action="store_true",
        help="Skip the slow World Bank macro fetch (Part B); ingest producer "
             "prices only. Also enabled by env INGEST_SKIP_MACRO=1.")
    args = parser.parse_args()
    include_macro = not (args.no_macro or os.environ.get("INGEST_SKIP_MACRO") == "1")
    run(include_macro=include_macro)
