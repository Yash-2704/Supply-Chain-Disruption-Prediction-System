"""Orchestrate SEC EDGAR ingestion into the `signal` table, in two parts.

Part A — capacity signals: latest capex (XBRL) for the 5 producers, one row per
resource that producer makes (via producer_resource), signal_type
'capacity_expansion', value set, producer_id set.

Part B — demand-intent signals: recent 8-K metadata for the 5 producers AND the
4 hyperscaler demand filers, one row per mapped resource, signal_type
'demand_intent_raw', value NULL. Producers attribute via producer_resource;
demand filers via data/demand_resource_mapping.json (producer_id stays NULL).

Dedup: (document-id, resource_id) composite, matching the news pipeline.
Run:  python scripts/ingest_sec_filings.py
"""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from ingestion import sec_edgar_client as sec  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
ALIAS_DICT_PATH = DATA_DIR / "alias_dictionary.json"
DEMAND_FILERS_PATH = DATA_DIR / "demand_filers.json"
DEMAND_MAPPING_PATH = DATA_DIR / "demand_resource_mapping.json"

EIGHTK_CAP = 20

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ingest_sec")


# ---- catalog loaders -------------------------------------------------------

def _sec_ticker(tickers):
    """Pick a US-listable ticker (no exchange suffix like '.KS'/'.TW'), else None."""
    return next((t for t in tickers if "." not in t), None)


def load_producers(conn):
    """[(producer_id, name, sec_ticker, [resource_ids])] using producer_resource."""
    aliases = json.loads(ALIAS_DICT_PATH.read_text(encoding="utf-8"))
    tickers_by_id = {p["producer_id"]: p.get("tickers", []) for p in aliases["producers"]}

    res_by_producer = {}
    for pid, rid in conn.execute("SELECT producer_id, resource_id FROM producer_resource"):
        res_by_producer.setdefault(pid, []).append(rid)

    producers = []
    for pid, name in conn.execute("SELECT producer_id, name FROM producer ORDER BY producer_id"):
        producers.append((pid, name, _sec_ticker(tickers_by_id.get(pid, [])),
                          sorted(res_by_producer.get(pid, []))))
    return producers


def load_demand_filers():
    filers = json.loads(DEMAND_FILERS_PATH.read_text(encoding="utf-8"))["demand_filers"]
    mapping = {m["ticker"]: m["resource_ids"]
               for m in json.loads(DEMAND_MAPPING_PATH.read_text(encoding="utf-8"))["mappings"]}
    return [(f["name"], f["ticker"], mapping.get(f["ticker"], [])) for f in filers]


def _existing_pairs(conn):
    return {(url, rid) for url, rid in conn.execute(
        "SELECT source_url, resource_id FROM signal WHERE source_url IS NOT NULL")}


# ---- Part A: capacity signals ---------------------------------------------

def ingest_capacity_signals(conn, producers, existing_pairs, ingested_at):
    print("\n--- PART A: capacity_expansion (producer capex) ---")
    for pid, name, ticker, resource_ids in producers:
        if not ticker:
            print(f"[{pid:>8}] no US-listable ticker -> skipped")
            continue
        capex = sec.get_latest_capex(ticker)
        if not capex:
            print(f"[{pid:>8} {ticker:>5}] no capex (IFRS/foreign/untagged) -> skipped")
            continue
        doc_id = f"sec-xbrl:{ticker}:{capex['xbrl_tag']}:{capex['fiscal_year']}:{capex['fiscal_period']}"
        raw_text = f"{capex['xbrl_tag']} {capex['fiscal_year']} {capex['fiscal_period']}"
        inserted = skipped = 0
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
        print(f"[{pid:>8} {ticker:>5}] capex={capex['value']:,.0f} USD "
              f"({capex['fiscal_year']} {capex['fiscal_period']}) "
              f"resources={len(resource_ids)} inserted={inserted} skipped_dup={skipped}")


# ---- Part B: demand-intent signals ----------------------------------------

def _insert_filings(conn, source_name, producer_id, resource_ids, filings,
                    existing_pairs, ingested_at):
    """Insert one demand_intent_raw row per (filing, mapped resource). Metadata only."""
    fetched = len(filings)
    inserted = skipped = 0
    for f in filings:
        # doc_id must equal the persisted source_url so cross-run dedup (which
        # reloads from signal.source_url) matches. Filing URL is unique+stable.
        doc_id = f.get("url") or f.get("accession_no")
        if not doc_id:
            continue
        # raw_text is METADATA ONLY — form type + item codes + short description.
        raw_text = " ".join(str(p) for p in (
            f.get("form_type"), f.get("items") and f"item {f['items']}", f.get("description")
        ) if p)
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
                ) VALUES (?, 'demand_intent_raw', ?, ?, NULL, NULL, NULL, NULL, NULL, ?, ?, ?, ?)
                """,
                (f.get("filing_date"), rid, producer_id, source_name,
                 doc_id, raw_text, ingested_at),
            )
            existing_pairs.add((doc_id, rid))
            inserted += 1
    return fetched, inserted, skipped


def ingest_demand_signals(conn, producers, demand_filers, existing_pairs, ingested_at):
    print("\n--- PART B: demand_intent_raw (8-K metadata) ---")
    # Producers: producer_id set, resources via producer_resource.
    for pid, name, ticker, resource_ids in producers:
        if not ticker:
            print(f"[{pid:>18}] no US-listable ticker -> skipped")
            continue
        filings = sec.get_recent_filings(ticker, form_type="8-K", max_count=EIGHTK_CAP)
        fetched, inserted, skipped = _insert_filings(
            conn, name, pid, resource_ids, filings, existing_pairs, ingested_at)
        conn.commit()
        print(f"[producer {name:>10} {ticker:>5}] fetched={fetched:2d} "
              f"inserted={inserted:3d} skipped_dup={skipped:3d}")

    # Demand filers: producer_id NULL, resources via hand-curated mapping.
    for name, ticker, resource_ids in demand_filers:
        filings = sec.get_recent_filings(ticker, form_type="8-K", max_count=EIGHTK_CAP)
        fetched, inserted, skipped = _insert_filings(
            conn, name, None, resource_ids, filings, existing_pairs, ingested_at)
        conn.commit()
        print(f"[demand   {name:>10} {ticker:>5}] fetched={fetched:2d} "
              f"inserted={inserted:3d} skipped_dup={skipped:3d} (producer_id=NULL)")


def run(db_path=None):
    conn = db_init.get_connection(db_path)
    try:
        producers = load_producers(conn)
        demand_filers = load_demand_filers()
        existing_pairs = _existing_pairs(conn)
        ingested_at = datetime.now(timezone.utc).isoformat()

        print("=" * 64)
        ingest_capacity_signals(conn, producers, existing_pairs, ingested_at)
        ingest_demand_signals(conn, producers, demand_filers, existing_pairs, ingested_at)
        print("=" * 64)
    finally:
        conn.close()


if __name__ == "__main__":
    run()
