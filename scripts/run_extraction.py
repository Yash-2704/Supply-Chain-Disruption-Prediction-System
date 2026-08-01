"""Orchestrate LLM event extraction over eligible signal rows.

Eligible = canonical news_raw rows (CAST-aware join to news_cluster_membership)
+ demand_intent_raw rows, in both cases only where signal.severity IS NULL.
capacity_expansion and non-canonical news_raw rows are never selected.

Writes signal.severity/confidence AND an event_extraction audit row together,
per row, in one transaction. One row's failure never aborts the batch.
Run:  python scripts/run_extraction.py [--force]
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from llm_extraction import llm_client, prompts  # noqa: E402

EXTRACTION_SCHEMA_PATH = PROJECT_ROOT / "llm_extraction" / "schema_extraction.sql"

# Providers' free tiers vary (Groq ~1000 req/day, Cerebras 2400/day but 5/min,
# Mistral 50/min); 50 rows/run leaves headroom and stays under per-minute caps.
# The demand backlog clears over several idempotent runs. Raise this (and dedupe
# near-identical demand metadata) if volume grows substantially.
MAX_ROWS_PER_RUN = 50

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_extraction")

# Canonical, non-duplicate news rows worth processing (note the CAST — signal_id
# is INTEGER in signal but TEXT in news_cluster_membership).
CANONICAL_NEWS_QUERY = """
SELECT s.signal_id, s.resource_id, s.raw_text
FROM signal s
JOIN news_cluster_membership m ON m.signal_id = CAST(s.signal_id AS TEXT)
WHERE m.is_canonical = 1 AND s.signal_type = 'news_raw'
{severity_filter}
ORDER BY s.signal_id
"""

DEMAND_QUERY = """
SELECT s.signal_id, s.resource_id, s.raw_text
FROM signal s
WHERE s.signal_type = 'demand_intent_raw'
{severity_filter}
ORDER BY s.signal_id
"""


def init_extraction_schema(conn):
    conn.executescript(EXTRACTION_SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()


def _resource_names(conn):
    return {rid: name for rid, name in conn.execute("SELECT resource_id, name FROM resource")}


def _fetch(conn, query, force):
    filt = "" if force else "AND s.severity IS NULL"
    return conn.execute(query.format(severity_filter=filt)).fetchall()


def _process_row(conn, signal_id, resource_name, prompt, kind, summary, res_names):
    result = llm_client.extract_structured(prompt)
    if result is None:
        logger.warning("Extraction FAILED for %s signal_id=%s (both providers).", kind, signal_id)
        summary["failed_ids"].append(signal_id)
        return
    # Write severity/confidence AND the audit row together (consistency).
    conn.execute(
        "UPDATE signal SET severity = ?, confidence = ? WHERE signal_id = ?",
        (result["severity"], float(result["confidence"]), signal_id),
    )
    conn.execute(
        "INSERT OR REPLACE INTO event_extraction "
        "(signal_id, event_type, extracted_entities, llm_provider, llm_model, "
        "raw_llm_response, extracted_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (signal_id, result["event_type"], json.dumps(result.get("entities", [])),
         result["_provider"], result["_model"], result.get("_raw"),
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    summary["succeeded"] += 1
    summary["by_provider"][result["_provider"]] = summary["by_provider"].get(result["_provider"], 0) + 1


def run(db_path=None, force=False, max_rows=None):
    if not llm_client.available_providers():
        logger.error("No LLM provider configured: set at least one of "
                     "%s. Aborting.", [c["env"] for c in llm_client.PROVIDERS.values()])
        return 1

    budget = MAX_ROWS_PER_RUN if max_rows is None else int(max_rows)
    conn = db_init.get_connection(db_path)
    try:
        init_extraction_schema(conn)
        res_names = _resource_names(conn)

        news = _fetch(conn, CANONICAL_NEWS_QUERY, force)
        demand = _fetch(conn, DEMAND_QUERY, force)
        total_eligible = len(news) + len(demand)

        # Shared cap, news prioritized (higher-value narrative), then demand.
        news_batch = news[:budget]
        demand_batch = demand[:max(0, budget - len(news_batch))]

        summary = {"succeeded": 0, "by_provider": {}, "failed_ids": [],
                   "processed": len(news_batch) + len(demand_batch)}

        logger.info("Eligible: %d news + %d demand = %d. Cap=%d. Processing %d.",
                    len(news), len(demand), total_eligible, budget,
                    summary["processed"])

        for signal_id, resource_id, raw_text in news_batch:
            rname = res_names.get(resource_id, resource_id or "unknown")
            _process_row(conn, signal_id, rname,
                         prompts.build_news_prompt(raw_text or "", rname),
                         "news", summary, res_names)

        for signal_id, resource_id, raw_text in demand_batch:
            rname = res_names.get(resource_id, resource_id or "unknown")
            _process_row(conn, signal_id, rname,
                         prompts.build_demand_intent_prompt(raw_text or "", rname),
                         "demand", summary, res_names)

        _print_summary(summary, total_eligible)
        return 0
    finally:
        conn.close()


def _print_summary(s, total_eligible):
    by_provider = dict(sorted(s["by_provider"].items()))
    print("=" * 60)
    print(f"eligible rows        : {total_eligible}")
    print(f"processed (capped)   : {s['processed']}")
    print(f"succeeded            : {s['succeeded']}  (by provider: {by_provider})")
    print(f"failed               : {len(s['failed_ids'])}")
    if s["failed_ids"]:
        print(f"  failed signal_ids  : {s['failed_ids']}")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LLM event extraction over eligible signals.")
    ap.add_argument("--force", action="store_true",
                    help="Reprocess rows even if severity is already set (default: skip them).")
    ap.add_argument("--max", type=int, default=None, dest="max_rows",
                    help=f"Max rows to process this run (default {MAX_ROWS_PER_RUN}).")
    args = ap.parse_args()
    sys.exit(run(force=args.force, max_rows=args.max_rows))
