"""One-off historical news backfill, 2023–2025, into signal (news_raw).

WHY
---
The live news ingest (scripts/ingest_news.py) only pulls a recent window, so
signal.news_raw had no articles inside the 2023–2025 curated episode windows —
leaving the news-severity FEATURES empty exactly where the independent labels
live. This script fills that gap using GDELT's DOC API historical window
(start_date/end_date; GDELT covers 2017→present), so extraction + feature
building finally have dated news in those windows.

HOW
---
For each resource, walk MONTH BY MONTH (GDELT caps at 250 records per query, so
a narrow window gives far better coverage than one 3-year query). Reuse the exact
tagging / dedup / insert path from scripts.ingest_news, so rows are identical in
shape to live-ingested news (severity left NULL for the extraction stage to fill).

Idempotent: dedup is on (source_url, resource_id), so re-runs and overlap with
the live ingest never double-insert.

NewsAPI is intentionally NOT used here — its free tier only serves ~last 30 days,
so it cannot reach 2023–2025. GDELT is the historical source.

Run:  python scripts/ingest_historical_news.py [--start 2023-01] [--end 2025-12]
      [--per-window 250] [--sleep 1.0]
"""
import argparse
import logging
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from ingestion import gdelt_client, tagger  # noqa: E402
from scripts.ingest_news import (  # noqa: E402  (reuse the live-ingest internals)
    ALIAS_DICT_PATH, build_query_terms, ingest_batch, _existing_pairs,
    _load_resources, _normalize_gdelt,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ingest_historical_news")

DEFAULT_START = "2023-01"
DEFAULT_END = "2025-12"
DEFAULT_PER_WINDOW = 250   # GDELT hard cap
DEFAULT_SLEEP = 1.0        # be polite between GDELT queries (it rate-limits hard)


def _month_windows(start_ym: str, end_ym: str):
    """Yield (start_date, end_date) ISO strings for each month in [start, end],
    end_date being the first day of the following month (GDELT end is exclusive-ish)."""
    sy, sm = (int(x) for x in start_ym.split("-"))
    ey, em = (int(x) for x in end_ym.split("-"))
    y, m = sy, sm
    while (y, m) <= (ey, em):
        start = date(y, m, 1)
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        end = date(ny, nm, 1)
        yield start.isoformat(), end.isoformat()
        y, m = ny, nm


def run(db_path=None, start=DEFAULT_START, end=DEFAULT_END,
        per_window=DEFAULT_PER_WINDOW, sleep=DEFAULT_SLEEP):
    conn = db_init.get_connection(db_path)
    try:
        resources = _load_resources(conn)
        resource_cat = tagger.load_resource_catalog(conn)
        producer_cat = tagger.load_producer_catalog(ALIAS_DICT_PATH)
        existing_pairs = _existing_pairs(conn)
        ingested_at = datetime.now(timezone.utc).isoformat()
        windows = list(_month_windows(start, end))

        totals = {"fetched": 0, "tagged": 0, "inserted": 0, "skipped": 0}
        print("=" * 64)
        print(f"HISTORICAL NEWS BACKFILL  {start}..{end}  ({len(windows)} monthly "
              f"windows x {len(resources)} resources)  source=gdelt")
        print("=" * 64)

        for res in resources:
            terms = build_query_terms(res["name"], res["aliases"])
            res_inserted = 0
            for w_start, w_end in windows:
                articles = gdelt_client.search_articles(
                    terms, max_records=per_window, start_date=w_start, end_date=w_end)
                normalized = [_normalize_gdelt(a) for a in articles]
                tagged, inserted, skipped = ingest_batch(
                    conn, normalized, resource_cat, producer_cat,
                    existing_pairs, ingested_at)
                conn.commit()
                totals["fetched"] += len(articles)
                totals["tagged"] += tagged
                totals["inserted"] += inserted
                totals["skipped"] += skipped
                res_inserted += inserted
                if articles:
                    logger.info("[%4s %s] fetched=%3d tagged=%3d inserted=%3d skip=%3d",
                                res["resource_id"], w_start[:7], len(articles),
                                tagged, inserted, skipped)
                if sleep:
                    time.sleep(sleep)  # throttle so GDELT doesn't rate-limit us out
            print(f"[{res['resource_id']:>4}] total inserted this backfill: {res_inserted}")

        print("=" * 64)
        print(f"TOTAL fetched={totals['fetched']} tagged={totals['tagged']} "
              f"inserted={totals['inserted']} skipped_dup={totals['skipped']}")
        _report_coverage(conn)
        print("=" * 64)
    finally:
        conn.close()


def _report_coverage(conn):
    print("\nnews_raw coverage inside 2023–2025 per resource (post-backfill):")
    rows = conn.execute(
        "SELECT resource_id, COUNT(*), MIN(timestamp), MAX(timestamp) FROM signal "
        "WHERE signal_type='news_raw' AND timestamp >= '2023' AND timestamp < '2026' "
        "GROUP BY resource_id ORDER BY resource_id").fetchall()
    for rid, n, lo, hi in rows:
        print(f"  {str(rid):>4}: {n:4d} rows  {str(lo)[:10]} .. {str(hi)[:10]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Backfill 2023–2025 historical news via GDELT.")
    ap.add_argument("--start", default=DEFAULT_START, help="YYYY-MM (default 2023-01).")
    ap.add_argument("--end", default=DEFAULT_END, help="YYYY-MM inclusive (default 2025-12).")
    ap.add_argument("--per-window", type=int, default=DEFAULT_PER_WINDOW, dest="per_window",
                    help="Records per monthly query (<=250).")
    ap.add_argument("--sleep", type=float, default=DEFAULT_SLEEP,
                    help="Seconds to sleep between GDELT queries (default 1.0).")
    args = ap.parse_args()
    run(start=args.start, end=args.end, per_window=args.per_window, sleep=args.sleep)
