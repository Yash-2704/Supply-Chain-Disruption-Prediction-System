"""Orchestrate news ingestion into the `signal` table (signal_type='news_raw').

For each of the 4 seeded resources: build a query from name + top 2 aliases,
fetch from GDELT (cap 100) and NewsAPI (cap 50, skipped cleanly if no key),
run the coarse tagger, dedupe against existing (source_url, resource_id) pairs,
and insert new rows. Leaves value/unit/severity/confidence NULL by design.

Run:  python scripts/ingest_news.py
"""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from ingestion import gdelt_client, newsapi_client, tagger  # noqa: E402

ALIAS_DICT_PATH = PROJECT_ROOT / "data" / "alias_dictionary.json"

GDELT_MAX = 250          # GDELT ArtList hard cap; pull the max for volume
GDELT_TIMESPAN = "1m"    # 1-month window (broader than a single week) for coverage
NEWSAPI_MAX = 50

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ingest_news")


def build_query_terms(name: str, aliases: list) -> list:
    """name + first 2 distinct aliases, deduped case-insensitively."""
    terms, seen = [], set()
    for t in [name] + list(aliases):
        key = t.lower()
        if key not in seen:
            seen.add(key)
            terms.append(t)
        if len(terms) >= 3:  # name + 2 aliases
            break
    return terms


def _load_resources(conn):
    return [
        {"resource_id": rid, "name": name, "aliases": json.loads(aliases or "[]")}
        for rid, name, aliases in conn.execute(
            "SELECT resource_id, name, aliases FROM resource"
        )
    ]


def _existing_pairs(conn) -> set:
    """Load existing (source_url, resource_id) pairs already in signal."""
    return {
        (url, rid)
        for url, rid in conn.execute(
            "SELECT source_url, resource_id FROM signal WHERE source_url IS NOT NULL"
        )
    }


def _normalize_gdelt(a: dict) -> dict:
    return {
        "title": a.get("title"),
        "snippet": None,  # GDELT ArtList has no snippet/description field
        "url": a.get("url"),
        "source_name": a.get("domain"),
        "country": a.get("sourcecountry"),
        "published": a.get("seendate"),
        "source_tag": "gdelt",
    }


def _normalize_newsapi(a: dict) -> dict:
    return {
        "title": a.get("title"),
        "snippet": a.get("description"),
        "url": a.get("url"),
        "source_name": a.get("source_name"),
        "country": None,  # /v2/everything has no source country
        "published": a.get("publishedAt"),
        "source_tag": "newsapi",
    }


def ingest_batch(conn, articles, resource_cat, producer_cat,
                 existing_pairs, ingested_at):
    """Tag + insert a batch of normalized articles. Returns (tagged, inserted, skipped)."""
    tagged = inserted = skipped = 0
    for art in articles:
        url = art.get("url")
        if not url:
            continue
        resource_ids, producer_id = tagger.tag_article(
            art.get("title"), art.get("snippet"), resource_cat, producer_cat
        )
        if not resource_ids:
            continue
        tagged += 1
        raw_text = " ".join(p for p in (art.get("title"), art.get("snippet")) if p)
        for rid in resource_ids:
            pair = (url, rid)
            if pair in existing_pairs:
                skipped += 1
                continue
            conn.execute(
                """
                INSERT INTO signal (
                    timestamp, signal_type, resource_id, producer_id, country,
                    value, unit, severity, confidence,
                    source_name, source_url, raw_text, ingested_at
                ) VALUES (?, 'news_raw', ?, ?, ?, NULL, NULL, NULL, NULL, ?, ?, ?, ?)
                """,
                (art.get("published"), rid, producer_id, art.get("country"),
                 art.get("source_name"), url, raw_text, ingested_at),
            )
            existing_pairs.add(pair)
            inserted += 1
    return tagged, inserted, skipped


def run(db_path=None):
    conn = db_init.get_connection(db_path)
    try:
        resources = _load_resources(conn)
        resource_cat = tagger.load_resource_catalog(conn)
        producer_cat = tagger.load_producer_catalog(ALIAS_DICT_PATH)
        existing_pairs = _existing_pairs(conn)
        ingested_at = datetime.now(timezone.utc).isoformat()

        totals = {"fetched": 0, "tagged": 0, "inserted": 0, "skipped": 0}
        print("=" * 60)
        for res in resources:
            terms = build_query_terms(res["name"], res["aliases"])
            for source_tag, articles in _fetch_all(terms):
                normalizer = _normalize_gdelt if source_tag == "gdelt" else _normalize_newsapi
                normalized = [normalizer(a) for a in articles]
                tagged, inserted, skipped = ingest_batch(
                    conn, normalized, resource_cat, producer_cat,
                    existing_pairs, ingested_at
                )
                conn.commit()
                print(f"[{res['resource_id']:>4} | {source_tag:>7}] "
                      f"fetched={len(articles):3d} tagged={tagged:3d} "
                      f"inserted={inserted:3d} skipped_dup={skipped:3d}")
                totals["fetched"] += len(articles)
                totals["tagged"] += tagged
                totals["inserted"] += inserted
                totals["skipped"] += skipped
        print("=" * 60)
        print(f"TOTAL fetched={totals['fetched']} tagged={totals['tagged']} "
              f"inserted={totals['inserted']} skipped_dup={totals['skipped']}")
    finally:
        conn.close()


def _fetch_all(terms):
    """Yield (source_tag, raw_articles) for each API. GDELT first (no key needed)."""
    gdelt_query = "\"" + "\" OR \"".join(terms) + "\""  # phrase-OR for NewsAPI
    yield "gdelt", gdelt_client.search_articles(terms, max_records=GDELT_MAX, timespan=GDELT_TIMESPAN)
    yield "newsapi", newsapi_client.search_articles(gdelt_query, max_page_size=NEWSAPI_MAX)


if __name__ == "__main__":
    run()
