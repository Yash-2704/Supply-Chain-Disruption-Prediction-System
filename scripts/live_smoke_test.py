"""OPTIONAL manual live smoke test — NOT part of the automated test suite.

This file is NEVER run by pytest/CI. It makes REAL network calls to GDELT and
(if NEWSAPI_KEY is set) NewsAPI, so a human can confirm the live endpoints are
reachable and returning sane data. It does not write to the database.

Run manually:  python scripts/live_smoke_test.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion import gdelt_client, newsapi_client  # noqa: E402


def main():
    print("=== LIVE GDELT (no key required) ===")
    gdelt = gdelt_client.search_articles(["DRAM", "Dynamic RAM"], max_records=5)
    print(f"GDELT returned {len(gdelt)} articles")
    for a in gdelt[:3]:
        print(f"  - {a['seendate']} | {a['domain']} | {a['title']}")

    print("\n=== LIVE NewsAPI (needs NEWSAPI_KEY) ===")
    news = newsapi_client.search_articles('"DRAM" OR "Dynamic RAM"', max_page_size=5)
    print(f"NewsAPI returned {len(news)} articles "
          "(0 is expected if NEWSAPI_KEY is unset)")
    for a in news[:3]:
        print(f"  - {a['publishedAt']} | {a['source_name']} | {a['title']}")


if __name__ == "__main__":
    main()
