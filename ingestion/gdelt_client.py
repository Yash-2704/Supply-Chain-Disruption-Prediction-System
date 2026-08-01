"""Thin wrapper around the `gdeltdoc` library (GDELT 2.0 DOC API, no key needed).

Fetches recent articles and normalizes them to a stable dict shape. Never
raises on network errors or empty results — logs a warning and returns [].
GDELT rate-limits queries aggressively (returns a non-JSON error the library
raises on), so transient failures are retried with backoff before giving up.
"""
import logging
import time
from typing import List

from gdeltdoc import Filters, GdeltDoc

logger = logging.getLogger(__name__)

# GDELT ArtList caps at 250 records per query; we stay well under free-use norms.
_MAX_ALLOWED = 250

# Fields we expose downstream, sourced from the GDELT ArtList response columns.
_FIELDS = ("title", "url", "domain", "seendate", "sourcecountry", "language")

_DEFAULT_RETRIES = 3   # GDELT frequently rate-limits; retry transient failures
_BACKOFF_SECONDS = 2.0


def search_articles(query_terms: List[str], max_records: int = 250,
                    timespan: str = "1m", max_retries: int = _DEFAULT_RETRIES,
                    start_date: str = None, end_date: str = None) -> List[dict]:
    """Query GDELT for articles matching any of `query_terms`.

    Params
    ------
    query_terms : list of keyword strings (OR-combined by GDELT)
    max_records : cap on records returned (clamped to GDELT's 250 max)
    timespan    : GDELT timespan expression (e.g. "1w", "1m", "3m"); default "1m"
                  for broader coverage than a single week. IGNORED when an explicit
                  start_date/end_date window is given.
    max_retries : retries on a transient GDELT failure (rate-limit), with backoff
    start_date,
    end_date    : optional ISO 'YYYY-MM-DD' bounds for a HISTORICAL window (GDELT
                  DOC covers 2017→present). When both are provided they replace
                  `timespan`, enabling 2023–2025 backfill of dated news. The 250
                  per-query cap still applies, so callers should window narrowly
                  (e.g. month by month) for coverage.

    Returns a list of dicts with keys: title, url, domain, seendate,
    sourcecountry, language. Returns [] on persistent error or genuine empty.
    """
    if not query_terms:
        return []

    num_records = max(1, min(int(max_records), _MAX_ALLOWED))
    if start_date and end_date:
        filters = Filters(keyword=list(query_terms), start_date=start_date,
                          end_date=end_date, num_records=num_records)
    else:
        filters = Filters(keyword=list(query_terms), timespan=timespan, num_records=num_records)

    df = None
    for attempt in range(1, max_retries + 1):
        try:
            df = GdeltDoc().article_search(filters)
            break  # got a (possibly empty) DataFrame — no exception
        except Exception as exc:  # network error, rate-limit non-JSON, library error
            if attempt < max_retries:
                wait = _BACKOFF_SECONDS * attempt
                logger.warning("GDELT query failed for %s (attempt %d/%d): %s — retrying in %.0fs",
                               query_terms, attempt, max_retries, exc, wait)
                time.sleep(wait)
            else:
                logger.warning("GDELT query failed for %s after %d attempts: %s",
                               query_terms, max_retries, exc)
                return []

    if df is None or df.empty:
        logger.info("GDELT returned no articles for %s", query_terms)
        return []

    articles = []
    for record in df.to_dict(orient="records"):
        articles.append({field: _clean(record.get(field)) for field in _FIELDS})
    return articles


def _clean(value):
    """Coerce pandas NaN / missing values to None, everything else to str."""
    if value is None:
        return None
    # pandas NaN is a float that is not equal to itself.
    if isinstance(value, float) and value != value:
        return None
    text = str(value).strip()
    return text or None
