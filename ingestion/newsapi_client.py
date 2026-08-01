"""Thin wrapper around NewsAPI's /v2/everything endpoint (free tier).

Reads NEWSAPI_KEY from the environment. If the key is missing or the request
fails for any reason, logs a warning and returns [] — it never raises up to
the caller, so a broken/absent NewsAPI must not stop GDELT ingestion.
"""
import logging
import os
from typing import List

import requests

logger = logging.getLogger(__name__)

_ENDPOINT = "https://newsapi.org/v2/everything"
_TIMEOUT_SECONDS = 15


def search_articles(query: str, max_page_size: int = 50,
                    language: str = "en") -> List[dict]:
    """Query NewsAPI /v2/everything for `query`.

    Returns a list of dicts with keys: title, url, source_name, description,
    publishedAt, language. Returns [] if NEWSAPI_KEY is unset or on any error.

    Note: /v2/everything does not return a per-article language field, so the
    normalized `language` is set to the request language used.
    """
    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key:
        logger.warning("NEWSAPI_KEY not set — skipping NewsAPI ingestion.")
        return []

    if not query:
        return []

    page_size = max(1, min(int(max_page_size), 100))  # NewsAPI hard cap is 100
    params = {
        "q": query,
        "pageSize": page_size,
        "language": language,
        "sortBy": "publishedAt",
        "apiKey": api_key,
    }
    try:
        resp = requests.get(_ENDPOINT, params=params, timeout=_TIMEOUT_SECONDS)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # network error, timeout, non-2xx, bad JSON
        logger.warning("NewsAPI query failed for %r: %s", query, exc)
        return []

    if payload.get("status") != "ok":
        logger.warning("NewsAPI returned non-ok status for %r: %s",
                       query, payload.get("message", payload.get("status")))
        return []

    articles = []
    for item in payload.get("articles", []):
        articles.append({
            "title": item.get("title"),
            "url": item.get("url"),
            "source_name": (item.get("source") or {}).get("name"),
            "description": item.get("description"),
            "publishedAt": item.get("publishedAt"),
            "language": language,
        })
    return articles
