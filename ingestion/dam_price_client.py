"""Fetch REAL memory commodity prices from the Stanford DAM dataset — I/O only.

Unlike yfinance_client (which returns a producer's STOCK price, only a weak
sentiment proxy), this returns the actual USD/GB price of the memory itself —
a genuine, resource-level commodity signal for DRAM / NAND / HBM.

Source: https://dam.stanford.edu/assets/memory-prices/memory-prices.csv
Caveat (surface it, don't hide it): these are "cheapest listed RETAIL" figures
(largely Keepa-scraped), NOT contract/spot prices. A real price, but a proxy for
the contract market. Stored under its own unit/source so nothing conflates it
with the stock proxy.

Knows nothing about the database. Network/parse failure returns [] with a named
warning, so a bad fetch never crashes the pipeline.
"""
import csv
import io
import logging
from typing import List

from datetime import date

import requests

logger = logging.getLogger(__name__)

CSV_URL = "https://dam.stanford.edu/assets/memory-prices/memory-prices.csv"
_TIMEOUT = 30

# We only want the actual price metric (USD/GB) — the file also carries
# percent_share / usd_billion / usd_per_tbps rows that are NOT prices.
PRICE_METRIC = "usd_per_gb"

# Recency floor: only prices from this year onward are useful for our 2023-2025
# episodes (~3 years is the truly-useful horizon, ~5 the outer edge). The DAM CSV
# reaches back to 1957 (memory cost billions/GB then) — those ancient rows are
# irrelevant clutter and are dropped at ingest. Future-dated rows (DAM includes
# forward projections) are also dropped: a price feature must never be a forecast.
MIN_YEAR = 2021

# DAM category -> our resource_id. GPU has no memory-price analogue here (it is
# covered by the yfinance stock proxy), so it is intentionally absent.
CATEGORY_TO_RESOURCE = {"DRAM": "dram", "NAND": "nand", "HBM": "hbm"}


def _slug(series: str) -> str:
    """Compact, stable token for a series name, for the source field."""
    return "".join(ch if ch.isalnum() else "_" for ch in series.lower()).strip("_")


def get_memory_prices(url: str = CSV_URL) -> List[dict]:
    """Return [{date, resource_id, price_usd, unit, source, series}] for every
    USD/GB price row of a mapped resource (DRAM/NAND/HBM), across all dates.

    The `source` embeds the series name (e.g. 'stanford_dam:dram_cheapest_keepa')
    so distinct series for the same resource+month never collide on the
    price_series dedup key (resource_id, timestamp, source) — no real row is lost.
    """
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("stanford_dam: fetch failed (%s) — returning no rows.", exc)
        return []

    today_iso = date.today().isoformat()
    out = []
    reader = csv.DictReader(io.StringIO(resp.text))
    for r in reader:
        if r.get("metric") != PRICE_METRIC:
            continue
        resource_id = CATEGORY_TO_RESOURCE.get(r.get("category", ""))
        if resource_id is None:
            continue
        raw_val = r.get("value")
        try:
            price = float(raw_val)
        except (TypeError, ValueError):
            continue
        if price <= 0 or price != price:  # skip nonpositive / NaN
            continue
        row_date = (r.get("date") or "").strip()
        if not row_date:
            continue
        # Recency floor + no future/projection rows (see MIN_YEAR).
        if row_date[:4] < str(MIN_YEAR):
            continue
        if row_date > today_iso:
            continue
        series = (r.get("series") or "unknown").strip()
        out.append({
            "date": row_date,
            "resource_id": resource_id,
            "price_usd": price,
            "unit": "USD_per_GB",
            "source": f"stanford_dam:{_slug(series)}",
            "series": series,
        })
    if not out:
        logger.warning("stanford_dam: parsed 0 usable USD/GB price rows.")
    return out
