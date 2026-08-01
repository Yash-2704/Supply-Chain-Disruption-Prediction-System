"""Client for monthly macro commodity price context — I/O only.

Implements the WORLD BANK "Pink Sheet" (Commodity Markets) monthly historical
data file. IMF Primary Commodity Prices is intentionally NOT implemented: it is
served through a portal / SDMX flow without an equally stable direct-download
static file, so per the task's allowance we implement the one reliably
scriptable source and document the omission here.

These are broad commodity indices (energy, metals) — coarse, low-frequency
BACKGROUND context, NOT semiconductor-specific prices. Never raises upward.
"""
import io
import logging
from typing import List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

SOURCE_NAME = "world_bank_pink_sheet"

# Verified live (HTTP 200, ~765 KB .xlsx) during development. World Bank rotates
# the hash-prefixed path periodically, so a 404 here is an expected failure mode
# the caller must tolerate (returns [] + warning), not a crash.
PINK_SHEET_URL = (
    "https://thedocs.worldbank.org/en/doc/"
    "5d903e848db1d1b83e0ec8f744e55570-0350012021/related/"
    "CMO-Historical-Data-Monthly.xlsx"
)

_SHEET = "Monthly Prices"
_NAME_ROW = 4      # 0-indexed row holding commodity names
_UNIT_ROW = 5      # 0-indexed row holding units, e.g. "($/bbl)"
_DATA_START = 6    # first data row; col 0 is the period like "1960M01"
_MISSING = {"…", "..", "...", "n/a", "", "nan"}
_TIMEOUT = 30


def _period_to_date(period: str) -> Optional[str]:
    """'2024M12' -> '2024-12-01'. Returns None if unparseable."""
    try:
        year, month = str(period).strip().split("M")
        return f"{int(year):04d}-{int(month):02d}-01"
    except Exception:
        return None


def get_latest_prices(months_back: int = 3, url: str = PINK_SHEET_URL,
                      session=None) -> List[dict]:
    """Return recent monthly commodity prices as normalized dicts.

    Each dict: {date: 'YYYY-MM-01', commodity_name, price: float, unit}.
    Returns the most recent `months_back` months across all commodities.
    Returns [] (with a warning) on any download/parse failure — e.g. the WB
    URL rotating to a 404.
    """
    getter = session.get if session is not None else requests.get
    try:
        resp = getter(url, timeout=_TIMEOUT, headers={"User-Agent": "SupplyChainMVP/1.0"})
        resp.raise_for_status()
        raw = pd.read_excel(io.BytesIO(resp.content), sheet_name=_SHEET, header=None)
    except Exception as exc:
        logger.warning("macro: could not download/parse World Bank Pink Sheet "
                       "from %s: %s", url, exc)
        return []

    try:
        names = raw.iloc[_NAME_ROW]
        units = raw.iloc[_UNIT_ROW]
        data = raw.iloc[_DATA_START:]
        tail = data.tail(max(1, int(months_back)))
    except Exception as exc:
        logger.warning("macro: unexpected Pink Sheet layout: %s", exc)
        return []

    out = []
    for _, row in tail.iterrows():
        date = _period_to_date(row.iloc[0])
        if not date:
            continue
        for col in range(1, len(row)):
            name = names.iloc[col]
            if not isinstance(name, str) or not name.strip():
                continue
            value = row.iloc[col]
            if isinstance(value, str) and value.strip().lower() in _MISSING:
                continue
            try:
                price = float(value)
            except (TypeError, ValueError):
                continue
            if price != price:  # NaN
                continue
            unit = units.iloc[col]
            out.append({
                "date": date,
                "commodity_name": name.strip(),
                "price": price,
                "unit": (unit.strip() if isinstance(unit, str) else None),
            })
    if not out:
        logger.warning("macro: Pink Sheet parsed but yielded no usable rows")
    return out
