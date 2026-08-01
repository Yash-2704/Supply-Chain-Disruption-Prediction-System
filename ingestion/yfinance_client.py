"""Thin wrapper around `yfinance` for daily producer stock closes — I/O only.

A producer's stock price is a MARKET-SENTIMENT PROXY, never the price of the
resource it makes. This client only returns USD-denominated daily closes;
non-USD listings (e.g. Korean .KS / Taiwan .TW quotes in KRW/TWD) and
delisted/invalid tickers return [] with a named warning, so one bad ticker
never crashes the run. Knows nothing about the database.
"""
import logging
from typing import List

import yfinance as yf

logger = logging.getLogger(__name__)

MAX_LOOKBACK_DAYS = 90  # market-sentiment trend, not a trading feed


def get_daily_closes(ticker: str, lookback_days: int = 90,
                     require_currency: str = "USD") -> List[dict]:
    """Return [{date: 'YYYY-MM-DD', close_price: float}] for `ticker`.

    Caps lookback at MAX_LOOKBACK_DAYS. Returns [] (with a warning naming the
    ticker) for delisted/invalid tickers, empty history, or a non-USD listing
    when require_currency is set — a USD_per_share_proxy must not be populated
    from a KRW/TWD quote.
    """
    days = max(1, min(int(lookback_days), MAX_LOOKBACK_DAYS))
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=f"{days}d", interval="1d")
    except Exception as exc:
        logger.warning("yfinance: failed to fetch %s: %s", ticker, exc)
        return []
    return _parse_history(ticker, tk, hist, require_currency)


def get_daily_closes_range(ticker: str, start: str, end: str,
                           require_currency: str = "USD") -> List[dict]:
    """Return [{date, close_price}] for `ticker` over an explicit [start, end)
    date range (ISO 'YYYY-MM-DD'). This is the HISTORICAL BACKFILL path — it is
    deliberately NOT capped at MAX_LOOKBACK_DAYS (that cap only governs the daily
    sentiment-trend refresh). Same currency guard and NaN handling as
    get_daily_closes: a non-USD listing or empty history returns [] with a
    named warning, so one bad ticker never crashes a multi-year backfill.
    """
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(start=start, end=end, interval="1d")
    except Exception as exc:
        logger.warning("yfinance: failed to fetch %s [%s..%s]: %s", ticker, start, end, exc)
        return []
    return _parse_history(ticker, tk, hist, require_currency)


def _parse_history(ticker: str, tk, hist, require_currency: str) -> List[dict]:
    """Shared post-processing: reject non-USD listings, drop NaN closes, and
    return sorted [{date, close_price}] rows. Empty/invalid history -> []."""
    if hist is None or getattr(hist, "empty", True):
        logger.warning("yfinance: no price history for %s (delisted/invalid?)", ticker)
        return []

    if require_currency:
        currency = None
        try:
            currency = (tk.fast_info or {}).get("currency")
        except Exception:
            currency = None
        # Only reject when we positively know it's a different currency.
        if currency and currency != require_currency:
            logger.warning("yfinance: %s is priced in %s, not %s — skipping "
                           "(cannot label as USD proxy).", ticker, currency, require_currency)
            return []

    out = []
    for idx, row in hist.iterrows():
        close = row.get("Close")
        if close is None or close != close:  # skip NaN (non-trading artifacts)
            continue
        out.append({"date": idx.date().isoformat(), "close_price": float(close)})
    if not out:
        logger.warning("yfinance: %s returned only empty/NaN closes", ticker)
    return out
