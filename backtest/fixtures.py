"""SYNTHETIC historical data generator for backtest-harness validation ONLY.

!!! THIS IS NOT A REAL HISTORICAL RECORD. !!!
Every row produced here is fabricated fixture data with a fixed random seed. It
exists solely to exercise the backtest harness's as-of/cutoff/isolation machinery
on historically-SHAPED data. It must never be presented, logged, or interpreted as
genuine 2023-2025 market data. Genuine historical news is unobtainable through this
project's free-tier clients (see backtest/run_backtest.py investigation note).
"""
from datetime import date, timedelta
from typing import Dict, List

import numpy as np

# The synthetic historical window we simulate (matches the DRAM curated episode era).
DEFAULT_START = date(2023, 11, 1)
DEFAULT_END = date(2025, 6, 30)
DEFAULT_SEED = 42


def generate_synthetic_history(resource_id: str, start: date = DEFAULT_START,
                               end: date = DEFAULT_END, seed: int = DEFAULT_SEED) -> Dict[str, list]:
    """Return {'prices': [(iso_date, price)], 'signals': [(iso_date, severity, raw_text)]}.

    SYNTHETIC: a gently rising price ramp (simulating a tightening episode) plus
    intermittent news signals with rising severity. Reproducible for a given seed.
    """
    rng = np.random.default_rng(seed)
    n_days = (end - start).days + 1

    prices: List[tuple] = []
    base, drift = 100.0, 0.12  # upward drift simulates tightening; NOT a real price
    for i in range(n_days):
        d = start + timedelta(days=i)
        if d.weekday() >= 5:      # skip weekends, like real market data
            continue
        value = base + drift * i + float(rng.normal(0, 2.0))
        prices.append((d.isoformat(), round(value, 4)))

    signals: List[tuple] = []
    # ~ one synthetic news signal every ~10 days, severity trending up over the window.
    for i in range(0, n_days, 10):
        d = start + timedelta(days=i)
        frac = i / max(n_days - 1, 1)
        severity = int(min(10, max(1, round(3 + 6 * frac + rng.normal(0, 0.5)))))
        signals.append((d.isoformat(), severity,
                        f"[SYNTHETIC fixture] {resource_id} tightening chatter (severity {severity})"))

    return {"prices": prices, "signals": signals}


def seed_backtest_db(conn, resource_id: str, history: Dict[str, list]) -> None:
    """Insert synthetic fixture rows into an ISOLATED backtest DB. source is marked
    'synthetic_fixture' so provenance is unmistakable in the data itself."""
    conn.execute("INSERT OR IGNORE INTO resource (resource_id, name, aliases) VALUES (?, ?, '[]')",
                 (resource_id, resource_id.upper()))
    for ts, price in history["prices"]:
        conn.execute(
            "INSERT INTO price_series (resource_id, timestamp, price_usd, unit, source) "
            "VALUES (?, ?, ?, 'USD_per_share_proxy', 'synthetic_fixture')",
            (resource_id, ts, price))
    for ts, severity, raw in history["signals"]:
        conn.execute(
            "INSERT INTO signal (signal_type, resource_id, timestamp, severity, source_name, "
            "source_url, raw_text) VALUES ('news_raw', ?, ?, ?, 'synthetic_fixture', ?, ?)",
            (resource_id, ts, severity, f"synthetic://{resource_id}/{ts}", raw))
    conn.commit()
