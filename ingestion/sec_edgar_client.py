"""Thin wrapper around `edgartools` (SEC EDGAR) — I/O + normalization only.

Sets the EDGAR identity from SEC_EDGAR_USER_AGENT_EMAIL (SEC requires a contact
email in the User-Agent). Throttles to stay under EDGAR's 10 req/sec limit.
Never raises on missing tickers, network errors, or filers with no data — logs
a warning and returns None / [] instead. Knows nothing about the database.
"""
import logging
import os
import threading
import time
from datetime import date
from typing import List, Optional

from edgar import Company, set_identity

logger = logging.getLogger(__name__)

# SEC allows a fake-but-formatted contact only for exploration; a real one is
# required for any production run. If the env var is unset we fall back loudly.
_FALLBACK_EMAIL = "REPLACE_ME_placeholder@example.com"

# Capex is not tagged identically across filers. Try these US-GAAP cash-outflow
# concepts in priority order; the caller picks the most recent across all of them.
CAPEX_CONCEPTS = (
    "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
    "us-gaap:PaymentsToAcquireProductiveAssets",
)

# Throttle: >= 0.12s between EDGAR-touching calls => <= ~8.3 req/s, safely under 10.
_MIN_INTERVAL = 0.12
_throttle_lock = threading.Lock()
_last_call_ts = 0.0

_identity_set = False


def ensure_identity() -> None:
    """Set the EDGAR identity from the environment (once). Loud fallback if unset."""
    global _identity_set
    if _identity_set:
        return
    email = os.environ.get("SEC_EDGAR_USER_AGENT_EMAIL")
    if not email:
        email = _FALLBACK_EMAIL
        logger.warning(
            "SEC_EDGAR_USER_AGENT_EMAIL is not set — using placeholder %r. "
            "This MUST be replaced with a real contact email before any real "
            "production run, or SEC may block requests.", email
        )
    set_identity(f"SupplyChain Disruption MVP {email}")
    _identity_set = True


def _throttle() -> None:
    """Block just long enough to respect the min inter-request interval."""
    global _last_call_ts
    with _throttle_lock:
        elapsed = time.monotonic() - _last_call_ts
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        _last_call_ts = time.monotonic()


def _period_key(fact):
    return (getattr(fact, "period_end", None) or date.min,
            getattr(fact, "filing_date", None) or date.min)


def get_latest_capex(ticker: str) -> Optional[dict]:
    """Most recent capital-expenditure figure for `ticker`, or None.

    Returns {value: float(USD), fiscal_period, fiscal_year, period_end,
    unit, xbrl_tag} using whichever CAPEX_CONCEPTS tag has the most recent
    non-dimensioned USD fact. None if the company/tag isn't found (e.g.
    IFRS/foreign filers that don't tag these US-GAAP concepts).
    """
    ensure_identity()
    _throttle()
    try:
        facts = Company(ticker).get_facts()
    except Exception as exc:
        logger.warning("EDGAR: could not load facts for %s: %s", ticker, exc)
        return None
    if facts is None:
        logger.info("EDGAR: no facts available for %s", ticker)
        return None

    best = best_tag = None
    for tag in CAPEX_CONCEPTS:
        try:
            rows = facts.query().by_concept(tag).execute()
        except Exception as exc:
            logger.warning("EDGAR: concept query failed for %s/%s: %s", ticker, tag, exc)
            continue
        for fact in rows or []:
            if getattr(fact, "is_dimensioned", False):
                continue
            if getattr(fact, "numeric_value", None) is None:
                continue
            if getattr(fact, "unit", None) not in (None, "USD"):
                continue
            if best is None or _period_key(fact) > _period_key(best):
                best, best_tag = fact, tag

    if best is None:
        logger.info("EDGAR: no usable capex concept for %s (likely IFRS/foreign filer)", ticker)
        return None

    return {
        "value": float(best.numeric_value),
        "fiscal_period": getattr(best, "fiscal_period", None),
        "fiscal_year": getattr(best, "fiscal_year", None),
        "period_end": best.period_end.isoformat() if getattr(best, "period_end", None) else None,
        "unit": "USD",
        "xbrl_tag": best_tag,
    }


def get_capex_history(ticker: str, start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> List[dict]:
    """ALL non-dimensioned USD capex facts for `ticker`, not just the latest.

    Same concept/unit/dimension filtering as get_latest_capex, but returns every
    qualifying fact (deduped by period_end, keeping the latest-filed value per
    period) optionally restricted to [start_date, end_date] by period_end (ISO
    'YYYY-MM-DD'). This is the HISTORICAL path — it surfaces the 2023–2025 capex
    trajectory that overlaps the curated episode windows, instead of one latest
    point. Sorted ascending by period_end. [] on any error / no data.
    """
    ensure_identity()
    _throttle()
    try:
        facts = Company(ticker).get_facts()
    except Exception as exc:
        logger.warning("EDGAR: could not load facts for %s: %s", ticker, exc)
        return []
    if facts is None:
        logger.info("EDGAR: no facts available for %s", ticker)
        return []

    by_period: dict = {}   # period_end -> (fact, tag), keeping most-recently-filed
    for tag in CAPEX_CONCEPTS:
        try:
            rows = facts.query().by_concept(tag).execute()
        except Exception as exc:
            logger.warning("EDGAR: concept query failed for %s/%s: %s", ticker, tag, exc)
            continue
        for fact in rows or []:
            if getattr(fact, "is_dimensioned", False):
                continue
            if getattr(fact, "numeric_value", None) is None:
                continue
            if getattr(fact, "unit", None) not in (None, "USD"):
                continue
            pe = getattr(fact, "period_end", None)
            if pe is None:
                continue
            pe_iso = pe.isoformat()
            if start_date and pe_iso < start_date:
                continue
            if end_date and pe_iso > end_date:
                continue
            prev = by_period.get(pe_iso)
            if prev is None or _period_key(fact) > _period_key(prev[0]):
                by_period[pe_iso] = (fact, tag)

    out = []
    for pe_iso in sorted(by_period):
        fact, tag = by_period[pe_iso]
        out.append({
            "value": float(fact.numeric_value),
            "fiscal_period": getattr(fact, "fiscal_period", None),
            "fiscal_year": getattr(fact, "fiscal_year", None),
            "period_end": pe_iso,
            "unit": "USD",
            "xbrl_tag": tag,
        })
    return out


def get_filings_in_range(ticker: str, form_type: str = "8-K",
                         start_date: str = None, end_date: str = None,
                         max_count: int = 200) -> List[dict]:
    """HISTORICAL filing METADATA for `ticker` within [start_date, end_date].

    Same normalized shape as get_recent_filings, but bounded by an explicit
    filing-date window (ISO 'YYYY-MM-DD') using EDGAR's native filing_date range
    ('start:end'), so 2023–2025 8-Ks can be pulled — not only the latest few.
    Either bound may be omitted for an open-ended range. [] on error / none.
    """
    ensure_identity()
    _throttle()
    # EDGAR range syntax: "YYYY-MM-DD:YYYY-MM-DD", or open-ended "start:" / ":end".
    if start_date or end_date:
        date_filter = f"{start_date or ''}:{end_date or ''}"
    else:
        date_filter = None
    try:
        filings = Company(ticker).get_filings(form=form_type, filing_date=date_filter) \
            if date_filter else Company(ticker).get_filings(form=form_type)
    except Exception as exc:
        logger.warning("EDGAR: could not load %s filings for %s [%s]: %s",
                       form_type, ticker, date_filter, exc)
        return []
    if filings is None or len(filings) == 0:
        logger.info("EDGAR: no %s filings for %s in %s", form_type, ticker, date_filter)
        return []

    out = []
    for i in range(min(len(filings), max_count)):
        f = filings[i]
        out.append({
            "form_type": getattr(f, "form", None),
            "filing_date": f.filing_date.isoformat() if getattr(f, "filing_date", None) else None,
            "accession_no": getattr(f, "accession_no", None),
            "url": getattr(f, "url", None) or getattr(f, "filing_url", None),
            "description": getattr(f, "primary_doc_description", None),
            "items": getattr(f, "items", None),
        })
    return out


def get_recent_filings(ticker: str, form_type: str = "8-K",
                       max_count: int = 20) -> List[dict]:
    """Recent filing METADATA for `ticker` (never document bodies).

    Returns up to `max_count` dicts: form_type, filing_date, accession_no,
    url, description, items. Returns [] on any error or no filings.
    """
    ensure_identity()
    _throttle()
    try:
        filings = Company(ticker).get_filings(form=form_type)
    except Exception as exc:
        logger.warning("EDGAR: could not load %s filings for %s: %s", form_type, ticker, exc)
        return []
    if filings is None or len(filings) == 0:
        logger.info("EDGAR: no %s filings for %s", form_type, ticker)
        return []

    out = []
    for i in range(min(len(filings), max_count)):
        f = filings[i]
        out.append({
            "form_type": getattr(f, "form", None),
            "filing_date": f.filing_date.isoformat() if getattr(f, "filing_date", None) else None,
            "accession_no": getattr(f, "accession_no", None),
            "url": getattr(f, "url", None) or getattr(f, "filing_url", None),
            "description": getattr(f, "primary_doc_description", None),
            "items": getattr(f, "items", None),  # 8-K item codes, e.g. "5.02" — metadata only
        })
    return out
