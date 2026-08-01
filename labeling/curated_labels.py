"""Expand hand-curated episodes into weekly label=1 rows.

Pure domain knowledge: this NEVER consults price_series. A curated label is real
regardless of whether the database currently has feature data covering that week
(that coverage gap is reported separately by the orchestrator).
"""
import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import List

import pandas as pd

from . import week_start

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURATED_EPISODES_PATH = PROJECT_ROOT / "data" / "curated_episodes.json"


def load_episodes(path=CURATED_EPISODES_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def expand_episodes(doc: dict) -> List[dict]:
    """Expand curated periods into weekly label rows (source='curated_episode').

    Two kinds of period, both real domain knowledge:
      * `episodes`         -> label=1 (TIGHTENING), spanning start_date - lead_weeks
                              through end_date (the lead captures the pre-tightening
                              run-up in signals).
      * `negative_periods` -> label=0 (NON-tightening / oversupply), spanning exactly
                              start_date through end_date with NO lead.

    Positives are emitted first and share the dedup guard, so if a negative period
    ever overlaps a positive episode on the same resource the positive wins (a week
    is never emitted as both classes, honoring the UNIQUE(resource,week,source)
    constraint).

    Returns dicts: {resource_id, week_start, label, source, status, justification}.
    """
    lead_weeks = int(doc.get("lead_weeks", 0))
    rows, seen = [], set()

    def _emit(period, label, lead):
        start = pd.Timestamp(period["start_date"]) - timedelta(weeks=lead)
        end = pd.Timestamp(period["end_date"])
        first_monday = pd.Timestamp(week_start(start))
        for ws in pd.date_range(start=first_monday, end=end, freq="W-MON"):
            wk = ws.date().isoformat()
            key = (period["resource_id"], wk)
            if key in seen:  # overlapping period on one resource — first (positive) wins
                continue
            seen.add(key)
            rows.append({
                "resource_id": period["resource_id"],
                "week_start": wk,
                "label": label,
                "source": "curated_episode",
                "status": "ok",
                "justification": period["justification"],
            })

    for ep in doc.get("episodes", []):            # label=1, with lead-in
        _emit(ep, 1, lead_weeks)
    for neg in doc.get("negative_periods", []):   # label=0, no lead
        _emit(neg, 0, 0)
    return rows
