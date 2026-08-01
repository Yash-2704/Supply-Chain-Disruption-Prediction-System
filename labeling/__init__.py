"""Label construction: hand-curated episode labels + mechanical price-derived labels.

The two sources stay provenance-tagged and may disagree; reconciliation is a
later fusion stage's job, never this one's.
"""
import pandas as pd


def week_start(date) -> str:
    """Canonical weekly grid key: the ISO-week Monday, as 'YYYY-MM-DD'.

    Both label sources snap to this same grid so curated and price-derived weeks
    align exactly for cross-source disagreement detection.
    """
    ts = pd.Timestamp(date)
    return ts.to_period("W-SUN").start_time.date().isoformat()
