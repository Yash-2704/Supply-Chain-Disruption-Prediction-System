"""Orchestrate label construction into the additive `label` table.

Writes two provenance-tagged sources — curated_episode (domain knowledge,
coverage-independent) and price_derived (mechanical anomaly rule on price_series)
— WITHOUT merging or reconciling disagreements.

Idempotency: REPLACE per (resource_id, source). Each run deletes that group's
prior rows then inserts the current set, so re-running on unchanged input yields
identical rows. The UNIQUE(resource_id, week_start, source) constraint guards
against intra-run duplicates.
Run:  python scripts/build_labels.py
"""
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from labeling import week_start, curated_labels, price_derived_labels  # noqa: E402
from forecasting.series_builder import build_price_series  # noqa: E402

LABELS_SCHEMA_PATH = PROJECT_ROOT / "schema_labels.sql"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_labels")


def init_labels_schema(conn):
    conn.executescript(LABELS_SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()


def _replace_source(conn, resource_id, source, rows, created_at):
    """Replace all rows for one (resource_id, source) with `rows`."""
    conn.execute("DELETE FROM label WHERE resource_id = ? AND source = ?",
                 (resource_id, source))
    for r in rows:
        conn.execute(
            "INSERT INTO label (resource_id, week_start, label, source, status, "
            "justification, created_at) VALUES (?,?,?,?,?,?,?)",
            (resource_id, r["week_start"], r["label"], source, r["status"],
             r.get("justification"), created_at),
        )
    conn.commit()


def _covered_weeks(conn, resource_id) -> set:
    """Set of week_start keys the live price_series actually covers for a resource."""
    df = build_price_series(resource_id, conn)
    return {week_start(d) for d in df["ds"]} if len(df) else set()


def run(db_path=None, episodes_path=None):
    conn = db_init.get_connection(db_path)
    try:
        init_labels_schema(conn)
        created_at = datetime.now(timezone.utc).isoformat()

        # --- curated (domain knowledge; coverage-independent) ---
        doc = (curated_labels.load_episodes(episodes_path) if episodes_path
               else curated_labels.load_episodes())
        curated_rows = curated_labels.expand_episodes(doc)
        by_resource = {}
        for r in curated_rows:
            by_resource.setdefault(r["resource_id"], []).append(r)
        # Replace curated rows per resource that appears in the file.
        for resource_id, rows in by_resource.items():
            _replace_source(conn, resource_id, "curated_episode", rows, created_at)
        curated_written = len(curated_rows)

        # --- price-derived (mechanical; needs real trailing history) ---
        resources = [r[0] for r in conn.execute("SELECT resource_id FROM resource ORDER BY resource_id")]
        pd_ok = pd_insuff = 0
        for resource_id in resources:
            rows = price_derived_labels.build_price_derived_for_resource(resource_id, conn)
            _replace_source(conn, resource_id, "price_derived", rows, created_at)
            pd_ok += sum(1 for r in rows if r["status"] == "ok")
            pd_insuff += sum(1 for r in rows if r["status"] == "insufficient_data")

        # --- honesty check: curated labels with zero live price coverage ---
        uncovered = 0
        covered_cache = {}
        for r in curated_rows:
            rid = r["resource_id"]
            covered = covered_cache.setdefault(rid, _covered_weeks(conn, rid))
            if r["week_start"] not in covered:
                uncovered += 1

        # --- cross-source disagreement (real query, never hardcoded) ---
        disagreements = conn.execute(
            "SELECT c.resource_id, c.week_start, c.label, p.label "
            "FROM label c JOIN label p "
            "ON c.resource_id = p.resource_id AND c.week_start = p.week_start "
            "WHERE c.source = 'curated_episode' AND p.source = 'price_derived' "
            "AND c.status = 'ok' AND p.status = 'ok' AND c.label != p.label "
            "ORDER BY c.resource_id, c.week_start"
        ).fetchall()

        _print_summary(curated_written, pd_ok, pd_insuff, uncovered, disagreements)
        return 0
    finally:
        conn.close()


def _print_summary(curated_written, pd_ok, pd_insuff, uncovered, disagreements):
    print("=" * 66)
    print("LABEL BUILD SUMMARY")
    print("-" * 66)
    print(f"curated_episode rows written        : {curated_written}")
    print(f"price_derived rows written          : {pd_ok + pd_insuff} "
          f"(ok={pd_ok}, insufficient_data={pd_insuff})")
    print(f"curated rows with ZERO live price coverage : {uncovered} "
          f"(recorded but NOT currently data-backed)")
    print(f"cross-source disagreements found     : {len(disagreements)}")
    for rid, wk, cl, pl in disagreements:
        print(f"    DISAGREE {rid} {wk}: curated={cl} price_derived={pl}")
    print("=" * 66)


if __name__ == "__main__":
    sys.exit(run())
