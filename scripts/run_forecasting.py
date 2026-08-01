"""Orchestrate forecasting for all resources x both series types.

For each (resource, series_type): build the weekly series; if it has fewer than
MIN_DATA_POINTS, record a single 'insufficient_data' row (horizon='all'); else
run Prophet + Holt-Winters across the 4 horizons, compute per-horizon
disagreement, and write one row per horizon ('ok' or 'unstable_disagreement').

Idempotency: REPLACE-LATEST. Each run deletes a (resource, series_type)'s prior
rows before inserting fresh ones stamped with run_at, so re-running replaces
rather than accumulates. One model raising is recorded as 'model_error'
(horizon='all') and never stops the loop.
Run:  python scripts/run_forecasting.py
"""
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from forecasting import series_builder, forecaster  # noqa: E402
from forecasting.forecaster import (  # noqa: E402
    HORIZONS, MIN_DATA_POINTS, DISAGREEMENT_THRESHOLD_PCT, ForecastError,
)

FORECASTING_SCHEMA_PATH = PROJECT_ROOT / "schema_forecasting.sql"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_forecasting")

SERIES_BUILDERS = {
    "price_proxy": series_builder.build_price_series,
    "demand_intent_activity": series_builder.build_demand_intent_series,
}


def init_forecasting_schema(conn):
    conn.executescript(FORECASTING_SCHEMA_PATH.read_text(encoding="utf-8"))
    # Idempotent add for DBs created before chronos_point existed (CREATE TABLE
    # IF NOT EXISTS won't add a column to an already-present table).
    cols = {row[1] for row in conn.execute("PRAGMA table_info(forecast_result)")}
    if "chronos_point" not in cols:
        conn.execute("ALTER TABLE forecast_result ADD COLUMN chronos_point REAL")
    conn.commit()


def _replace_rows(conn, resource_id, series_type, rows, run_at):
    """Replace-latest: delete prior rows for this (resource, series), insert fresh."""
    conn.execute(
        "DELETE FROM forecast_result WHERE resource_id = ? AND series_type = ?",
        (resource_id, series_type),
    )
    for r in rows:
        conn.execute(
            "INSERT INTO forecast_result (resource_id, series_type, horizon, run_at, "
            "prophet_point, prophet_lower, prophet_upper, holt_winters_point, "
            "chronos_point, disagreement_pct, status, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (resource_id, series_type, r["horizon"], run_at,
             r.get("prophet_point"), r.get("prophet_lower"), r.get("prophet_upper"),
             r.get("holt_winters_point"), r.get("chronos_point"), r.get("disagreement_pct"),
             r["status"], r.get("notes")),
        )
    conn.commit()


def _forecast_one(df):
    """Return list of per-horizon result dicts, or raise ForecastError.

    Prophet + Holt-Winters are REQUIRED (either raising fails the series). Chronos
    is OPTIONAL: it is attempted once and, if it loads/predicts, its point is
    recorded per horizon and a note flags any large Chronos-vs-Prophet divergence
    — but a Chronos failure is swallowed (chronos_point stays None) and never
    fails the run. disagreement_pct remains Prophet-vs-HW for continuity."""
    prophet = forecaster.run_prophet(df)          # may raise ForecastError
    hw = forecaster.run_holt_winters(df)          # may raise ForecastError
    try:
        chronos = forecaster.run_chronos(df)      # optional third model
    except ForecastError as exc:
        chronos = None
        logger.info("Chronos skipped for this series: %s", exc)

    rows = []
    for key in HORIZONS:
        p, h = prophet[key], hw[key]
        disagree = forecaster.compute_disagreement(p["point"], h)
        status = "unstable_disagreement" if disagree > DISAGREEMENT_THRESHOLD_PCT else "ok"
        c_point = chronos[key] if chronos else None
        note = None
        if c_point is not None:
            c_disagree = forecaster.compute_disagreement(p["point"], c_point)
            if c_disagree > DISAGREEMENT_THRESHOLD_PCT:
                note = f"chronos diverges from prophet by {c_disagree:.0f}%"
        rows.append({
            "horizon": key, "status": status,
            "prophet_point": p["point"], "prophet_lower": p["lower"],
            "prophet_upper": p["upper"], "holt_winters_point": h,
            "chronos_point": c_point,
            "disagreement_pct": round(disagree, 2),
            "notes": note,
        })
    return rows


def run(db_path=None):
    conn = db_init.get_connection(db_path)
    try:
        init_forecasting_schema(conn)
        resources = [r[0] for r in conn.execute("SELECT resource_id FROM resource ORDER BY resource_id")]
        run_at = datetime.now(timezone.utc).isoformat()
        grid = {}  # (resource, series) -> representative status/snapshot for summary

        for resource_id in resources:
            for series_type, builder in SERIES_BUILDERS.items():
                df = builder(resource_id, conn)
                n = 0 if df is None else len(df)

                if n < MIN_DATA_POINTS:
                    rows = [{"horizon": "all", "status": "insufficient_data",
                             "notes": f"{n} weekly points < MIN_DATA_POINTS={MIN_DATA_POINTS}"}]
                    _replace_rows(conn, resource_id, series_type, rows, run_at)
                    grid[(resource_id, series_type)] = ("insufficient_data", None, None)
                    logger.info("%s/%s: insufficient_data (%d pts)", resource_id, series_type, n)
                    continue

                try:
                    rows = _forecast_one(df)
                except ForecastError as exc:
                    rows = [{"horizon": "all", "status": "model_error", "notes": str(exc)[:200]}]
                    _replace_rows(conn, resource_id, series_type, rows, run_at)
                    grid[(resource_id, series_type)] = ("model_error", None, None)
                    logger.warning("%s/%s: model_error: %s", resource_id, series_type, exc)
                    continue

                _replace_rows(conn, resource_id, series_type, rows, run_at)
                one_month = next(r for r in rows if r["horizon"] == "1m")
                overall = "unstable_disagreement" if any(
                    r["status"] == "unstable_disagreement" for r in rows) else "ok"
                grid[(resource_id, series_type)] = (
                    overall, one_month["prophet_point"], one_month["disagreement_pct"])
                logger.info("%s/%s: %s (%d pts)", resource_id, series_type, overall, n)

        _print_summary(resources, grid)
        return 0
    finally:
        conn.close()


def _print_summary(resources, grid):
    print("=" * 78)
    print(f"{'resource':<8} {'series_type':<24} {'status':<22} {'1m_point':>10} {'disagr%':>8}")
    print("-" * 78)
    for resource_id in resources:
        for series_type in SERIES_BUILDERS:
            status, point, disagree = grid.get((resource_id, series_type), ("MISSING", None, None))
            pt = f"{point:.2f}" if point is not None else "-"
            dg = f"{disagree:.1f}" if disagree is not None else "-"
            print(f"{resource_id:<8} {series_type:<24} {status:<22} {pt:>10} {dg:>8}")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(run())
