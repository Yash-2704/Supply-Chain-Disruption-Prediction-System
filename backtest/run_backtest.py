"""Backtest harness: isolated DB, as-of cutoff, outcome comparison — HONEST.

Default example: DRAM, cutoff 2024-04-30, synthetic fixtures. Because genuine
2023-2025-dated NEWS is unobtainable through this project's free-tier clients
(GDELT recent-window; NewsAPI ~30-day free tier — see investigation below), this
harness validates the MECHANISM on clearly-labeled synthetic historical data. It
does NOT constitute a real historical validation of the prediction system.

Run:  python backtest/run_backtest.py            (DRAM / 2024-04-30 / synthetic)
      python backtest/run_backtest.py <resource> <cutoff_date> <real_historical|synthetic_fixture>
"""
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from backtest import cutoff as co  # noqa: E402
from backtest import fixtures as fx  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_backtest")

CURATED_EPISODES_PATH = PROJECT_ROOT / "data" / "curated_episodes.json"
DEFAULT_BACKTEST_DB = PROJECT_ROOT / "data" / "backtest_dram_2024.db"

# Which pipeline stages participate in the backtest, stated explicitly (never
# silently skipped). Printed in the summary block.
STAGES_INCLUDED = [
    "as-of price z-score / anomaly label (reuses labeling.price_derived_labels)",
    "as-of Prophet/Holt-Winters forecast direction (reuses forecasting.forecaster)",
    "outcome comparison vs curated_episodes.json known target",
]
STAGES_EXCLUDED = [
    ("live ingestion (news/SEC/market)", "fetch present-day data from APIs; no historical mode"),
    ("LLM extraction", "requires a live LLM API key and severity is not historically reconstructable; "
                       "fixtures supply synthetic severities directly"),
    ("entity resolution / clustering", "timestamp-agnostic; do not affect the cutoff-safe tightening signal"),
    ("fusion model train/predict", "reads the whole DB with NO cutoff parameter -> would leak future rows; "
                                   "also only a synthetic_validation model exists. Harness uses the "
                                   "cutoff-safe forecast/anomaly signal as its prediction proxy instead"),
]

INVESTIGATION_NOTE = (
    "Genuine 2023-2025 data availability: GDELT=NO (client is recent-window only), "
    "NewsAPI=NO (free tier ~30 days), SEC EDGAR=data exists but client returns latest-only, "
    "yfinance=data exists but client caps to 90 days. The news signal driving the DRAM story "
    "is unobtainable historically -> synthetic fixtures used.")


# ------------------------- pure outcome-comparison logic ------------------------- #

def load_episode(resource_id, path=CURATED_EPISODES_PATH):
    """Return the curated episode dict for a resource, or None."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return next((e for e in doc.get("episodes", []) if e.get("resource_id") == resource_id), None)


def expected_outcome(episode) -> str:
    """Curated episodes are, by construction, known tightening episodes."""
    return "tightening" if episode else "unknown"


def compare_to_expected_outcome(asof_features: dict, forecast_direction: str,
                                episode) -> dict:
    """Pure: does the as-of signal move in the direction the known episode implies?

    Non-overclaiming: this reports whether the harness FLAGGED a tightening signal
    from cutoff-safe data — it is NOT a claim of real predictive validation.
    """
    expected = expected_outcome(episode)
    z = asof_features.get("price_zscore_latest")
    signal_present = (
        asof_features.get("anomaly_label_latest") == 1
        or forecast_direction == "rising"
        or (z is not None and z >= 2.0))
    observed = "tightening_signal_present" if signal_present else "no_tightening_signal"
    return {
        "expected_outcome": expected,
        "observed_signal": observed,
        "direction_match": bool(signal_present and expected == "tightening"),
    }


# ------------------------------ isolation guarantee ------------------------------ #

def real_db_row_counts(real_db_path=None) -> dict:
    """READ-ONLY per-table row counts of the real database (for isolation proof)."""
    conn = db_init.get_connection(real_db_path)  # default resolves to the real supply_chain.db
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in sorted(tables)}
    finally:
        conn.close()


# ------------------------------- forecast direction ------------------------------ #

def _asof_forecast_direction(weekly) -> str:
    """'rising'|'not_rising'|'unavailable' from a cutoff-safe forecast on as-of prices."""
    from forecasting.forecaster import run_prophet, MIN_DATA_POINTS, ForecastError
    if len(weekly) < MIN_DATA_POINTS:
        return "unavailable"
    try:
        fc = run_prophet(weekly)
    except ForecastError:
        return "unavailable"
    latest = float(weekly["y"].iloc[-1])
    return "rising" if fc["1m"]["point"] > latest else "not_rising"


# --------------------------------- orchestration --------------------------------- #

def run_backtest(resource_id="dram", cutoff_date="2024-04-30",
                 data_source="synthetic_fixture",
                 backtest_db_path=DEFAULT_BACKTEST_DB, real_db_path=None,
                 seed=fx.DEFAULT_SEED):
    real_before = real_db_row_counts(real_db_path)

    if data_source == "real_historical":
        logger.error("real_historical requested but NO genuine 2023-2025 source is wired. %s",
                     INVESTIGATION_NOTE)
        return {"status": "aborted_no_real_historical_data", "data_source": data_source,
                "real_db_unchanged": True, "real_before": real_before,
                "real_after": real_before}

    # Fresh, isolated backtest DB — never the real one.
    backtest_db_path = Path(backtest_db_path)
    if backtest_db_path.exists():
        backtest_db_path.unlink()
    db_init.init_db(backtest_db_path)

    conn = db_init.get_connection(backtest_db_path)
    try:
        history = fx.generate_synthetic_history(resource_id, seed=seed)
        fx.seed_backtest_db(conn, resource_id, history)

        asof = co.compute_asof_features(conn, resource_id, cutoff_date)
        weekly_asof = co.weekly_price_series_asof(conn, resource_id, cutoff_date)
        direction = _asof_forecast_direction(weekly_asof)
        episode = load_episode(resource_id)
        result = compare_to_expected_outcome(asof, direction, episode)
    finally:
        conn.close()

    real_after = real_db_row_counts(real_db_path)
    isolated = (real_before == real_after)

    summary = {
        "status": "completed",
        "data_source": data_source,
        "resource_id": resource_id,
        "cutoff_date": cutoff_date,
        "asof_features": asof,
        "forecast_direction": direction,
        "comparison": result,
        "real_db_unchanged": isolated,
        "real_before": real_before,
        "real_after": real_after,
        "backtest_db": str(backtest_db_path),
    }
    _print_summary(summary)
    return summary


def _print_summary(s):
    print("\n" + "#" * 74)
    print("#  BACKTEST HARNESS RESULT")
    print("#" * 74)
    print(f"#  DATA SOURCE : {s['data_source'].upper()}")
    if s["data_source"] == "synthetic_fixture":
        print("#  >>> SYNTHETIC FIXTURE DATA — HARNESS-MECHANISM VALIDATION ONLY. <<<")
        print("#  >>> This is NOT a real historical validation. No claim is made that <<<")
        print("#  >>> the system 'would have caught' the real DRAM 2024-25 shortage.  <<<")
    print(f"#  resource   : {s['resource_id']}   |   as-of cutoff: {s['cutoff_date']}")
    print("#  " + "-" * 70)
    print("#  STAGES INCLUDED (cutoff-safe):")
    for st in STAGES_INCLUDED:
        print(f"#     + {st}")
    print("#  STAGES EXCLUDED (with reason):")
    for name, why in STAGES_EXCLUDED:
        print(f"#     - {name}: {why}")
    print("#  " + "-" * 70)
    af = s["asof_features"]
    print(f"#  as-of features: n_price_weeks={af['n_price_points']} "
          f"latest_price={af['latest_price']} (@{af['latest_price_date']}) "
          f"zscore={af['price_zscore_latest']} anomaly_label={af['anomaly_label_latest']}")
    print(f"#                  n_signals={af['n_signals']} mean_severity={af['mean_severity']}")
    print(f"#  as-of forecast direction: {s['forecast_direction']}")
    c = s["comparison"]
    print(f"#  expected(known episode)={c['expected_outcome']}  observed={c['observed_signal']}  "
          f"direction_match={c['direction_match']}")
    print("#     (QUALIFIED: 'direction_match' means the harness flagged a tightening")
    print("#      signal from a CONSTRUCTED synthetic scenario — not real validation.)")
    print("#  " + "-" * 70)
    print(f"#  ISOLATION: real DB unchanged = {s['real_db_unchanged']}")
    print(f"#     real DB row counts before: {s['real_before']}")
    print(f"#     real DB row counts after : {s['real_after']}")
    print(f"#  backtest DB (isolated): {s['backtest_db']}")
    print("#" * 74 + "\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    rid = args[0] if args else "dram"
    cutoff = args[1] if len(args) > 1 else "2024-04-30"
    src = args[2] if len(args) > 2 else "synthetic_fixture"
    run_backtest(rid, cutoff, src)
