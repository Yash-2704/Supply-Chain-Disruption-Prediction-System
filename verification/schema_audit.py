"""Audit the REAL data/supply_chain.db: every table's columns + row counts, plus
the specific 'known contentious' columns each prior NOTE was about.

Read-only. Prints freshly-observed values so they can be compared to prior claims.
Run:  python verification/schema_audit.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402


def audit():
    conn = db_init.get_connection()  # the real DB (SUPPLY_CHAIN_DB default)
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        print(f"=== {len(tables)} tables in {db_init.get_db_path()} ===")
        for t in tables:
            cols = [(r[1], r[2], "NOT NULL" if r[3] else "nullable")
                    for r in conn.execute(f"PRAGMA table_info({t})")]
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"\n{t}  (rows={n})")
            for name, ctype, nn in cols:
                print(f"    {name:<24} {ctype:<8} {nn}")

        print("\n=== KNOWN CONTENTIOUS COLUMNS (resolving each prior NOTE) ===")

        # NOTE 4 — price_series.resource_id nullability fix
        nn = [r[3] for r in conn.execute("PRAGMA table_info(price_series)")
              if r[1] == "resource_id"][0]
        print(f"[NOTE 4] price_series.resource_id NOT NULL = {bool(nn)}  "
              f"=> nullability fix applied? {not nn}")

        # NOTE 7 — signal.severity/confidence population
        sev = conn.execute("SELECT COUNT(*) FROM signal WHERE severity IS NOT NULL").fetchone()[0]
        conf = conn.execute("SELECT COUNT(*) FROM signal WHERE confidence IS NOT NULL").fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM signal").fetchone()[0]
        print(f"[NOTE 7] signal.severity non-null = {sev}/{total}; confidence non-null = {conf}")
        ee = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='event_extraction'").fetchone()
        print(f"[NOTE 7] event_extraction table exists = {bool(ee)}  "
              "(created lazily only when run_extraction.py actually runs)")

        # NOTE 10 — training_run.data_mode
        modes = conn.execute(
            "SELECT data_mode, COUNT(*), MAX(n_examples) FROM training_run GROUP BY data_mode").fetchall()
        print(f"[NOTE 10] training_run data_modes (mode, count, max_n_examples) = {modes}")

        # label / explanation distinct sets
        print("[label] sources =",
              conn.execute("SELECT source, COUNT(*) FROM label GROUP BY source").fetchall(),
              "| statuses =",
              conn.execute("SELECT DISTINCT status FROM label").fetchall())
        print("[label] nand curated_episode rows (expected 0) =",
              conn.execute("SELECT COUNT(*) FROM label WHERE resource_id='nand' "
                           "AND source='curated_episode'").fetchone()[0])
        print("[explanation] statuses =",
              conn.execute("SELECT status, COUNT(*) FROM explanation GROUP BY status").fetchall())
        print("[explanation] is_real_data_model values =",
              conn.execute("SELECT DISTINCT is_real_data_model FROM explanation").fetchall())
        print("[signal] signal_type breakdown =",
              conn.execute("SELECT signal_type, COUNT(*) FROM signal GROUP BY signal_type").fetchall())
        print("[forecast_result] statuses =",
              conn.execute("SELECT status, COUNT(*) FROM forecast_result GROUP BY status").fetchall())
    finally:
        conn.close()


if __name__ == "__main__":
    audit()
