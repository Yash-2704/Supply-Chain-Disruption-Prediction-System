"""End-to-end integration check against a FRESH TEMPORARY COPY of the real DB.

Each orchestration stage is run as its OWN SUBPROCESS with SUPPLY_CHAIN_DB pointed
at the copy — exactly how the pipeline is really invoked (`python scripts/X.py`).
This is deliberate: chaining every heavy native stage (torch via sentence-
transformers + LightGBM) into ONE long-lived Python process segfaults due to an
OpenMP runtime conflict (documented as a finding); separate processes are the
supported execution model and are what this check exercises.

Isolation: the real DB (md5), the real Chroma store, and the real models/ dir are
all snapshotted before and restored/verified after — nothing real is mutated.
Run:  python verification/integration_check.py
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402  (sqlite only — main process never imports torch/lightgbm)

REAL_DB = Path(db_init.get_db_path())
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
MODELS_DIR = PROJECT_ROOT / "models"
PY = str(PROJECT_ROOT / ".venv" / "bin" / "python")


def _md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _run(argv, env):
    p = subprocess.run(argv, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)


def _copy_conn(copy_db):
    return db_init.get_connection(copy_db)


def main():
    results = []
    real_before = _md5(REAL_DB)
    print(f"REAL DB: {REAL_DB}\n  md5 BEFORE = {real_before}\n")

    tmp = Path(tempfile.mkdtemp(prefix="integration_"))
    copy_db = tmp / "copy.db"
    shutil.copy2(REAL_DB, copy_db)

    # Snapshot real non-DB artifacts so stages can run without mutating them.
    chroma_bak = tmp / "chroma_bak"
    models_bak = tmp / "models_bak"
    if CHROMA_DIR.exists():
        shutil.copytree(CHROMA_DIR, chroma_bak)
    if MODELS_DIR.exists():
        shutil.copytree(MODELS_DIR, models_bak)

    env = {**os.environ, "SUPPLY_CHAIN_DB": str(copy_db)}

    try:
        # 1) Entity resolution: run twice (subprocess), confirm idempotent.
        er_log = tmp / "er.jsonl"
        rc1, out1 = _run([PY, "-c",
            f"from entity_resolution.reprocess_signals import reprocess; "
            f"reprocess({str(copy_db)!r}, log_path={str(er_log)!r})"], env)
        conn = _copy_conn(copy_db)
        snap1 = conn.execute("SELECT signal_id, producer_id FROM signal "
                             "WHERE signal_type='news_raw' ORDER BY signal_id").fetchall()
        conn.close()
        rc2, out2 = _run([PY, "-c",
            f"from entity_resolution.reprocess_signals import reprocess; "
            f"reprocess({str(copy_db)!r}, log_path={str(tmp / 'er2.jsonl')!r})"], env)
        conn = _copy_conn(copy_db)
        snap2 = conn.execute("SELECT signal_id, producer_id FROM signal "
                             "WHERE signal_type='news_raw' ORDER BY signal_id").fetchall()
        conn.close()
        ok = (rc1 == 0 and rc2 == 0 and snap1 == snap2)
        results.append(("entity_resolution reprocess + idempotent", ok,
                        f"rc1={rc1} rc2={rc2}; {len(snap1)} news_raw rows; 2nd-run-identical={snap1==snap2}"))

        # 2) Clustering: membership == news_raw count.
        rc, out = _run([PY, "scripts/run_clustering.py"], env)
        conn = _copy_conn(copy_db)
        n_news = conn.execute("SELECT COUNT(*) FROM signal WHERE signal_type='news_raw'").fetchone()[0]
        n_memb = conn.execute("SELECT COUNT(*) FROM news_cluster_membership").fetchone()[0]
        conn.close()
        ok = (rc == 0 and n_news == n_memb)
        results.append(("clustering membership == news_raw count", ok,
                        f"rc={rc}; news_raw={n_news}, membership={n_memb}"))

        # 3) Forecasting: 4 resources x 2 series types, valid statuses only.
        rc, out = _run([PY, "scripts/run_forecasting.py"], env)
        conn = _copy_conn(copy_db)
        combos = conn.execute("SELECT DISTINCT resource_id, series_type FROM forecast_result").fetchall()
        statuses = {r[0] for r in conn.execute("SELECT DISTINCT status FROM forecast_result")}
        conn.close()
        valid = {"ok", "insufficient_data", "model_error", "unstable_disagreement"}
        ok = (rc == 0 and len(combos) == 8 and statuses.issubset(valid))
        results.append(("forecasting 4x2 combos + valid statuses", ok,
                        f"rc={rc}; {len(combos)} combos; statuses={sorted(statuses)}"))

        # 4) Labels: unique (resource,week,source) holds; rows present.
        rc, out = _run([PY, "scripts/build_labels.py"], env)
        conn = _copy_conn(copy_db)
        total = conn.execute("SELECT COUNT(*) FROM label").fetchone()[0]
        dupes = conn.execute("SELECT COUNT(*) FROM (SELECT resource_id, week_start, source FROM label "
                             "GROUP BY resource_id, week_start, source HAVING COUNT(*)>1)").fetchone()[0]
        curated = conn.execute("SELECT COUNT(*) FROM label WHERE source='curated_episode'").fetchone()[0]
        conn.close()
        ok = (rc == 0 and dupes == 0 and total > 0 and curated > 0)
        results.append(("labels unique-constraint holds + rows present", ok,
                        f"rc={rc}; total={total}, curated={curated}, duplicate_keys={dupes}"))

        # 5) Fusion training: report actual data_mode produced.
        rc, out = _run([PY, "modeling/train_fusion_model.py"], env)
        conn = _copy_conn(copy_db)
        row = conn.execute("SELECT data_mode, n_examples FROM training_run "
                           "ORDER BY run_id DESC LIMIT 1").fetchone()
        conn.close()
        ok = (rc == 0 and row is not None)
        results.append(("fusion training runs (mode reported honestly)", ok,
                        f"rc={rc}; latest training_run data_mode={row[0]!r} n_examples={row[1]}"))

        # 6) Dashboard leaderboard: one entry per resource, no exception (pure-python, in-process).
        from dashboard_data.queries import get_leaderboard
        conn = _copy_conn(copy_db)
        board = get_leaderboard(conn=conn)
        n_res = conn.execute("SELECT COUNT(*) FROM resource").fetchone()[0]
        conn.close()
        ok = (len(board) == n_res)
        results.append(("dashboard get_leaderboard one-per-resource", ok,
                        f"{len(board)} leaderboard rows for {n_res} resources"))
    finally:
        # Restore real artifacts that stages may have written to.
        if chroma_bak.exists():
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)
            shutil.copytree(chroma_bak, CHROMA_DIR)
        if models_bak.exists():
            shutil.rmtree(MODELS_DIR, ignore_errors=True)
            shutil.copytree(models_bak, MODELS_DIR)

    real_after = _md5(REAL_DB)
    unchanged = (real_before == real_after)

    print("=== INTEGRATION CHECK RESULTS (each stage in its own subprocess) ===")
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")
    print("\n=== ISOLATION ===")
    print(f"  real DB md5 AFTER = {real_after}")
    print(f"  [{'PASS' if unchanged else 'FAIL'}] real database UNCHANGED (md5 identical)")

    shutil.rmtree(tmp, ignore_errors=True)
    all_ok = unchanged and all(ok for _, ok, _ in results)
    print(f"\nOVERALL: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED — see above'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
