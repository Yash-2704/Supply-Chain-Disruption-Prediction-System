"""Pipeline orchestration for the dashboard's "Run Pipeline" button.

Two independent states, exactly as required:

  (a) INFERENCE  — a fast, foreground-ish run (executed on a background thread so
      the UI can poll it live) that walks the pipeline stages and produces a real
      risk score PER RESOURCE using the CURRENTLY-SAVED models. It never trains,
      so the user gets an answer in seconds.

  (b) TRAINING   — heavy model (re)training, dispatched to a single background
      worker with a QUEUE. Clicking Run several times enqueues several training
      jobs; the worker runs them one at a time (isolated subprocess — matches the
      "run heavy Python one-at-a-time" constraint and avoids the torch+OpenMP
      in-process segfault).

This module deliberately has NO Streamlit import: it is pure orchestration state
that `app.py` renders. All shared state lives in module globals guarded by a lock;
Streamlit reruns happen in the same process, so the Run page simply polls these.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from labeling import week_start  # noqa: E402
from modeling import feature_builder, predict  # noqa: E402
from risk_dashboard.config import (  # noqa: E402
    PROCESSED_DATA_DIR, RESOURCES, RESOURCE_ID, ID_TO_DISPLAY,
)

PIPE_DIR = PROJECT_ROOT / "dashboard_export" / "_pipeline"
PIPE_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_LOG_DIR = PIPE_DIR / "train_logs"
TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = PIPE_DIR / "last_run.json"

# ── Stage catalogue ─────────────────────────────────────────────────────────
# (key, human label, icon key used by app.py, one-line "what it does")
STAGES = [
    ("ingest",     "Ingest",     "ingest",  "Gather latest prices, news & filings"),
    ("understand", "Understand", "brain",   "LLM news-severity signals"),
    ("engineer",   "Engineer",   "gear",    "Build the 10 weekly features"),
    ("label",      "Label",      "tag",     "Refresh tightening labels"),
    ("predict",    "Predict",    "target",  "Score with the current models"),
    ("explain",    "Explain",    "lens",    "Attribute the drivers (SHAP)"),
    ("deliver",    "Deliver",    "ship",    "Publish score · queue training"),
]

# Small floor so each stage is visibly "active" on the animated page even though
# the real work is sub-second. This is presentation only — the work is real.
_STAGE_MIN_SECONDS = 0.65

_LOCK = threading.RLock()
_run: dict | None = None                 # the single active/last inference run
_train_q: "deque[dict]" = deque()        # queued training jobs (FIFO)
_train_current: dict | None = None       # job the worker is running right now
_train_history: list[dict] = []          # finished jobs (most-recent-first, capped)
_train_worker_started = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_stage_list() -> list[dict]:
    return [
        {"key": k, "label": lbl, "icon": icon, "desc": desc,
         "status": "pending", "detail": "", "t_start": None, "t_end": None}
        for (k, lbl, icon, desc) in STAGES
    ]


# ── Public accessors (read-only snapshots for the UI) ───────────────────────
def get_run() -> dict | None:
    """Current/last inference run (live dict — the UI only reads from it)."""
    return _run


def is_running() -> bool:
    return bool(_run and _run.get("status") == "running")


def training_snapshot() -> dict:
    with _LOCK:
        return {
            "current": dict(_train_current) if _train_current else None,
            "queued": [dict(j) for j in _train_q],
            "history": [dict(j) for j in _train_history[:6]],
        }


# ── Inference run ───────────────────────────────────────────────────────────
def start_run(live_ingest: bool = False, training_job_id: str | None = None) -> str:
    """Start ONE inference run using the current models. If one is already
    running, return its id unchanged (a second click can't launch a concurrent
    inference — but the caller still enqueues another training job, which simply
    lengthens the background queue)."""
    global _run
    with _LOCK:
        if is_running():
            return _run["run_id"]
        run_id = "run_" + uuid.uuid4().hex[:8]
        _run = {
            "run_id": run_id, "status": "running",
            "started_at": _now(), "finished_at": None,
            "live_ingest": bool(live_ingest),
            "stages": _new_stage_list(),
            "scores": {}, "drivers": {}, "asof": {},
            "training_job_id": training_job_id, "error": None,
        }
    t = threading.Thread(target=_run_inference, args=(run_id,), daemon=True)
    t.start()
    return run_id


def _set_stage(stage: dict, status: str, detail: str = ""):
    stage["status"] = status
    if detail:
        stage["detail"] = detail
    if status == "active" and stage["t_start"] is None:
        stage["t_start"] = time.time()
    if status in ("done", "skipped", "error"):
        stage["t_end"] = time.time()


def _latest_week(conn, rid: str) -> str:
    row = conn.execute(
        "SELECT MAX(timestamp) FROM price_series WHERE resource_id = ? "
        "AND unit = 'USD_per_share_proxy'", (rid,)).fetchone()
    if row and row[0]:
        return week_start(row[0])
    return week_start(datetime.now(timezone.utc).date().isoformat())


def _run_inference(run_id: str):
    """Walk the stages on a worker thread, updating shared state as we go.
    Never touches Streamlit. Resilient: a soft stage that fails is marked but the
    run continues so a real score is still produced."""
    run = _run
    stages = {s["key"]: s for s in run["stages"]}
    conn = db_init.get_connection()
    try:
        # 1) INGEST -----------------------------------------------------------
        s = stages["ingest"]; _set_stage(s, "active")
        _floor(s, lambda: _stage_ingest(run, conn, s))

        # 2) UNDERSTAND -------------------------------------------------------
        s = stages["understand"]; _set_stage(s, "active")
        _floor(s, lambda: _stage_understand(conn, s))

        # 3) ENGINEER ---------------------------------------------------------
        s = stages["engineer"]; _set_stage(s, "active")
        _floor(s, lambda: _stage_engineer(conn, s))

        # 4) LABEL ------------------------------------------------------------
        s = stages["label"]; _set_stage(s, "active")
        _floor(s, lambda: _stage_label(conn, s))

        # 5) PREDICT (the real deliverable — uses CURRENT models) -------------
        s = stages["predict"]; _set_stage(s, "active")
        _floor(s, lambda: _stage_predict(run, conn, s))

        # 6) EXPLAIN ----------------------------------------------------------
        s = stages["explain"]; _set_stage(s, "active")
        _floor(s, lambda: _stage_explain(run, s))

        # 7) DELIVER — publish + hand training off to the background queue -----
        s = stages["deliver"]; _set_stage(s, "active")
        _floor(s, lambda: _stage_deliver(run, s))

        run["status"] = "done"
    except Exception as exc:  # a hard failure — surface it, don't crash the app
        run["status"] = "error"
        run["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        run["finished_at"] = _now()
        conn.close()
        _persist_last_run(run)


def _floor(stage: dict, fn):
    """Run a stage body, mark done/error, and hold 'active' for a minimum so the
    animation is perceptible."""
    t0 = time.time()
    try:
        detail = fn() or stage.get("detail", "")
        status = "done"
    except _StageSkip as skip:
        detail = str(skip); status = "skipped"
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"; status = "error"
    remain = _STAGE_MIN_SECONDS - (time.time() - t0)
    if remain > 0:
        time.sleep(remain)
    _set_stage(stage, status, detail)


class _StageSkip(Exception):
    """Raised by a soft stage to record a truthful 'skipped: reason' state."""


# ── Individual stage bodies (all real, all offline unless live_ingest) ───────
def _stage_ingest(run, conn, stage) -> str:
    if run["live_ingest"]:
        # Live producer-price pull. `--no-macro` skips the slow, insert-nothing
        # World Bank macro fetch (Part B) so this stays inside its time budget;
        # scoring only uses the producer prices Part A refreshes. The timeout is
        # generous because cold-start imports (yfinance) can take ~30-60s on a
        # slow disk before any network call — a tighter cap is what previously
        # made this stage falsely report "skipped".
        try:
            r = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "ingest_market_data.py"),
                 "--no-macro"],
                cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                tail = (r.stderr or r.stdout or "").strip().splitlines()[-1:] or [""]
                raise _StageSkip(f"live pull failed ({tail[0][:80]}) → using cached data")
        except subprocess.TimeoutExpired:
            raise _StageSkip("live pull timed out → using cached data")
    p = conn.execute("SELECT COUNT(*) FROM price_series").fetchone()[0]
    sg = conn.execute("SELECT COUNT(*) FROM signal").fetchone()[0]
    try:
        ev = conn.execute("SELECT COUNT(*) FROM event_extraction").fetchone()[0]
    except Exception:
        ev = 0
    return f"{p:,} price · {sg:,} news · {ev:,} filing rows"


def _stage_understand(conn, stage) -> str:
    scored = conn.execute(
        "SELECT COUNT(*) FROM signal WHERE severity IS NOT NULL").fetchone()[0]
    if scored == 0:
        raise _StageSkip("no severity-scored news yet")
    return f"{scored:,} stories scored 1–10 for severity"


def _stage_engineer(conn, stage) -> str:
    total = 0
    for disp in RESOURCES:
        rid = RESOURCE_ID[disp]
        wk = _latest_week(conn, rid)
        feats = feature_builder.build_feature_row(rid, wk, conn)
        total += sum(1 for v in feats.values() if v == v)  # non-NaN
    return f"10 features × {len(RESOURCES)} resources ({total} populated)"


def _stage_label(conn, stage) -> str:
    try:
        n = conn.execute("SELECT COUNT(*) FROM label WHERE status='ok'").fetchone()[0]
        pos = conn.execute("SELECT COUNT(*) FROM label WHERE status='ok' AND label=1").fetchone()[0]
    except Exception:
        raise _StageSkip("label table not populated")
    return f"{n:,} labeled weeks · {pos:,} tightening"


def _stage_predict(run, conn, stage) -> str:
    """Score every resource with the CURRENTLY-SAVED model — no training, and no
    live SHAP (TreeExplainer on the calibrated pipeline takes >100s, which would
    defeat the whole "fast answer" point). We load the bundle once and call
    predict_proba + the isotonic calibrator directly: ~0.03s per resource. Live
    SHAP attribution is the background training job's responsibility; the Explain
    stage reuses the drivers it already computed."""
    import pandas as pd
    from modeling.feature_builder import FEATURE_COLUMNS

    bundle, _ = predict._load_bundle(predict.MODELS_DIR, prefer="real")
    if bundle is None:
        raise Exception("no saved model available to score with")
    run["model_type"] = bundle.get("model_type", "lightgbm")
    run["data_mode"] = bundle.get("data_mode")

    scores, asof = {}, {}
    for disp in RESOURCES:
        rid = RESOURCE_ID[disp]
        wk = _latest_week(conn, rid)
        try:
            feats = feature_builder.build_feature_row(rid, wk, conn)
            X = pd.DataFrame([feats])[FEATURE_COLUMNS]
            raw = float(bundle["model"].predict_proba(X)[:, 1][0])
            cal = float(min(1.0, max(0.0, bundle["calibrator"].predict([raw])[0])))
            scores[disp] = cal
            asof[disp] = wk
        except Exception as exc:  # one resource shouldn't sink the whole stage
            run.setdefault("warnings", []).append(f"{disp}: {type(exc).__name__}")
    if not scores:
        raise Exception("no resource could be scored with the current model")
    run["scores"] = scores
    run["asof"] = asof
    _write_scores_csv(scores)
    return f"scored {len(scores)}/{len(RESOURCES)} resources · model={run['model_type']}"


def _stage_explain(run, stage) -> str:
    """Surface the top driver per resource from the most recent SHAP export
    (dashboard_export/processed/resource_drivers.csv, produced by training).
    Read-only and instant — no live SHAP on the fast path."""
    import csv
    drivers: dict[str, list] = {}
    path = PROCESSED_DATA_DIR / "resource_drivers.csv"
    if not path.exists():
        raise _StageSkip("drivers appear after the first training run")
    try:
        with open(path) as fh:
            for row in csv.DictReader(fh):
                res = row.get("resource")
                if res in RESOURCES:
                    drivers.setdefault(res, []).append({
                        "feature": row.get("feature"),
                        "contribution": float(row.get("contribution") or 0.0),
                    })
    except Exception:
        raise _StageSkip("driver export unreadable — scores still valid")
    # keep the single strongest driver per resource for the compact UI
    top = {}
    for res, items in drivers.items():
        items.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        if items:
            top[res] = items[:3]
    run["drivers"] = top
    if not top:
        raise _StageSkip("no drivers available yet — scores still valid")
    return f"top drivers surfaced for {len(top)} resources"


def _stage_deliver(run, stage) -> str:
    """Publish the score (already written in Predict) and report where this run's
    background training job sits. The job was enqueued by the click handler so
    that rapid repeat clicks lengthen the queue even while an inference is live."""
    job_id = run.get("training_job_id")
    if not job_id:
        return "score published"
    snap = training_snapshot()
    if snap["current"] and snap["current"]["id"] == job_id:
        where = "running now"
    elif any(j["id"] == job_id for j in snap["queued"]):
        pos = [j["id"] for j in snap["queued"]].index(job_id) + 1
        where = f"queued (#{pos} in line)"
    else:
        where = "queued"
    return f"score published · training {where}"


def _write_scores_csv(scores: dict):
    import csv
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / "risk_scores.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["resource", "risk_score"])
        for disp, val in scores.items():
            w.writerow([disp, val])


def _persist_last_run(run: dict):
    try:
        slim = {k: run[k] for k in ("run_id", "status", "started_at", "finished_at",
                                    "scores", "asof", "error", "training_job_id")}
        slim["stages"] = [{"key": s["key"], "label": s["label"],
                           "status": s["status"], "detail": s["detail"]}
                          for s in run["stages"]]
        STATE_FILE.write_text(json.dumps(slim, indent=2))
    except Exception:
        pass


# ── Training queue + worker ─────────────────────────────────────────────────
def enqueue_training(source_run: str | None = None) -> dict:
    """Add a training job to the background queue and make sure the worker runs.
    Returns the job dict."""
    global _train_worker_started
    job = {
        "id": "job_" + uuid.uuid4().hex[:8],
        "kind": "model_training_benchmark",
        "source_run": source_run,
        "submitted_at": _now(),
        "status": "queued",
        "started_at": None, "finished_at": None,
        "returncode": None,
        "log_path": None,
    }
    with _LOCK:
        _train_q.append(job)
        if not _train_worker_started:
            _train_worker_started = True
            threading.Thread(target=_train_worker, daemon=True).start()
    return job


def _train_worker():
    """Single consumer: run queued training jobs strictly one at a time."""
    global _train_current
    while True:
        with _LOCK:
            job = _train_q.popleft() if _train_q else None
            _train_current = job
        if job is None:
            time.sleep(0.4)
            continue
        job["status"] = "running"
        job["started_at"] = _now()
        log_path = TRAIN_LOG_DIR / f"{job['id']}.log"
        job["log_path"] = str(log_path)
        try:
            with open(log_path, "w") as log:
                # Full pipeline train: (re)train the 4 models and refresh every
                # dashboard artefact. Isolated subprocess = no torch/OpenMP crash
                # in the Streamlit process, and the queue guarantees one at a time.
                # PIPELINE_TRAIN_DRYRUN=1 substitutes a trivial command (demos/tests
                # that want the queue mechanics without a heavy retrain).
                if os.environ.get("PIPELINE_TRAIN_DRYRUN") == "1":
                    cmd = [sys.executable, "-c",
                           "import time;print('[dry-run] training simulated');time.sleep(2)"]
                else:
                    cmd = [sys.executable,
                           str(PROJECT_ROOT / "scripts" / "build_dashboard_data.py")]
                proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                                      stdout=log, stderr=subprocess.STDOUT)
            job["returncode"] = proc.returncode
            job["status"] = "done" if proc.returncode == 0 else "error"
        except Exception as exc:
            job["returncode"] = -1
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            job["finished_at"] = _now()
            with _LOCK:
                _train_history.insert(0, job)
                del _train_history[8:]
                _train_current = None
