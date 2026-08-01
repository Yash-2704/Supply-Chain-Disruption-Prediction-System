"""Reprocess `news_raw` producer attribution with three-layer resolution.

Scope is strictly `signal_type = 'news_raw'` — demand_intent_raw and
capacity_expansion rows are never selected, read, or written. All existing
news_raw producer_ids are treated as naive-tagger guesses (per carried-forward
state): a confident new resolution overwrites them, an 'unresolved' outcome
nulls them, and every row is logged either way.

Audit log: append-only JSONL, one timestamped line per processed row per run.
DB idempotency comes from the resolver being deterministic (second run finds
new == current and writes nothing); log lines are distinguished by processed_at.
Run:  python entity_resolution/reprocess_signals.py
"""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from entity_resolution import resolver  # noqa: E402

LOG_PATH = PROJECT_ROOT / "data" / "entity_resolution_log.jsonl"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("reprocess_signals")


def reprocess(db_path=None, catalog=None, embed_fn=None, log_path=LOG_PATH):
    """Resolve producer for every news_raw row; update + audit-log. Returns summary."""
    catalog = catalog or resolver.load_producer_catalog()
    conn = db_init.get_connection(db_path)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {"processed": 0, "changed": 0,
               "exact_alias": 0, "fuzzy": 0, "embedding": 0, "unresolved": 0}
    try:
        rows = conn.execute(
            "SELECT signal_id, producer_id, raw_text FROM signal "
            "WHERE signal_type = 'news_raw' ORDER BY signal_id"
        ).fetchall()

        with open(log_path, "a", encoding="utf-8") as log_file:
            for signal_id, old_pid, raw_text in rows:
                result = resolver.resolve_producer(raw_text or "", catalog, embed_fn=embed_fn)
                new_pid = result.producer_id  # None when unresolved

                if new_pid != old_pid:
                    # Scope guard: WHERE signal_type='news_raw' on every write.
                    conn.execute(
                        "UPDATE signal SET producer_id = ? "
                        "WHERE signal_id = ? AND signal_type = 'news_raw'",
                        (new_pid, signal_id),
                    )
                    summary["changed"] += 1

                log_file.write(json.dumps({
                    "signal_id": signal_id,
                    "old_producer_id": old_pid,
                    "new_producer_id": new_pid,
                    "method": result.method,
                    "confidence": result.confidence,
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                }) + "\n")

                summary["processed"] += 1
                summary[result.method] += 1
        conn.commit()
    finally:
        conn.close()

    _print_summary(summary)
    return summary


def _print_summary(s):
    print("=" * 56)
    print(f"Processed news_raw rows : {s['processed']}")
    print(f"  exact_alias : {s['exact_alias']}")
    print(f"  fuzzy       : {s['fuzzy']}")
    print(f"  embedding   : {s['embedding']}")
    print(f"  unresolved  : {s['unresolved']}")
    print(f"producer_id changed     : {s['changed']}")
    print("=" * 56)


if __name__ == "__main__":
    reprocess()
