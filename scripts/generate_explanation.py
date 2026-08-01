"""Orchestrate RAG explanation generation for a resource.

Retrieves canonical evidence; if none, writes 'insufficient_evidence' and never
calls the LLM. Otherwise builds the prompt, calls the ranked provider chain (reusing the
established fallback pattern), validates every citation against retrieved
evidence, applies the disclaimer in code, and writes an 'ok' row — or a clean
failure row on LLM/validation failure. New timestamped row per run (idempotency
by regeneration; is_real_data_model fixed at write time).
Run:  python scripts/generate_explanation.py [resource_id] [prediction_id]
"""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import db_init  # noqa: E402
from clustering import vector_store  # noqa: E402
from rag import retriever, narrator  # noqa: E402
# Reuse the exact provider-fallback machinery from the LLM extraction stage.
from llm_extraction import llm_client  # noqa: E402

EXPLANATION_SCHEMA_PATH = PROJECT_ROOT / "schema_explanations.sql"
RESOURCES = ("dram", "hbm", "nand", "gpu")
MAX_LLM_ATTEMPTS = 2  # one initial + one retry on invalid citations

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("generate_explanation")


def init_explanation_schema(conn):
    conn.executescript(EXPLANATION_SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()


def _load_prediction(conn, prediction_id):
    """Return (prediction_dict_or_None, is_real_data_model) for a prediction_id."""
    row = conn.execute(
        "SELECT p.id, p.resource_id, p.calibrated_probability, p.raw_score, "
        "p.top_shap_features, t.data_mode FROM prediction p "
        "JOIN training_run t ON p.run_id = t.run_id WHERE p.id = ?",
        (prediction_id,)).fetchone()
    if not row:
        return None, False
    pred = {
        "prediction_id": row[0], "resource_id": row[1],
        "calibrated_probability": row[2], "raw_score": row[3],
        "top_shap_features": json.loads(row[4]) if row[4] else None,
    }
    return pred, (row[5] == "real")


def _generate_once(prompt):
    """One narration attempt through the ranked provider chain (llm_client).
    Returns (raw_text, provider_name) of the first provider that responds."""
    return llm_client.generate_text(prompt)


def _write(conn, resource_id, prediction_id, is_real, evidence_count, status,
           narrative=None, citations=None, provider=None):
    conn.execute(
        "INSERT INTO explanation (resource_id, prediction_id, narrative, citations, "
        "evidence_count, llm_provider, is_real_data_model, status, generated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (resource_id, prediction_id, narrative,
         json.dumps(citations) if citations is not None else None,
         evidence_count, provider, 1 if is_real else 0, status,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def explain_resource(resource_id, prediction_id=None, db_path=None,
                     collection=None, embed_fn=None):
    conn = db_init.get_connection(db_path)
    try:
        init_explanation_schema(conn)

        # Evidence-only mode (no prediction) is forced to is_real=False: nothing
        # backs this explanation, so it must carry the disclaimer.
        if prediction_id is not None:
            prediction, is_real = _load_prediction(conn, prediction_id)
        else:
            prediction, is_real = None, False

        coll = collection if collection is not None else \
            vector_store.get_collection(vector_store.get_client())
        ev_kwargs = {"embed_fn": embed_fn} if embed_fn is not None else {}
        evidence = retriever.retrieve_canonical_evidence(resource_id, conn, coll, **ev_kwargs)

        if not evidence:
            _write(conn, resource_id, prediction_id, is_real, 0, "insufficient_evidence")
            return {"resource_id": resource_id, "status": "insufficient_evidence",
                    "evidence_count": 0, "provider": None}

        prompt = narrator.build_narration_prompt(resource_id, evidence, prediction)

        validated, provider = None, None
        llm_ever_succeeded = False
        for attempt in range(MAX_LLM_ATTEMPTS):
            raw, prov = _generate_once(prompt)
            if raw is None:
                continue  # both providers failed this attempt
            llm_ever_succeeded = True
            parsed = llm_client._parse_json_object(raw)
            candidate = narrator.validate_narration(parsed, evidence) if parsed else None
            if candidate is not None:
                validated, provider = candidate, prov
                break
            logger.warning("Attempt %d produced invalid/unfounded citations; retrying.",
                           attempt + 1)

        if validated is None:
            status = "invalid_citations" if llm_ever_succeeded else "llm_error"
            _write(conn, resource_id, prediction_id, is_real, len(evidence), status)
            return {"resource_id": resource_id, "status": status,
                    "evidence_count": len(evidence), "provider": None}

        # Enforce the disclaimer IN CODE, unconditionally.
        final_narrative = narrator.apply_disclaimer(validated["narrative"], is_real)
        _write(conn, resource_id, prediction_id, is_real, len(evidence), "ok",
               narrative=final_narrative, citations=validated["citations"], provider=provider)
        return {"resource_id": resource_id, "status": "ok",
                "evidence_count": len(evidence), "provider": provider,
                "is_real_data_model": is_real}
    finally:
        conn.close()


def main():
    args = sys.argv[1:]
    if args:
        rid = args[0]
        pid = int(args[1]) if len(args) > 1 else None
        out = explain_resource(rid, pid)
        print(f"[{out['resource_id']:>4}] status={out['status']} "
              f"evidence={out['evidence_count']} provider={out.get('provider')}")
    else:
        print("=" * 60)
        for rid in RESOURCES:
            out = explain_resource(rid)
            print(f"[{rid:>4}] status={out['status']:<22} "
                  f"evidence={out['evidence_count']} provider={out.get('provider')}")
        print("=" * 60)


if __name__ == "__main__":
    main()
