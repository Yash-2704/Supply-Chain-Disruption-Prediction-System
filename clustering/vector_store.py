"""Thin wrapper around chromadb in EMBEDDED mode (no server, no service).

Persists news_raw embeddings so a later RAG explanation stage can retrieve
supporting articles. This stage only populates the store; it does not query it.
Upserts are keyed on str(signal_id), so re-runs never create duplicate vectors.
"""
import logging
from pathlib import Path
from typing import Optional

import chromadb

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "news_signals"

# Chroma metadata values must be str/int/float/bool — never None.
_META_FIELDS = ("resource_id", "producer_id", "timestamp", "source_url")


def get_client(path: Optional[Path] = None, ephemeral: bool = False):
    """Persistent local client by default; ephemeral (in-memory) for tests."""
    if ephemeral:
        return chromadb.EphemeralClient()
    path = Path(path) if path else DEFAULT_CHROMA_PATH
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def get_collection(client, name: str = COLLECTION_NAME):
    """Get/create the collection. We always pass explicit embeddings, so the
    default embedding function is never invoked (no model download). `name` is
    overridable so tests can isolate collections (Chroma's in-memory client can
    share state across instances in one process)."""
    return client.get_or_create_collection(name=name)


def _clean_metadata(metadata: dict) -> dict:
    """Coerce None -> '' so Chroma accepts the metadata."""
    return {k: (metadata.get(k) if metadata.get(k) is not None else "") for k in _META_FIELDS}


def upsert_signal(collection, signal_id, embedding, metadata: dict) -> None:
    """Upsert one signal's vector. Idempotent by id — a repeated signal_id
    replaces its entry rather than adding a duplicate."""
    vec = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
    collection.upsert(
        ids=[str(signal_id)],
        embeddings=[vec],
        metadatas=[_clean_metadata(metadata or {})],
    )
