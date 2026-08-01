"""Multi-provider LLM abstraction (Gemini removed).

All providers are OpenAI-compatible chat-completions endpoints, called uniformly
through `call_provider` with plain `requests` (no per-provider SDKs). Providers
are tried in `PROVIDER_ORDER` (ranked by observed free-tier quota generosity —
see verification/llm_provider_quota_test.py); the first that yields parseable,
schema-valid output wins. Any provider whose API key is absent is skipped.
No call ever raises to the caller — failures return None and are logged.
"""
import json
import logging
import os
import re
import threading
import time
from typing import Optional, Tuple

import requests

from .validator import validate_extraction

logger = logging.getLogger(__name__)

_TIMEOUT = 45

# Per-provider free-tier requests/minute PER KEY. Providers listed here are paced
# so we don't exceed the cap and waste calls on 429s. Cerebras is 5 req/min/key;
# with N rotation keys the effective spacing is 60/(rpm*N) seconds. Groq/Mistral
# have generous per-minute limits and are not paced.
_PROVIDER_RPM = {"cerebras": 5}
_pace_lock = threading.Lock()
_last_call_ts: dict = {}   # provider -> monotonic timestamp of last request
_rr_index: dict = {}       # provider -> round-robin key counter

# OpenAI-compatible providers. Each: chat-completions URL, model id, env-var key.
# (OpenRouter removed — its free pool was tiny/shared and constantly 429'd.)
PROVIDERS = {
    "groq":     {"url": "https://api.groq.com/openai/v1/chat/completions",
                 "model": "llama-3.3-70b-versatile", "env": "GROQ_API_KEY"},
    "mistral":  {"url": "https://api.mistral.ai/v1/chat/completions",
                 "model": "mistral-small-latest", "env": "MISTRAL_API_KEY"},
    "cerebras": {"url": "https://api.cerebras.ai/v1/chat/completions",
                 "model": "gemma-4-31b", "env": "CEREBRAS_API_KEY"},
}

# Fallback order, ranked by an empirical burst probe (verification/
# llm_provider_quota_test.py). Observed free-tier limits:
#   groq     — 1000 req/day, 12k tok/min; fastest (~0.3s). PRIMARY.
#   mistral  — 50 req/min, 50k tok/min; reliable fallback.
#   cerebras — 2400 req/day + 1M tok/day but only 5 req/min (needs pacing);
#              highest DAILY budget, throttles under burst — deepest fallback.
PROVIDER_ORDER = ["groq", "mistral", "cerebras"]


def _provider_keys(name: str) -> list:
    """All API keys configured for a provider, for rotation. Convention:
    PRIMARY env var (e.g. GROQ_API_KEY) plus numbered extras GROQ_API_KEY_2,
    GROQ_API_KEY_3, ... — tried in order, rotating past a rate-limited key."""
    env = PROVIDERS[name]["env"]
    keys = []
    if os.environ.get(env):
        keys.append(os.environ[env])
    i = 2
    while os.environ.get(f"{env}_{i}"):
        keys.append(os.environ[f"{env}_{i}"])
        i += 1
    return keys


def _pace(name: str, n_keys: int) -> None:
    """Throttle a paced provider (e.g. Cerebras) to its per-minute cap, spread
    across its rotation keys. No-op for un-paced providers."""
    rpm = _PROVIDER_RPM.get(name)
    if not rpm:
        return
    min_interval = 60.0 / (rpm * max(1, n_keys))
    with _pace_lock:
        wait = min_interval - (time.monotonic() - _last_call_ts.get(name, 0.0))
        if wait > 0:
            time.sleep(wait)
        _last_call_ts[name] = time.monotonic()


def call_provider(name: str, prompt: str) -> Optional[str]:
    """Call one OpenAI-compatible provider; return raw text or None (logged).

    Rotates round-robin across all configured keys (so load spreads evenly) and,
    on a 429, moves to the next key — so extra keys extend the effective quota.
    Paced providers (Cerebras) are throttled to their per-minute cap first."""
    cfg = PROVIDERS.get(name)
    if cfg is None:
        logger.warning("Unknown provider %r.", name)
        return None
    keys = _provider_keys(name)
    if not keys:
        logger.warning("%s: %s not set — skipping this provider.", name, cfg["env"])
        return None
    payload = {"model": cfg["model"],
               "messages": [{"role": "user", "content": prompt}], "temperature": 0}

    # Round-robin: start this call at a rotating key so successive calls spread
    # across keys rather than always hammering key #1.
    n = len(keys)
    start = _rr_index.get(name, 0) % n
    _rr_index[name] = start + 1
    ordered = keys[start:] + keys[:start]

    for offset, key in enumerate(ordered):
        _pace(name, n)  # respect the per-minute cap (Cerebras) before each request
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        key_no = (start + offset) % n + 1
        try:
            resp = requests.post(cfg["url"], headers=headers, timeout=_TIMEOUT, json=payload)
        except Exception as exc:
            logger.warning("%s key #%d network error: %s", name, key_no, exc)
            continue
        if resp.status_code == 429:  # rate-limited -> rotate to the next key
            logger.warning("%s key #%d rate-limited (429); rotating to next key.", name, key_no)
            continue
        try:
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.warning("%s key #%d call failed: %s", name, key_no, exc)
            continue
    return None


def available_providers() -> list:
    """Providers (in fallback order) that have at least one API key configured."""
    return [n for n in PROVIDER_ORDER if _provider_keys(n)]


def generate_text(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """Try providers in order; return (raw_text, provider_name) of the first that
    responds, or (None, None) if all are unavailable/failing. (call_provider
    returns None for any provider whose key is absent, so no pre-skip needed.)"""
    for name in PROVIDER_ORDER:
        raw = call_provider(name, prompt)
        if raw is not None:
            return raw, name
    return None, None


def _parse_json_object(text: Optional[str]) -> Optional[dict]:
    """Best-effort parse of a JSON object from model output. Strips ```code
    fences``` and surrounding prose; returns None if nothing valid parses."""
    if not text or not text.strip():
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t.strip())
    try:
        return json.loads(t)
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\{.*\}", t, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def extract_structured(prompt: str) -> Optional[dict]:
    """Try providers in PROVIDER_ORDER; return the first parseable, schema-valid
    result with `_provider`/`_model`/`_raw` added, or None if all fail. A provider
    that responds with unparseable or invalid output falls through to the next."""
    for name in PROVIDER_ORDER:
        raw = call_provider(name, prompt)  # None if the provider's key is absent
        if raw is None:
            continue
        parsed = _parse_json_object(raw)
        if parsed is None:
            logger.warning("%s returned output that could not be parsed as JSON.", name)
            continue
        if not validate_extraction(parsed):
            logger.warning("%s output failed schema validation; rejecting.", name)
            continue
        parsed["_provider"] = name
        parsed["_model"] = PROVIDERS[name]["model"]
        parsed["_raw"] = raw
        return parsed
    return None
