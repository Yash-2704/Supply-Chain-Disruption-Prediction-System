"""Empirically probe each configured LLM provider's free-tier generosity and rank.

For every provider in llm_client.PROVIDERS with a key present, fires a short burst
of minimal requests, counts HTTP 200 vs 429 (rate-limited) vs other, measures
throughput/latency, and captures any rate-limit response headers. Prints a ranking.

Run:  python verification/llm_provider_quota_test.py
NOTE: this spends a small amount of each provider's quota by design.
"""
import os
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llm_extraction.llm_client import PROVIDERS  # noqa: E402

BURST = 15
PROMPT = "Reply with exactly one word: OK"


def _headers(name, key):
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _rate_headers(resp):
    return {k: v for k, v in resp.headers.items() if "ratelimit" in k.lower()}


def probe(name):
    cfg = PROVIDERS[name]
    key = os.environ.get(cfg["env"])
    if not key:
        return {"provider": name, "status": f"SKIPPED (no {cfg['env']})"}
    succ = err429 = other = 0
    latencies = []
    rate_headers = {}
    t0 = time.time()
    for _ in range(BURST):
        s = time.time()
        try:
            r = requests.post(cfg["url"], headers=_headers(name, key), timeout=45, json={
                "model": cfg["model"],
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": 8, "temperature": 0})
            latencies.append(time.time() - s)
            if r.status_code == 200:
                succ += 1
            elif r.status_code == 429:
                err429 += 1
            else:
                other += 1
            found = _rate_headers(r)
            if found:
                rate_headers = found
        except Exception:
            other += 1
    elapsed = time.time() - t0
    result = {
        "provider": name, "model": cfg["model"], "status": "probed",
        "success": succ, "rate_limited_429": err429, "other_errors": other,
        "burst": BURST, "elapsed_s": round(elapsed, 2),
        "throughput_req_s": round(BURST / elapsed, 2) if elapsed else None,
        "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "rate_limit_headers": rate_headers,
    }
    return result


def main():
    results = [probe(name) for name in PROVIDERS]

    print("=" * 74)
    print("LLM PROVIDER QUOTA / GENEROSITY PROBE")
    print("=" * 74)
    for r in results:
        if r["status"].startswith("SKIPPED"):
            print(f"\n[{r['provider']}] {r['status']}")
            continue
        print(f"\n[{r['provider']}] model={r['model']}")
        print(f"    burst {r['burst']} reqs -> success={r['success']} "
              f"rate_limited(429)={r['rate_limited_429']} other_errors={r['other_errors']}")
        print(f"    throughput={r['throughput_req_s']} req/s  avg_latency={r['avg_latency_s']}s")
        if r["rate_limit_headers"]:
            print(f"    rate-limit headers: {r['rate_limit_headers']}")

    # Rank: probed providers by (success desc, 429 asc, throughput desc).
    probed = [r for r in results if r["status"] == "probed"]
    ranked = sorted(probed, key=lambda r: (-r["success"], r["rate_limited_429"],
                                           -(r["throughput_req_s"] or 0)))
    print("\n" + "=" * 74)
    print("RANKING (most generous / reliable first) — based on this burst probe:")
    for i, r in enumerate(ranked, 1):
        print(f"  {i}. {r['provider']:11s} success={r['success']}/{r['burst']} "
              f"429={r['rate_limited_429']} throughput={r['throughput_req_s']} req/s")
    print("=" * 74)
    if ranked:
        print("Suggested PROVIDER_ORDER:",
              [r["provider"] for r in ranked] + [p for p in PROVIDERS
                                                 if p not in {r["provider"] for r in probed}])
    return ranked


if __name__ == "__main__":
    main()
