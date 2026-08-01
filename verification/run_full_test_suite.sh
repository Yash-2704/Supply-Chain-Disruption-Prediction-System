#!/usr/bin/env bash
# Runs EVERY test file in tests/ and reports the ACTUAL current pass/fail count.
# Nothing here is transcribed from any prior document — it is whatever pytest
# says right now. Run from the project root.
set -uo pipefail
cd "$(dirname "$0")/.."

PYTEST=".venv/bin/python -m pytest"
echo "=== Full test suite — fresh run at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "--- per-file result ---"
overall=0
for f in tests/test_*.py; do
  out=$($PYTEST "$f" -q 2>&1 | tail -1)
  echo "$(printf '%-40s' "$f") $out"
  echo "$out" | grep -qE "failed|error" && overall=1
done
echo "--- aggregate ---"
$PYTEST tests/ -q 2>&1 | tail -1
exit $overall
