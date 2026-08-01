"""Pure, standalone validation of an LLM extraction dict.

Kept deliberately separate from the client so it is directly unit-testable.
A response that fails ANY check is rejected wholesale — we never repair or
guess substitute values.
"""
from typing import Any

# The closed set of event types the LLM may choose from. Single source of truth,
# imported by prompts.py so the prompt and the validator can never drift apart.
EVENT_TYPES = frozenset({
    "supply_disruption",
    "demand_signal",
    "price_movement",
    "regulatory_policy",
    "other",
})


def validate_extraction(parsed: Any) -> bool:
    """Return True iff `parsed` is a dict matching the required extraction shape:
    event_type in EVENT_TYPES; severity int in [1,10]; confidence number in
    [0.0,1.0]; entities a list; reasoning a non-empty string."""
    if not isinstance(parsed, dict):
        return False

    if parsed.get("event_type") not in EVENT_TYPES:
        return False

    severity = parsed.get("severity")
    # bool is a subclass of int — exclude it explicitly.
    if isinstance(severity, bool) or not isinstance(severity, int):
        return False
    if not (1 <= severity <= 10):
        return False

    confidence = parsed.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return False
    if not (0.0 <= float(confidence) <= 1.0):
        return False

    if not isinstance(parsed.get("entities"), list):
        return False

    reasoning = parsed.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        return False

    return True
