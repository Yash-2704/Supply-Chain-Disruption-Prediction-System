"""Two separately-designed prompt builders — news vs. filing metadata.

Both instruct the model to return ONLY a bare JSON object (no prose, no code
fences) with the exact required fields. They differ in how they frame the input:
a full news article vs. sparse SEC 8-K metadata.
"""
from .validator import EVENT_TYPES

_EVENT_TYPE_LIST = ", ".join(sorted(EVENT_TYPES))

# Shared description of the exact output contract. Both prompts embed this so the
# schema the LLM is told to produce always matches validator.validate_extraction.
_OUTPUT_CONTRACT = f"""Return ONLY a single valid JSON object and nothing else — no explanation, no
commentary, no markdown, no code fences. The object must have exactly these fields:
  "event_type": one of [{_EVENT_TYPE_LIST}]
  "severity": integer from 1 (negligible) to 10 (severe supply-chain impact)
  "confidence": float from 0.0 to 1.0 — YOUR confidence in this judgment
  "entities": list of strings — company/place names mentioned (may be empty)
  "reasoning": one short sentence justifying the event_type and severity"""


def build_news_prompt(raw_text: str, resource_name: str) -> str:
    """Prompt for a canonical news article about `resource_name`."""
    return f"""You are a supply-chain risk analyst. Classify the news item below,
which is potentially relevant to the semiconductor resource "{resource_name}".

Judge how much this event signals disruption to the supply or demand of that
resource, and score its severity accordingly.

NEWS TEXT:
\"\"\"{raw_text}\"\"\"

{_OUTPUT_CONTRACT}"""


def build_demand_intent_prompt(raw_text: str, resource_name: str) -> str:
    """Prompt for a demand-intent SEC 8-K row. Input is SPARSE metadata (form
    type + item codes + brief description), NOT a full filing — the model must
    reflect that uncertainty with a LOWER confidence rather than inventing a
    severity from almost nothing."""
    return f"""You are a supply-chain risk analyst. The item below is NOT a full document —
it is only brief metadata from an SEC 8-K filing (form type, item codes, and a
short description) by a company whose activity may affect demand for the
semiconductor resource "{resource_name}".

This is LIMITED information. If the metadata is too sparse to meaningfully judge
supply-chain impact, say so through a LOW "confidence" value and a conservative
"severity" — do NOT fabricate a confident, high severity from almost no content.

FILING METADATA:
\"\"\"{raw_text}\"\"\"

{_OUTPUT_CONTRACT}"""
