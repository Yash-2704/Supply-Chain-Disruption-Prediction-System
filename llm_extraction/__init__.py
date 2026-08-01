"""LLM event-extraction stage: multi-provider (Groq primary; Mistral, Cerebras fallback).

Populates signal.severity / signal.confidence (LLM judgment, not hard facts)
and an additive event_extraction audit table. Every LLM response is parsed and
schema-validated before use; malformed output yields NULL, never a guess.
"""
