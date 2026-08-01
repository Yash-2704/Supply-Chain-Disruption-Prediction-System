"""Near-duplicate detection for news_raw signals: embed -> cluster -> canonical.

Reuses entity_resolution.resolver's embedding loader so vectors are identical
and comparable across the resolution and clustering stages.
"""
