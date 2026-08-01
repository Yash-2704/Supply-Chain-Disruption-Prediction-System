"""Producer entity-resolution stage: three-layer resolution + reprocessing.

Replaces the naive first-match producer tagging on `news_raw` signal rows with
an honest exact -> fuzzy -> embedding pipeline that records confidence + method.
"""
