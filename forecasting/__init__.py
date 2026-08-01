"""Forecasting stage: per-resource price-proxy and demand-intent-activity trends.

Prophet is the primary model (with uncertainty intervals); Holt-Winters is an
independent point-estimate sanity check. Series with too little history report
'insufficient_data' rather than a meaningless forecast.
"""
