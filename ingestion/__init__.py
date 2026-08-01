"""News ingestion package: API clients, coarse tagger, and orchestration.

Clients (gdelt_client, newsapi_client) only fetch + normalize. tagger only
maps text to resource/producer ids. Only scripts/ingest_news.py touches the DB.
"""
