"""Read-only data-access layer between the SQLite database and the (next-stage) UI.

Pure Python + sqlite3. No UI imports, no writes, no new business logic — only
retrieval, filtering, and "latest row" selection, returning typed, JSON-
serializable dataclasses whose None/sentinel values make real-data gaps explicit.
"""
