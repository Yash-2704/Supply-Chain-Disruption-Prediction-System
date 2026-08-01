"""Streamlit dashboard: faithful, read-only rendering of the dashboard_data contract.

All presentation logic lives in formatters.py (pure functions, no UI imports);
app.py only wires those to Streamlit widgets. No SQL, no invented scoring.
"""
