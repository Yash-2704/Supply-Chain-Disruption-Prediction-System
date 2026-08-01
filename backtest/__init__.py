"""Backtest harness with a rigorously tested as-of cutoff mechanism.

Operates against an ISOLATED database (never the real one). Because genuine
2023-2025-dated news is unobtainable through this project's free-tier clients
(GDELT is recent-window; NewsAPI free tier ~30 days), the harness runs on
clearly-labeled SYNTHETIC historical fixtures and validates the harness
MECHANISM — it does NOT constitute a real historical validation.
"""
