"""
BitbotBY Web Dashboard entry point — runs on port 8080.

Application logic lives in src/dashboard/:
  data.py             — Redis/SQLite helpers, Prometheus gauges
  routers/api.py      — /api/status, /api/trades, /api/backtest
  routers/ops.py      — /metrics, /health, /webhook/alerts
  routers/frontend.py — / (HTML SPA)
  index.html          — single-page dashboard UI
"""

import os
import sys

import uvicorn

sys.path.insert(0, os.path.dirname(__file__))

from src.dashboard import app  # noqa: F401 — re-exported for uvicorn discovery

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",  # bind to localhost; use a reverse proxy for external access
        port=8080,
        log_level="warning",
    )
