"""
BitbotBY Web Dashboard entry point — runs on port 8080.

Application logic lives in src/dashboard/:
  data.py             — Redis/SQLite helpers, Prometheus gauges
  routers/api.py      — /api/status, /api/trades, /api/backtest
  routers/ops.py      — /metrics, /health, /webhook/alerts
  routers/frontend.py — / → приложение /app/ (webapp/dist)
"""

import os
import sys

import uvicorn
from dotenv import load_dotenv

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
# Без Docker настройки берутся из .env; заданные раньше (Docker, shell) не трогаем.
load_dotenv(os.path.join(_ROOT, ".env"))

from src.dashboard import app  # noqa: F401,E402 — re-exported for uvicorn discovery

if __name__ == "__main__":
    if not os.path.exists(os.path.join(_ROOT, "webapp", "dist", "index.html")):
        print(
            "Приложение /app/ не собрано: cd webapp && npm ci && npm run build",
            file=sys.stderr,
        )
    import logging

    from src.infra.logger import install_masking

    # Свой обработчик вместо конфигурации uvicorn: логи uvicorn идут в корень,
    # где секреты маскируются (как в процессе бота).
    logging.basicConfig(level=logging.WARNING)
    for _h in logging.getLogger().handlers:
        install_masking(_h)
    uvicorn.run(
        app,
        # Вне Docker — только этот компьютер; образ задаёт BIND_HOST=0.0.0.0
        # (наружу порт открывает compose: 127.0.0.1 локально, Caddy на сервере).
        host=os.getenv("BIND_HOST", "127.0.0.1"),
        port=8080,
        log_level="warning",
        log_config=None,
    )
