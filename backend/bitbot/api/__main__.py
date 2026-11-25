"""Запуск API приложения: python -m bitbot.api (порт 8080).

Приложение FastAPI — bitbot/api/__init__.py; маршруты — bitbot/api/routers/.
"""

import os
import sys

import uvicorn

from bitbot.api import app  # noqa: F401 — re-exported for uvicorn discovery
from bitbot.config.constants import FRONTEND_DIST

if __name__ == "__main__":
    if not (FRONTEND_DIST / "index.html").exists():
        print(
            "Приложение /app/ не собрано: cd frontend && npm ci && npm run build",
            file=sys.stderr,
        )
    import logging

    from bitbot.infra.logger import install_masking

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
