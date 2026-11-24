"""Запуск API приложения: python -m bitbot.api (порт 8080).

Приложение FastAPI — bitbot/api/__init__.py; маршруты — bitbot/api/routers/.
"""

import os
import sys

import uvicorn
from dotenv import load_dotenv

# Без Docker настройки берутся из .env; заданные раньше (Docker, shell) не трогаем.
load_dotenv(".env")

from bitbot.api import app  # noqa: F401,E402 — re-exported for uvicorn discovery
from bitbot.config.constants import WEBAPP_DIST  # noqa: E402

if __name__ == "__main__":
    if not (WEBAPP_DIST / "index.html").exists():
        print(
            "Приложение /app/ не собрано: cd webapp && npm ci && npm run build",
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
