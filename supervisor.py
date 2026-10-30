"""
Точка входа супервизора: настройка логирования и запуск торгового бота.
"""

from __future__ import annotations

import os
import subprocess
import sys

from dotenv import load_dotenv

# Без Docker настройки берутся из .env рядом с кодом. Docker (env_file) и
# Makefile задают переменные раньше — уже заданные значения не перезаписываются.
_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_ROOT, ".env"))

from src.infra.logger import setup_logging  # noqa: E402

setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    json_logs=os.getenv("JSON_LOGS", "false").lower() == "true",
)

import asyncio  # noqa: E402
import logging  # noqa: E402
import signal  # noqa: E402

from config import Config  # noqa: E402
from src.core.trading_bot import TradingBot  # noqa: E402
from src.infra.health_server import heartbeat_loop, start_health_server  # noqa: E402
from src.settings.secrets_store import apply_secrets  # noqa: E402
from src.settings.settings_registry import effective_config  # noqa: E402

logger = logging.getLogger(__name__)

_RETRY_DELAY: int = 30


async def main() -> None:
    """
    Запускает торгового бота и обрабатывает завершение работы.

    При SIGINT/SIGTERM (в т.ч. docker stop) корректно вызывает bot.stop().
    При незапланированном исключении выжидает _RETRY_DELAY сек и перезапускает.
    """
    bot = TradingBot()
    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()

    def _handle_signal(sig: int) -> None:
        logger.info(
            "Получен сигнал %s — завершаем работу",
            signal.Signals(sig).name,
        )
        if main_task and not main_task.done():
            main_task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal, sig)

    if Config.HEALTH_PORT > 0:
        await start_health_server(bot, port=Config.HEALTH_PORT)
    heartbeat = asyncio.create_task(heartbeat_loop())

    try:
        while True:
            try:
                await bot.initialize()
                await bot.trading_loop()
                break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(
                    "Критическая ошибка: %s. Перезапуск через %dс...",
                    e,
                    _RETRY_DELAY,
                    exc_info=True,
                )
                try:
                    await bot.stop()
                except Exception:
                    pass
                await asyncio.sleep(_RETRY_DELAY)
                bot = TradingBot()
                if Config.HEALTH_PORT > 0:
                    await start_health_server(bot, port=Config.HEALTH_PORT)
    except asyncio.CancelledError:
        logger.info("Задача отменена")
    except Exception as e:
        logger.critical("Критическая ошибка супервизора: %s", e, exc_info=True)
        raise
    finally:
        heartbeat.cancel()
        await bot.stop()
        logger.info("Бот остановлен")


def _start_dashboard() -> "subprocess.Popen[bytes]":
    """Дашборд и приложение /app/ отдельным процессом — для запуска без Docker.

    В Docker у дашборда свой контейнер, поэтому только по флагу --with-dashboard.
    """
    logger.info("Запускаю дашборд: приложение — http://localhost:8080/app/")
    return subprocess.Popen([sys.executable, os.path.join(_ROOT, "dashboard.py")])


if __name__ == "__main__":
    # Ключи, введённые в приложении, участвуют в проверке так же, как .env.
    apply_secrets(Config)
    try:
        effective_config(Config).validate()
    except ValueError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    dashboard = _start_dashboard() if "--with-dashboard" in sys.argv else None
    try:
        asyncio.run(main())
    finally:
        if dashboard is not None:
            dashboard.terminate()
            try:
                dashboard.wait(timeout=10)
            except subprocess.TimeoutExpired:
                dashboard.kill()
