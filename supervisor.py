"""
Точка входа супервизора: настройка логирования и запуск торгового бота.
"""

from __future__ import annotations

import os
import sys

from src.logger import setup_logging

setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    json_logs=os.getenv("JSON_LOGS", "false").lower() == "true",
)

import asyncio  # noqa: E402
import logging  # noqa: E402
import signal  # noqa: E402

from config import Config  # noqa: E402
from src.health_server import start_health_server  # noqa: E402
from src.trading_bot import TradingBot  # noqa: E402

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
        await bot.stop()
        logger.info("Бот остановлен")


if __name__ == "__main__":
    try:
        Config().validate()
    except ValueError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(main())
