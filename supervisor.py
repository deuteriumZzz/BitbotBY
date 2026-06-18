"""
Точка входа супервизора: настройка логирования и запуск торгового бота.
"""

from __future__ import annotations

import os

from src.logger import setup_logging

setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    json_logs=os.getenv("JSON_LOGS", "false").lower() == "true",
)

import asyncio  # noqa: E402
import logging  # noqa: E402

from src.trading_bot import TradingBot  # noqa: E402

logger = logging.getLogger(__name__)


async def main() -> None:
    """
    Запускает торгового бота и обрабатывает завершение работы.

    При KeyboardInterrupt или отмене задачи корректно останавливает бота.
    """
    bot = TradingBot()
    try:
        await bot.initialize()
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Получен сигнал остановки (KeyboardInterrupt)")
    except asyncio.CancelledError:
        logger.info("Задача отменена")
        raise
    except Exception as e:
        logger.critical(f"Критическая ошибка супервизора: {e}", exc_info=True)
        raise
    finally:
        await bot.stop()
        logger.info("Бот остановлен")


if __name__ == "__main__":
    asyncio.run(main())
