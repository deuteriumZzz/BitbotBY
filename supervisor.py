"""
Process supervisor for the trading bot.

Usage: python supervisor.py

Restarts the bot on crash (max 5 times, 30s between retries).
Sends Telegram alert on crash and restart.
"""
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_MAX_RESTARTS = 5
_RESTART_DELAY = 30  # seconds


async def _notify(token: str, chat_id: str, text: str):
    if not token or not chat_id:
        return
    try:
        from telegram import Bot
        bot = Bot(token)
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="MarkdownV2",
        )
    except Exception as e:
        logger.error(f"Telegram notify failed: {e}")


async def run_with_supervision():
    restarts = 0
    while restarts <= _MAX_RESTARTS:
        try:
            logger.info(
                f"Starting bot "
                f"(attempt {restarts + 1}"
                f"/{_MAX_RESTARTS + 1})"
            )
            # Import here so each restart gets fresh state
            from src.trading_bot import TradingBot
            bot = TradingBot()
            try:
                await bot.initialize()
                await bot.trading_loop()
            except KeyboardInterrupt:
                raise
            finally:
                try:
                    await bot.stop()
                except Exception as stop_err:
                    logger.error(
                        f"Error during bot.stop(): "
                        f"{stop_err}"
                    )
        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            return
        except Exception as e:
            restarts += 1
            logger.error(
                f"Bot crashed (attempt {restarts}): {e}",
                exc_info=True,
            )
            if restarts > _MAX_RESTARTS:
                msg = (
                    "*BitbotBY ОСТАНОВЛЕН*\n"
                    f"Превышен лимит перезапусков"
                    f" \\({_MAX_RESTARTS}\\)\\.\n"
                    f"Последняя ошибка:"
                    f" `{str(e)[:200]}`"
                )
                await _notify(
                    Config.TELEGRAM_BOT_TOKEN,
                    Config.TELEGRAM_CHAT_ID,
                    msg,
                )
                logger.error("Max restarts reached. Exiting.")
                sys.exit(1)

            msg = (
                f"*BitbotBY упал* — перезапуск"
                f" \\({restarts}/{_MAX_RESTARTS}\\)\n"
                f"Ошибка: `{str(e)[:200]}`\n"
                f"Следующая попытка через"
                f" {_RESTART_DELAY}с"
            )
            await _notify(
                Config.TELEGRAM_BOT_TOKEN,
                Config.TELEGRAM_CHAT_ID,
                msg,
            )
            await asyncio.sleep(_RESTART_DELAY)
        else:
            # Clean exit
            logger.info("Bot stopped cleanly.")
            return


if __name__ == "__main__":
    asyncio.run(run_with_supervision())
