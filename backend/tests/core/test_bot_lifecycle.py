"""Старт и остановка: Redis проверяется сразу, остановка прерывает паузу цикла."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from bitbot.workers.bot import TradingBot


def _bot():
    return SimpleNamespace(_stop_event=asyncio.Event(), is_running=True)


async def test_request_stop_ends_pause_between_cycles():
    bot = _bot()
    pause = asyncio.create_task(TradingBot._pause(bot, 3600))
    await asyncio.sleep(0)
    TradingBot.request_stop(bot)
    await asyncio.wait_for(pause, 1)  # не ждём час до следующего цикла
    assert bot.is_running is False


async def test_pause_without_stop_waits_its_time():
    bot = _bot()
    await TradingBot._pause(bot, 0.01)
    assert bot.is_running is True


def test_startup_fails_clearly_without_redis():
    bot = SimpleNamespace(redis=MagicMock())
    bot.redis.redis_client.ping.side_effect = ConnectionError("refused")
    with pytest.raises(RuntimeError, match="Redis недоступен"):
        TradingBot._check_dependencies(bot)
