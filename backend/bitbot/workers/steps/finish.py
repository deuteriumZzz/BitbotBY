"""Конец цикла: обновление SAC, отметки цикла, сохранение позиций, пауза до следующего.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Optional

from bitbot.workers.health_server import cycles_counter
from bitbot.workers.steps.context import CycleContext

if TYPE_CHECKING:
    from bitbot.workers.bot import TradingBot

logger = logging.getLogger("bitbot.workers.bot")


async def run(bot: "TradingBot", ctx: CycleContext) -> Optional[CycleContext]:
    loop, t0, cycle = asyncio.get_running_loop(), ctx.t0, ctx.cycle
    bot.combiner.sac.reload_if_updated()
    bot._last_cycle_at = time.time()
    bot._trace.end_cycle()
    await bot._check_silent_death()
    bot._save_monitored_state()

    cycles_counter.inc()
    elapsed = loop.time() - t0
    _interval = bot._runtime_config.get_trading_interval()
    sleep_for = max(0, _interval - elapsed)
    logger.info(
        "Cycle #%d: %.1fs elapsed, sleeping %.1fs (interval=%ds)",
        cycle,
        elapsed,
        sleep_for,
        _interval,
    )
    await bot._pause(sleep_for)
    return ctx
