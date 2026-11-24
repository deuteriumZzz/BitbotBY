"""Ордер (узел «Ордер»): фильтр по балансу, порядок кандидатов, исполнение лучшего.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from bitbot.workers.steps.context import CycleContext

if TYPE_CHECKING:
    from bitbot.workers.bot import TradingBot

logger = logging.getLogger("bitbot.workers.bot")


async def run(bot: "TradingBot", ctx: CycleContext) -> Optional[CycleContext]:
    recs, balance, cycle, market_data = (
        ctx.recs,
        ctx.balance,
        ctx.cycle,
        ctx.market_data,
    )
    filtered = bot._filter_by_balance(recs, balance)
    bot._publish_signal_stages(recs, filtered)
    filtered.sort(key=bot._composite_score, reverse=True)

    bot._cycle.print_recommendations(filtered, balance, cycle)
    await bot._cycle.notify_new_signals(filtered, balance, cycle)
    await bot._execute_top_rec(filtered, market_data)
    await bot._update_performance_stats()
    return ctx
