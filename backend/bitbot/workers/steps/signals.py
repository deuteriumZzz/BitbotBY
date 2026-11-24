"""Сигналы (узлы «Стратегии», «AI», «SAC»): режимы рынка, контекст, предфильтр, комбайнер.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from typing import TYPE_CHECKING, Optional

from bitbot.workers.steps.context import CycleContext

if TYPE_CHECKING:
    from bitbot.workers.bot import TradingBot

logger = logging.getLogger("bitbot.workers.bot")


async def run(bot: "TradingBot", ctx: CycleContext) -> Optional[CycleContext]:
    snapshots, market_data, balance = ctx.snapshots, ctx.market_data, ctx.balance
    _open, _max = ctx.open_positions, ctx.max_positions
    # РЕЖИМ ПОИСКА: есть свободные слоты — полный скан + AI
    logger.info("Slots available (%d/%d) — full scan mode", _open, _max)
    regimes = await bot._detect_regimes(market_data)
    _prices = {
        sym: df["close"].iloc[-1]
        for sym, df in market_data.items()
        if df is not None and not df.empty
    }
    try:
        _market_ctx_map = await bot._market_context.get_context_for_symbols(
            list(market_data.keys()), _prices
        )
    except Exception as _mc_exc:
        logger.warning("MarketContext fetch failed: %s", _mc_exc)
        _market_ctx_map = {}
    await bot._check_funding_arb(_market_ctx_map)
    # Умный предфильтр: из 100 монет выбираем 30 лучших кандидатов
    # по быстрым метрикам — до отправки в AI
    _ai_limit = int(os.getenv("AI_MAX_SYMBOLS", "30"))
    ai_snapshots = bot._prescore_snapshots(
        bot._quality_filter(snapshots), regimes, _ai_limit
    )
    bot._publish_event("cycle", {"state": "scan", "used": _open, "max": _max})
    bot._publish_event(
        "radar.stage",
        {
            "stage": "prescore",
            "symbols": [s.get("symbol") for s in ai_snapshots],
        },
    )
    recs = await bot._generate_signals(
        ai_snapshots,
        balance,
        regimes,
        market_data,
        market_context=_market_ctx_map,
    )

    # БАГ 1: строим signals_map из реальных рекомендаций (после combine),
    # а не из снэпшотов, где top_signal/market_context не существуют.
    _signals_map = {r["symbol"]: r for r in recs if r.get("symbol")}
    bot._position_monitor.update_market_state(
        signals=_signals_map,
        market_ctx=_market_ctx_map,
        regime=bot._current_regime,
    )

    snap_map = {s["symbol"]: s for s in snapshots}
    for rec in recs:
        rec.setdefault("_snap", snap_map.get(rec.get("symbol")))
    return replace(ctx, recs=recs)
