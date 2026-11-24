"""Сопровождение (узел «Сопровождение»): слоты заняты — сигналы только по открытым
позициям и барометрам BTC/ETH для раннего выхода, и цикл заканчивается.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Optional

from bitbot.workers.health_server import cycles_counter
from bitbot.workers.steps.context import CycleContext

if TYPE_CHECKING:
    from bitbot.workers.bot import TradingBot

logger = logging.getLogger("bitbot.workers.bot")


async def run(bot: "TradingBot", ctx: CycleContext) -> Optional[CycleContext]:
    snapshots, market_data, balance, t0 = (
        ctx.snapshots,
        ctx.market_data,
        ctx.balance,
        ctx.t0,
    )
    loop = asyncio.get_running_loop()
    async with bot._monitored_lock:
        _open_syms = [s for s, p in bot._monitored.items() if p is not None]
    _open = len(_open_syms)
    _max = bot._runtime_config.get_max_positions()
    _slots = _max - _open

    if _slots <= 0:
        bot._publish_event("cycle", {"state": "monitor", "used": _open, "max": _max})
        # РЕЖИМ МОНИТОРИНГА: слоты заняты — сканируем только открытые монеты
        # Это даёт PositionMonitor актуальные сигналы для раннего выхода
        # (разворот, смена режима, funding squeeze) без лишних токенов
        logger.info(
            "Positions full (%d/%d) — monitoring mode (open: %s)",
            _open,
            _max,
            ", ".join(_open_syms),
        )
        # BTC и ETH всегда добавляем как барометры рынка:
        # BTC = главный драйвер (~85% корреляция), ETH = altcoin сектор
        _monitor_syms = list(_open_syms)
        for _barometer in ("BTC/USDT", "ETH/USDT"):
            if _barometer not in _monitor_syms and _barometer in market_data:
                _monitor_syms.append(_barometer)
        _mon_snaps = [s for s in snapshots if s.get("symbol") in _monitor_syms]
        if _mon_snaps:
            _mon_mdata = {
                sym: market_data[sym] for sym in _monitor_syms if sym in market_data
            }
            regimes = await bot._detect_regimes(_mon_mdata)
            _prices = {
                sym: df["close"].iloc[-1]
                for sym, df in _mon_mdata.items()
                if df is not None and not df.empty
            }
            try:
                _market_ctx_map = await bot._market_context.get_context_for_symbols(
                    list(_mon_mdata.keys()), _prices
                )
            except Exception as _mc_exc:
                logger.warning("MarketContext fetch failed: %s", _mc_exc)
                _market_ctx_map = {}
            await bot._check_funding_arb(_market_ctx_map)
            _mon_recs = await bot._generate_signals(
                _mon_snaps,
                balance,
                regimes,
                _mon_mdata,
                market_context=_market_ctx_map,
            )
            _signals_map = {r["symbol"]: r for r in _mon_recs if r.get("symbol")}
            bot._position_monitor.update_market_state(
                signals=_signals_map,
                market_ctx=_market_ctx_map,
                regime=bot._current_regime,
            )
        bot._save_monitored_state()
        bot._last_cycle_at = time.time()
        await bot._check_silent_death()
        cycles_counter.inc()
        elapsed = loop.time() - t0
        _interval = bot._runtime_config.get_trading_interval()
        await bot._pause(max(0, _interval - elapsed))
        return None
    return replace(ctx, open_positions=_open, max_positions=_max)
