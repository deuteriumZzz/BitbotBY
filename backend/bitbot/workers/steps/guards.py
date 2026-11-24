"""Защиты портфеля (узел «Риск»): пик equity, hard drawdown halt, дневной лимит потерь, всплеск ATR.

Каждая защита может закончить цикл сама: подождать и вернуть None.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

from bitbot.config.settings import Config
from bitbot.workers.steps.context import CycleContext

if TYPE_CHECKING:
    from bitbot.workers.bot import TradingBot

logger = logging.getLogger("bitbot.workers.bot")


async def run(bot: "TradingBot", ctx: CycleContext) -> Optional[CycleContext]:
    balance, snapshots, market_data, t0 = (
        ctx.balance,
        ctx.snapshots,
        ctx.market_data,
        ctx.t0,
    )
    # ── Обновляем пик equity ───────────────────────────────────
    equity = bot._live_equity_cache if not Config.PAPER_TRADING else balance
    if equity > bot._peak_balance:
        bot._peak_balance = equity

    # ── [П.2] Hard drawdown halt (с подтверждением) ───────────
    # Считаем по equity (включает unrealized PnL) — точнее чем
    # по свободному балансу. Подтверждение N циклов подряд.
    # В paper режиме: уведомляет но не останавливает торговлю.
    if (
        not bot._guards_off()
        and bot._peak_balance > 0
        and (bot._peak_balance - equity) / bot._peak_balance
        >= Config.MAX_DRAWDOWN_PERCENT
    ):
        bot._drawdown_consec += 1
        dd_pct = (bot._peak_balance - equity) / bot._peak_balance * 100
        confirm = bot._runtime_config.get_drawdown_confirm_cycles()
        _now = time.time()
        _dd_cool = bot._runtime_config.get_alert_cooldown_mins("drawdown") * 60
        if (
            bot._runtime_config.get_alert_enabled("drawdown")
            and _now - bot._drawdown_log_at >= _dd_cool
        ):
            bot._drawdown_log_at = _now
            logger.warning(
                "Drawdown %.1f%% from peak $%.2f — confirm %d/%d",
                dd_pct,
                bot._peak_balance,
                bot._drawdown_consec,
                confirm,
            )
        if bot._drawdown_consec >= confirm:
            if Config.PAPER_TRADING:
                logger.warning(
                    "[PAPER] Hard drawdown halt would trigger:"
                    " %.1f%% drawdown confirmed %d cycles",
                    dd_pct,
                    confirm,
                )
                if bot._runtime_config.get_alert_enabled("drawdown"):
                    await bot.telegram.edit_notify(
                        "drawdown",
                        f"⚠️ *[PAPER] Hard drawdown halt*: просадка"
                        f" {dd_pct:.1f}% от пика"
                        f" ${bot._peak_balance:.2f}"
                        f" подтверждена {confirm} цикла подряд.\n"
                        f"В live режиме: позиции закрыты,"
                        f" пауза {Config.DRAWDOWN_HALT_HOURS:.0f}ч.",
                    )
                bot._drawdown_consec = 0
            else:
                halt_secs = Config.DRAWDOWN_HALT_HOURS * 3600
                logger.critical(
                    "Hard drawdown halt: %.1f%% confirmed %d cycles"
                    " — closing losers, tightening winners, pausing %.0fh",
                    dd_pct,
                    confirm,
                    Config.DRAWDOWN_HALT_HOURS,
                )
                _dd_prices = {
                    snap["symbol"]: float(snap["price"]) for snap in snapshots if snap
                }
                await bot._position_monitor.close_losers_tighten_winners(
                    bot._monitored,
                    bot._monitored_lock,
                    _dd_prices,
                    reason="hard_drawdown",
                )
                await bot.telegram.notify(
                    f"🚨 *Hard drawdown halt*: просадка {dd_pct:.1f}%"
                    f" от пика ${bot._peak_balance:.2f}"
                    f" подтверждена {confirm} цикла подряд.\n"
                    f"Убыточные позиции закрыты."
                    f" Прибыльные: SL перенесён на breakeven.\n"
                    f"Новые входы заблокированы на"
                    f" {Config.DRAWDOWN_HALT_HOURS:.0f}ч."
                )
                bot._drawdown_consec = 0
                bot._publish_event(
                    "guard",
                    {
                        "kind": "drawdown_halt",
                        "active": True,
                        "detail": f"Просадка {dd_pct:.1f}% от пика",
                        "until": time.time() + halt_secs,
                    },
                )
                bot._entries_blocked = "просадка: новые входы остановлены"
                await bot._pause(halt_secs)
                bot._entries_blocked = ""
                bot._publish_event("guard", {"kind": "drawdown_halt", "active": False})
                bot._peak_balance = await bot._get_balance_usdt()
                return None
        else:
            await bot._sleep_rest_of_cycle(t0)
            return None
    else:
        bot._drawdown_consec = 0

    # ── [П.3] Дневной лимит потерь ────────────────────────────
    # Считаем по equity (кэш + нереализованный P&L), а не по
    # свободному балансу — иначе открытие позиций само по себе
    # выглядит как «убыток» и лимит срабатывает мгновенно.
    # При срабатывании: блокируем новые входы, открытые позиции
    # держим до их SL/TP (PositionMonitor управляет ими независимо).
    # В paper режиме: уведомляет но не останавливает торговлю.
    if Config.PAPER_TRADING:
        _paper_prices = {
            snap["symbol"]: float(snap["price"]) for snap in snapshots if snap
        }
        _equity_for_limit = await bot.portfolio_manager.get_portfolio_value(
            _paper_prices
        )
    else:
        _equity_for_limit = bot._live_equity_cache
    if not bot._guards_off() and not bot.risk_manager.check_daily_loss_limit(
        _equity_for_limit
    ):
        now_utc = datetime.utcnow()
        midnight = (now_utc + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        sleep_secs = (midnight - now_utc).total_seconds()
        if Config.PAPER_TRADING:
            _now = time.time()
            _dl_cool = bot._runtime_config.get_alert_cooldown_mins("daily_limit") * 60
            _dl_enabled = bot._runtime_config.get_alert_enabled("daily_limit")
            if _dl_enabled and _now - bot._daily_limit_notified_at >= _dl_cool:
                logger.warning(
                    "[PAPER] Daily loss limit would trigger:"
                    " equity $%.2f, would pause %.0f s",
                    _equity_for_limit,
                    sleep_secs,
                )
            if _dl_enabled and _now - bot._daily_limit_notified_at >= _dl_cool:
                bot._daily_limit_notified_at = _now
                await bot.telegram.edit_notify(
                    "daily_limit",
                    f"⚠️ *[PAPER] Дневной лимит потерь*:"
                    f" equity ${_equity_for_limit:.2f}.\n"
                    f"В live режиме: новые входы заблокированы до 00:00 UTC"
                    f" ({int(sleep_secs // 3600)}ч"
                    f" {int((sleep_secs % 3600) // 60)}м).",
                )
        else:
            await bot.telegram.notify(
                "⛔ Дневной лимит потерь достигнут. "
                f"Equity: ${_equity_for_limit:.2f}.\n"
                f"Новые входы заблокированы.\n"
                f"Торговля возобновится автоматически при восстановлении"
                f" equity или в 00:00 UTC"
                f" (через {int(sleep_secs // 3600)}ч"
                f" {int((sleep_secs % 3600) // 60)}м).\n"
                f"Открытые позиции работают по своим SL/TP."
            )
            logger.warning(
                "Daily loss limit hit — blocking new entries,"
                " checking recovery every 60s until midnight UTC",
            )
            bot._publish_event(
                "guard",
                {
                    "kind": "daily_limit",
                    "active": True,
                    "detail": "Дневной лимит убытка достигнут",
                    "until": time.time() + sleep_secs,
                },
            )
            _check_interval = 60
            _elapsed = 0.0
            bot._entries_blocked = "дневной лимит убытка"
            while _elapsed < sleep_secs and bot.is_running:
                await bot._pause(_check_interval)
                _elapsed += _check_interval
                _recovery_eq = await bot._get_balance_usdt()
                _recovery_eq = bot._live_equity_cache
                if bot.risk_manager.check_daily_loss_limit(_recovery_eq):
                    logger.info(
                        "Daily loss limit recovered:"
                        " equity $%.2f — resuming trading",
                        _recovery_eq,
                    )
                    await bot.telegram.notify(
                        f"✅ Equity восстановилась до"
                        f" ${_recovery_eq:.2f}.\n"
                        f"Торговля возобновлена досрочно."
                    )
                    break
            bot._entries_blocked = ""
            bot._publish_event("guard", {"kind": "daily_limit", "active": False})
            return None

    # ── [П.1] ATR spike / volatility circuit breaker ──────────
    # Если ATR вырос в ATR_SPIKE_MULT раз от нормы → закрыть
    # позиции и пропустить этот цикл (следующий через 30 с).
    if not bot._guards_off() and bot._is_atr_spike(market_data):
        _now = time.time()
        _atr_cool = bot._runtime_config.get_alert_cooldown_mins("atr_spike") * 60
        if _now - bot._atr_spike_alerted_at >= _atr_cool:
            bot._atr_spike_alerted_at = _now
            logger.warning("Volatility circuit breaker triggered — skipping cycle")
            await bot._close_all_open_positions("atr_spike")
            bot._publish_event(
                "guard",
                {
                    "kind": "atr_spike",
                    "active": True,
                    "detail": ("Резкий всплеск волатильности — позиции закрыты"),
                },
            )
            if bot._runtime_config.get_alert_enabled("atr_spike"):
                await bot.telegram.edit_notify(
                    "atr_spike",
                    "⚡ *Volatility circuit breaker*: скачок ATR в "
                    f"{Config.ATR_SPIKE_MULT:.0f}x от нормы.\n"
                    "Позиции закрыты, цикл пропущен.",
                )
        await bot._sleep_rest_of_cycle(t0)
        return None
    return ctx
