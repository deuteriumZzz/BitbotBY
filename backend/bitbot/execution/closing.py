"""Закрытие позиций на бирже: по SL/TP, принудительное, аварийное, при просадке.

Все пути идут через `_send_close`: закрытие не подтверждено биржей → биржевые
SL/TP возвращаются на весь объём, позиция остаётся в мониторе.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional

from bitbot.config.settings import Config
from bitbot.domain.order_states import Placed, Rejected
from bitbot.domain.types import PositionRecord
from bitbot.execution.booking import BookingMixin

logger = logging.getLogger(__name__)


class ClosingMixin(BookingMixin):
    # ── Закрытие на бирже ─────────────────────────────────────────────────────

    async def _send_close(
        self,
        sym: str,
        pos: PositionRecord,
        qty: float,
        label: str,
        monitored: Dict[str, PositionRecord],
        lock: asyncio.Lock,
    ) -> Optional[Dict[str, Any]]:
        """Рыночное reduce-only закрытие `qty`; не прошло — вернуть защиту позиции.

        Вызывается после отмены биржевых SL/TP. Если биржа ордер не приняла
        (Rejected) или исход неизвестен (Unknown — выяснит сверка по clientOrderId),
        SL/TP ставятся заново на весь объём, а позиция остаётся в мониторе для
        следующей попытки: иначе на бирже осталась бы открытая позиция без защиты
        и без наблюдения.

        :return: ответ биржи, если ордер принят; None — закрытие не подтверждено.
        """
        close_side = "sell" if pos.get("side", "buy") == "buy" else "buy"
        try:
            outcome = await self._api.create_order(
                sym, "market", close_side, qty, lock_suffix=label, reduce_only=True
            )
        except Exception as exc:  # noqa: BLE001 — ордер не отправлен, защиту вернуть
            outcome = Rejected(f"ошибка отправки: {exc}")
        if isinstance(outcome, Placed):
            return outcome.order
        logger.critical(
            "%s %s не прошло (%s) — восстанавливаем биржевые SL/TP,"
            " позиция остаётся в мониторе",
            label,
            sym,
            outcome,
        )
        sl = float(pos.get("stop_loss") or 0)
        tp = float(pos.get("take_profit") or 0)
        if not (sl or tp):
            return None
        try:
            re_sl, re_tp = await self._api.place_exchange_sl_tp(
                sym, close_side, pos.get("qty", 0.0), sl, tp
            )
        except Exception as exc:  # noqa: BLE001 — сообщить владельцу, не падать
            logger.critical("%s: restore SL/TP for %s failed: %s", label, sym, exc)
            await self._telegram.notify(
                f"🚨 *{sym}*: закрытие не прошло, и SL/TP восстановить не удалось — "
                f"позиция БЕЗ защиты на бирже! Требуется ручное вмешательство."
            )
            return None
        async with lock:
            if sym in monitored and monitored[sym] is not None:
                monitored[sym]["exchange_sl_id"] = re_sl
                monitored[sym]["exchange_tp_id"] = re_tp
        return None

    # ── Emergency close ───────────────────────────────────────────────────────

    async def close_all_positions(
        self,
        monitored: Dict[str, PositionRecord],
        lock: asyncio.Lock,
        reason: str = "emergency",
    ) -> None:
        """Принудительно закрывает все открытые позиции по рыночной цене.

        Используется при срабатывании защитных механизмов (дневной лимит,
        hard drawdown halt, ATR spike). Не требует достижения SL/TP.
        """
        async with lock:
            snapshot = list(monitored.items())
        for sym, pos in snapshot:
            if pos is None:
                continue
            side = pos.get("side", "buy")
            qty = pos.get("qty", 0.0)
            if qty <= 0:
                continue
            logger.warning(
                "Emergency close %s qty=%.6f side=%s reason=%s",
                sym,
                qty,
                side,
                reason,
            )
            if not Config.PAPER_TRADING:
                for oid in filter(
                    None, [pos.get("exchange_sl_id"), pos.get("exchange_tp_id")]
                ):
                    try:
                        await self._api.cancel_order(order_id=oid, symbol=sym)
                    except Exception as _e:
                        logger.debug("Cancel order %s failed: %s", oid, _e)
                if (
                    await self._send_close(sym, pos, qty, "close", monitored, lock)
                    is None
                ):
                    continue
            async with lock:
                monitored.pop(sym, None)
            try:
                price = float(await self._api.get_current_price(sym) or 0.0)
            except Exception as exc:
                price = 0.0
                logger.error("Emergency close %s: no price (%s)", sym, exc)
            await self._book_close(sym, pos, price, reason)
            logger.info("Emergency closed %s [%s]", sym, reason)

    async def close_losers_tighten_winners(
        self,
        monitored: Dict[str, PositionRecord],
        lock: asyncio.Lock,
        current_prices: Dict[str, float],
        reason: str = "hard_drawdown",
    ) -> None:
        """Hard drawdown: закрывает убыточные позиции, у прибыльных SL на breakeven.

        Убыточные (current < entry для buy, current > entry для sell) —
        закрываем по рынку, они и есть причина просадки.
        Прибыльные — переносим SL на цену входа (breakeven): позиция не может
        уйти в минус, но возьмёт профит если рынок продолжит движение в плюс.
        """
        async with lock:
            snapshot = list(monitored.items())
        for sym, pos in snapshot:
            if pos is None:
                continue
            qty = pos.get("qty", 0.0)
            if qty <= 0:
                continue
            side = pos.get("side", "buy")
            entry = pos.get("entry", 0.0)
            current = current_prices.get(sym, 0.0)
            if current <= 0 or entry <= 0:
                continue
            close_side = "sell" if side == "buy" else "buy"
            is_losing = (side == "buy" and current < entry) or (
                side == "sell" and current > entry
            )
            if is_losing:
                logger.warning(
                    "Drawdown close loser %s [%s] entry=%.4f current=%.4f reason=%s",
                    sym,
                    side,
                    entry,
                    current,
                    reason,
                )
                if not Config.PAPER_TRADING:
                    for oid in filter(
                        None, [pos.get("exchange_sl_id"), pos.get("exchange_tp_id")]
                    ):
                        try:
                            await self._api.cancel_order(order_id=oid, symbol=sym)
                        except Exception as _e:
                            logger.debug("Cancel order %s failed: %s", oid, _e)
                    closed = await self._send_close(
                        sym, pos, qty, "close", monitored, lock
                    )
                    if closed is None:
                        continue
                async with lock:
                    monitored.pop(sym, None)
                await self._book_close(sym, pos, current, reason)
            else:
                # Прибыльная позиция — переносим SL на breakeven
                tp_price = float(pos.get("take_profit") or 0)
                logger.info(
                    "Drawdown tighten winner %s [%s] entry=%.4f → SL=breakeven",
                    sym,
                    side,
                    entry,
                )
                unprotected = False
                if not Config.PAPER_TRADING:
                    for oid in filter(
                        None, [pos.get("exchange_sl_id"), pos.get("exchange_tp_id")]
                    ):
                        try:
                            await self._api.cancel_order(order_id=oid, symbol=sym)
                        except Exception as _e:
                            logger.debug("Cancel SL/TP %s failed: %s", oid, _e)
                    try:
                        new_sl_id, new_tp_id = await self._api.place_exchange_sl_tp(
                            sym, close_side, qty, entry, tp_price
                        )
                        async with lock:
                            if sym in monitored and monitored[sym] is not None:
                                monitored[sym]["stop_loss"] = entry
                                monitored[sym]["exchange_sl_id"] = new_sl_id
                                monitored[sym]["exchange_tp_id"] = new_tp_id
                    except Exception as exc:
                        logger.critical(
                            "Tighten SL for %s failed — old SL/TP already cancelled, "
                            "position has NO exchange-side protection: %s",
                            sym,
                            exc,
                        )
                        unprotected = True
                else:
                    async with lock:
                        if sym in monitored and monitored[sym] is not None:
                            monitored[sym]["stop_loss"] = entry
                if unprotected:
                    await self._telegram.notify(
                        f"🚨 *{sym}* [{side}]: не удалось выставить SL/TP на "
                        f"breakeven — позиция БЕЗ защиты на бирже! "
                        f"Требуется ручное вмешательство."
                    )
                else:
                    await self._telegram.notify(
                        f"🔒 *{sym}* [{side}]: SL перенесён на breakeven"
                        f" ${entry:.4f} (hard drawdown protection)"
                    )

    async def _force_close_at_market(
        self,
        sym: str,
        pos: PositionRecord,
        price: float,
        monitored: Dict[str, PositionRecord],
        lock: asyncio.Lock,
        reason: str = "dynamic_exit",
    ) -> None:
        """Принудительно закрывает позицию по рынку независимо от SL/TP.

        Используется при срабатывании dynamic exit (regime_flip, signal_reversal
        и т.д.) — когда нужно выйти немедленно, не дожидаясь уровней SL/TP.
        """
        side = pos.get("side", "buy")
        qty = pos.get("qty", 0.0)
        if not qty:
            async with lock:
                monitored.pop(sym, None)
            return

        close_side = "sell" if side == "buy" else "buy"
        logger.info(
            "Force-close %s [%s] qty=%.6f price=%.4f reason=%s",
            sym,
            side,
            qty,
            price,
            reason,
        )

        if not Config.PAPER_TRADING:
            sl_tp_ids = [pos.get("exchange_sl_id"), pos.get("exchange_tp_id")]
            for oid in filter(None, sl_tp_ids):
                try:
                    await self._api.cancel_order(order_id=oid, symbol=sym)
                except Exception as _e:
                    logger.debug("Cancel order %s failed: %s", oid, _e)
            if await self._send_close(sym, pos, qty, "close", monitored, lock) is None:
                return

        await self._portfolio_manager.update_portfolio(
            sym, close_side, qty, price, entry=pos["entry"], lev=pos.get("lev", 1.0)
        )
        async with lock:
            monitored.pop(sym, None)
        if self._on_position_closed:
            self._on_position_closed(sym, {"reason": reason, "price": price})

        entry = pos.get("entry", price)
        if entry:
            pnl_pct = (
                (price - entry) / entry if side == "buy" else (entry - price) / entry
            )
        else:
            pnl_pct = 0.0
        pnl_sign = "+" if pnl_pct >= 0 else ""

        if pos.get("snap") and self._online_learner:
            strategy = (pos.get("snap") or {}).get("strategy", "unknown")
            await self._online_learner.on_trade_closed(sym, side, pnl_pct, strategy)

        trade_id = pos.get("trade_id")
        pnl = None
        if trade_id:
            exit_commission = qty * price * Config.COMMISSION_RATE
            try:
                pnl = await self._trade_history.record_close(
                    trade_id=trade_id,
                    exit_price=price,
                    commission=exit_commission,
                    reason=reason,
                    funding=await self._funding_effect(sym, pos, price),
                )
            except Exception as exc:
                logger.error("record_close for trade %s failed: %s", trade_id, exc)
        self._ledger_outcome(sym, pos, price, reason, pnl)

        stats = await self._trade_history.get_summary()
        await self._telegram.notify(
            f"{'📈' if pnl_pct >= 0 else '📉'} Force-closed *{sym}* @ ${price:.4f}"
            f" ({pnl_sign}{pnl_pct*100:.2f}%) — {reason}\n"
            f"Win Rate: {stats['win_rate']:.0%}  Total PnL: ${stats['total_pnl']:+.2f}",
            kind="position_close",
        )
        self._update_circuit_breaker(side, price, entry)

    async def _check_and_close(
        self,
        sym: str,
        pos: PositionRecord,
        price: float,
        monitored: Dict[str, PositionRecord],
        lock: asyncio.Lock,
    ) -> bool:
        """
        Закрывает позицию если цена пересекла SL или TP.

        :param sym: Символ позиции.
        :param pos: Запись позиции.
        :param price: Текущая цена.
        :param monitored: Словарь открытых позиций.
        :param lock: Lock для изменения monitored.
        :return: True если позиция была закрыта.
        """
        sl = pos.get("stop_loss", 0.0)
        tp = pos.get("take_profit", 0.0)
        side = pos.get("side", "buy")
        qty = pos.get("qty", 0.0)

        triggered = False
        exit_reason = "sl"
        if side == "buy":
            if sl and price <= sl:
                logger.warning("SL hit %s: price=%.4f <= sl=%.4f", sym, price, sl)
                triggered = True
            elif tp and price >= tp:
                logger.info("TP hit %s: price=%.4f >= tp=%.4f", sym, price, tp)
                triggered, exit_reason = True, "tp"
        else:
            if sl and price >= sl:
                logger.warning("SL hit %s: price=%.4f >= sl=%.4f", sym, price, sl)
                triggered = True
            elif tp and price <= tp:
                logger.info("TP hit %s: price=%.4f <= tp=%.4f", sym, price, tp)
                triggered, exit_reason = True, "tp"

        if not triggered:
            return False
        # Стоп, подтянутый трейлингом за точку входа, — выход по трейлингу.
        _entry = pos.get("entry", 0.0)
        if (
            exit_reason == "sl"
            and _entry
            and (sl > _entry if side == "buy" else sl < _entry)
        ):
            exit_reason = "trail"

        close_side = "sell" if side == "buy" else "buy"
        # Default exit commission — overridden with actual fee in live mode.
        exit_commission = qty * price * Config.COMMISSION_RATE
        if not Config.PAPER_TRADING:
            # Шаг 1: сначала отменяем биржевые условные ордера, чтобы избежать
            # двойного закрытия (race condition между SL биржи и программным закрытием).
            exchange_sl_id = pos.get("exchange_sl_id")
            exchange_tp_id = pos.get("exchange_tp_id")
            if exchange_sl_id:
                try:
                    await self._api.cancel_order(order_id=exchange_sl_id, symbol=sym)
                except Exception as _ce:
                    logger.warning(
                        "Cancel SL order %s failed: %s — proceeding with close",
                        exchange_sl_id,
                        _ce,
                    )
            if exchange_tp_id:
                try:
                    await self._api.cancel_order(order_id=exchange_tp_id, symbol=sym)
                except Exception as _ce:
                    logger.warning(
                        "Cancel TP order %s failed: %s — proceeding with close",
                        exchange_tp_id,
                        _ce,
                    )

            # Шаг 2: программное закрытие позиции (lock_suffix="close" не конкурирует
            # с order_open:{sym} при открытии новой позиции).
            close_order = await self._send_close(
                sym, pos, qty, "close", monitored, lock
            )
            if close_order is None:
                return False

            # Используем реальную комиссию из ответа биржи (зависит от уровня аккаунта).
            fee_cost = float((close_order.get("fee") or {}).get("cost") or 0)
            if fee_cost > 0:
                exit_commission = fee_cost

        await self._portfolio_manager.update_portfolio(
            sym, close_side, qty, price, entry=pos["entry"], lev=pos.get("lev", 1.0)
        )
        async with lock:
            monitored.pop(sym, None)
        if self._on_position_closed:
            self._on_position_closed(sym, {"reason": exit_reason, "price": price})
        logger.info("Position closed: %s at %.4f", sym, price)

        # Save experience for the next SAC retraining run
        entry = pos.get("entry", price)
        if side == "buy":
            pnl_pct = (price - entry) / entry if entry else 0.0
        else:
            pnl_pct = (entry - price) / entry if entry else 0.0

        if pos.get("snap"):
            from bitbot.ml.experience_buffer import save as _exp_save  # noqa: PLC0415

            _snap = pos["snap"]
            _side = side
            _entry = entry
            _price = price
            _profile = os.environ.get("SAC_PROFILE", "")
            _exp_path = (
                "data/experiences_altcoin.jsonl"
                if _profile == "altcoin"
                else "data/experiences.jsonl"
            )
            _market_type = os.environ.get("MARKET_TYPE", Config.MARKET_TYPE)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: _exp_save(
                    snap=_snap,
                    action=_side,
                    entry_price=_entry,
                    exit_price=_price,
                    path=_exp_path,
                    market_type=_market_type,
                ),
            )

        if self._online_learner:
            strategy = (pos.get("snap") or {}).get("strategy", "unknown")
            await self._online_learner.on_trade_closed(sym, side, pnl_pct, strategy)

        trade_id = pos.get("trade_id")
        pnl = None
        if trade_id:
            try:
                pnl = await self._trade_history.record_close(
                    trade_id=trade_id,
                    exit_price=price,
                    commission=exit_commission,
                    reason=exit_reason,
                    funding=await self._funding_effect(sym, pos, price),
                )
            except Exception as exc:
                # Позиция уже удалена из монитора и закрыта на бирже.
                # Логируем потерю записи для ручной сверки — повторная попытка
                # закрыть ту же позицию хуже, чем потеря одной строки в истории.
                logger.error(
                    "record_close для trade %s не удалась: %s"
                    " — нужна ручная сверка trade_history",
                    trade_id,
                    exc,
                )
        self._ledger_outcome(sym, pos, price, exit_reason, pnl)

        stats = await self._trade_history.get_summary()
        pnl_sign = "+" if pnl_pct >= 0 else ""
        try:
            bal_val = float(self._portfolio_manager.current_balance)
            bal_line = f"\n💰 Баланс: ${bal_val:,.2f}" if Config.PAPER_TRADING else ""
        except (TypeError, ValueError):
            bal_line = ""
        await self._telegram.notify(
            f"{'📈' if pnl_pct >= 0 else '📉'} Closed *{sym}* @ ${price:.4f}"
            f"  ({pnl_sign}{pnl_pct*100:.2f}%)\n"
            f"Trades: {stats['closed_trades']}  "
            f"Win Rate: {stats['win_rate']:.0%}  "
            f"Total PnL: ${stats['total_pnl']:+.2f}"
            f"{bal_line}",
            kind="position_close",
        )

        self._update_circuit_breaker(side, price, pos.get("entry", price))
        return True
