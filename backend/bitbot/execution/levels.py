"""Уровни позиции: перенос SL/TP, трейлинг-стоп, частичная фиксация прибыли (Partial TP).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional, Tuple

from bitbot.config.settings import Config
from bitbot.domain.types import PositionRecord
from bitbot.execution.closing import ClosingMixin

logger = logging.getLogger(__name__)


class LevelsMixin(ClosingMixin):
    # ── Helpers ───────────────────────────────────────────────────────────────

    # ── Partial TP (УЛУЧШЕНИЕ 3) ─────────────────────────────────────────────

    async def _check_partial_tp(
        self,
        sym: str,
        pos: PositionRecord,
        price: float,
        monitored: Dict[str, PositionRecord],
        lock: asyncio.Lock,
    ) -> PositionRecord:
        """Частичная фиксация прибыли при достижении PARTIAL_TP_TRIGGER пути к TP.

        При срабатывании:
          - закрывается PARTIAL_TP_FRACTION позиции (paper: только обновление qty)
          - SL переносится на breakeven (entry price)
          - выставляется флаг partial_tp_triggered = True

        :return: Обновлённая запись позиции.
        """
        try:
            enabled = bool(Config.PARTIAL_TP_ENABLED)
            trigger = float(Config.PARTIAL_TP_TRIGGER)
            fraction = float(Config.PARTIAL_TP_FRACTION)
        except (TypeError, ValueError):
            return pos
        if not enabled:
            return pos
        if pos.get("partial_tp_triggered", False):
            return pos

        side = pos.get("side", "buy")
        entry = pos.get("entry", 0.0)
        tp = pos.get("take_profit", 0.0)
        qty = pos.get("qty", 0.0)

        if not tp or not entry or not qty:
            return pos

        # Вычисляем прогресс цены к TP
        if side == "buy":
            progress = (price - entry) / (tp - entry) if tp > entry else 0.0
        else:
            progress = (entry - price) / (entry - tp) if entry > tp else 0.0

        if progress < trigger:
            return pos

        updated, remaining_unprotected = await self.close_fraction(
            sym,
            pos,
            fraction,
            monitored,
            lock,
            sl_after=entry,
            label="partial_tp",
            price=price,
        )
        if updated is None:
            return pos
        async with lock:
            if sym in monitored and monitored[sym] is not None:
                monitored[sym]["partial_tp_triggered"] = True

        logger.info(
            "Partial TP triggered for %s: closed %.1f%% (%.6f units),"
            " SL moved to breakeven %.4f",
            sym,
            fraction * 100,
            qty * fraction,
            entry,
        )
        # При remaining_unprotected критический алерт уже отправлен —
        # не дублируем его успокаивающим сообщением про "SL → breakeven".
        if not remaining_unprotected:
            await self._telegram.notify(
                f"Partial TP *{sym}*: closed {fraction:.0%},"
                f" SL → breakeven ${entry:.4f}",
                kind="position_close",
            )
        return {**updated, "partial_tp_triggered": True}

    async def close_fraction(
        self,
        sym: str,
        pos: PositionRecord,
        fraction: float,
        monitored: Dict[str, PositionRecord],
        lock: asyncio.Lock,
        sl_after: Optional[float] = None,
        label: str = "partial",
        price: Optional[float] = None,
    ) -> Tuple[Optional[PositionRecord], bool]:
        """Закрывает долю позиции по рынку и переставляет SL/TP на остаток.

        Общий путь для автоматической частичной фиксации и ручного закрытия
        из приложения. sl_after — новый SL для остатка (None — прежний).

        :return: (обновлённая запись или None, если ордер не прошёл;
            True, если остаток остался без защиты на бирже).
        """
        side = pos.get("side", "buy")
        qty = pos.get("qty", 0.0)
        partial_qty = qty * fraction
        new_qty = qty * (1.0 - fraction)
        new_sl = float(pos.get("stop_loss") or 0) if sl_after is None else sl_after

        close_side = "sell" if side == "buy" else "buy"
        new_sl_id: str | None = None
        new_tp_id: str | None = None
        remaining_unprotected = False
        if not Config.PAPER_TRADING:
            # Отменяем биржевые SL/TP перед частичным закрытием —
            # иначе биржа попытается закрыть полный объём при достижении TP.
            exchange_sl_id = pos.get("exchange_sl_id")
            exchange_tp_id = pos.get("exchange_tp_id")
            for oid, kind in ((exchange_sl_id, "SL"), (exchange_tp_id, "TP")):
                if oid:
                    try:
                        await self._api.cancel_order(order_id=oid, symbol=sym)
                    except Exception as exc:
                        logger.warning(
                            "%s: cancel %s %s failed: %s", label, kind, oid, exc
                        )
            if (
                await self._send_close(sym, pos, partial_qty, label, monitored, lock)
                is None
            ):
                return None, False
            # Выставляем SL и TP для оставшегося объёма
            tp_price = float(pos.get("take_profit") or 0)
            if new_sl or tp_price:
                try:
                    new_sl_id, new_tp_id = await self._api.place_exchange_sl_tp(
                        sym, close_side, new_qty, new_sl, tp_price
                    )
                except Exception as exc:
                    logger.critical(
                        "%s: re-place SL/TP for %s failed — remaining "
                        "position has NO exchange-side protection: %s",
                        label,
                        sym,
                        exc,
                    )
                    remaining_unprotected = True
                    await self._telegram.notify(
                        f"🚨 *{sym}*: часть позиции закрыта, но новый SL/TP на "
                        f"оставшийся объём не выставлен — позиция БЕЗ защиты "
                        f"на бирже! Требуется ручное вмешательство."
                    )

        async with lock:
            if sym in monitored and monitored[sym] is not None:
                monitored[sym]["qty"] = new_qty
                monitored[sym]["stop_loss"] = new_sl
                if not Config.PAPER_TRADING:
                    monitored[sym]["exchange_sl_id"] = new_sl_id
                    monitored[sym]["exchange_tp_id"] = new_tp_id
        updated = {**pos, "qty": new_qty, "stop_loss": new_sl}
        if not Config.PAPER_TRADING:
            updated.update(exchange_sl_id=new_sl_id, exchange_tp_id=new_tp_id)
        await self._book_partial(sym, pos, partial_qty, close_side, price)
        return updated, remaining_unprotected  # type: ignore[return-value]

    async def _book_partial(
        self,
        sym: str,
        pos: PositionRecord,
        qty: float,
        close_side: str,
        price: Optional[float],
    ) -> None:
        """Учёт частичного закрытия: баланс портфеля и результат в истории.

        Раньше доля закрывалась на бирже, но в баланс и историю не попадала:
        деньги «терялись», а итоговое закрытие считалось на полный первоначальный
        объём.
        """
        if price is None:
            try:
                price = float(await self._api.get_current_price(sym) or 0.0)
            except Exception as exc:
                logger.error("Partial close %s: no price (%s)", sym, exc)
                price = 0.0
        if not price or price <= 0:
            logger.error("Partial close of %s not booked: unknown exit price", sym)
            return
        await self._portfolio_manager.update_portfolio(
            sym, close_side, qty, price, entry=pos["entry"], lev=pos.get("lev", 1.0)
        )
        trade_id = pos.get("trade_id")
        if not trade_id:
            return
        try:
            await self._trade_history.record_partial_close(
                trade_id, qty, price, qty * price * Config.COMMISSION_RATE
            )
        except Exception as exc:
            logger.error("record_partial_close for trade %s failed: %s", trade_id, exc)

    async def update_levels(
        self,
        sym: str,
        pos: PositionRecord,
        sl: float,
        tp: float,
        monitored: Dict[str, PositionRecord],
        lock: asyncio.Lock,
    ) -> PositionRecord:
        """Меняет SL/TP позиции; в Live — перевыставляет ордера на бирже.

        Новые ордера ставятся до отмены старых: позиция ни на миг не остаётся
        без защиты. Если новые не встали — старые не трогаем и бросаем ошибку.
        """
        updated: PositionRecord = {**pos, "stop_loss": sl, "take_profit": tp}
        if not Config.PAPER_TRADING:
            close_side = "sell" if pos.get("side", "buy") == "buy" else "buy"
            new_sl_id, new_tp_id = await self._api.place_exchange_sl_tp(
                sym, close_side, float(pos.get("qty") or 0), sl, tp
            )
            for oid in (pos.get("exchange_sl_id"), pos.get("exchange_tp_id")):
                if oid:
                    try:
                        await self._api.cancel_order(order_id=oid, symbol=sym)
                    except Exception as exc:
                        logger.warning("update_levels: cancel %s failed: %s", oid, exc)
            updated["exchange_sl_id"] = new_sl_id
            updated["exchange_tp_id"] = new_tp_id
        async with lock:
            if sym in monitored and monitored[sym] is not None:
                monitored[sym] = {
                    **monitored[sym],
                    "stop_loss": updated["stop_loss"],
                    "take_profit": updated["take_profit"],
                    "exchange_sl_id": updated.get("exchange_sl_id"),
                    "exchange_tp_id": updated.get("exchange_tp_id"),
                }
        return updated

    def _apply_trailing_stop(
        self,
        sym: str,
        pos: PositionRecord,
        price: float,
    ) -> PositionRecord:
        """
        Подтягивает SL к текущей цене на шаг ATR × multiplier.

        ATR-based шаг адаптируется к волатильности: узкий на спокойном рынке
        (меньше преждевременных стопов), широкий на волатильном (меньше шумовых).

        Returns a new pos dict if changed, same pos object if unchanged.
        The caller must write the result to monitored under the lock.

        :param sym: Символ позиции.
        :param pos: Текущая запись позиции.
        :param price: Текущая рыночная цена.
        :return: Обновлённая запись позиции (новый объект если изменилась).
        """
        atr = pos.get("atr", 0.0)
        mult = Config.TRAILING_STOP_ATR_MULT
        if not (atr and atr > 0 and mult > 0):
            return pos
        if pos.get("trailing") is False:  # выключен для этой позиции в приложении
            return pos

        side = pos.get("side", "buy")
        current_sl = pos.get("stop_loss", 0.0)

        if side == "buy":
            trail = price - atr * mult
            if trail > current_sl:
                return {**pos, "stop_loss": trail}
        else:
            trail = price + atr * mult
            if trail < pos.get("stop_loss", float("inf")):
                return {**pos, "stop_loss": trail}

        return pos

    async def _tighten_sl(
        self,
        sym: str,
        pos: PositionRecord,
        price: float,
        monitored: Dict[str, PositionRecord],
        lock: asyncio.Lock,
    ) -> PositionRecord:
        """Подтягивает SL к текущей цене на 1.5×ATR при смене режима.

        Не закрывает позицию сразу — даёт шанс на отскок, но ограничивает
        дальнейший убыток. Если новый SL хуже текущего — не трогаем.
        """
        side = pos.get("side", "buy")
        current_sl = pos.get("stop_loss", 0.0)
        atr = pos.get("atr") or 0.0
        step = atr * 1.5 if atr > 0 else price * 0.015  # fallback 1.5%

        if side == "buy":
            new_sl = price - step
            if current_sl and new_sl <= current_sl:
                return pos  # уже тише — не двигаем
        else:
            new_sl = price + step
            if current_sl and new_sl >= current_sl:
                return pos

        logger.info(
            "Tighten SL %s [%s]: %.4f → %.4f (price=%.4f atr=%.4f)",
            sym,
            side,
            current_sl,
            new_sl,
            price,
            atr,
        )
        async with lock:
            if sym in monitored and monitored[sym] is not None:
                monitored[sym] = {**monitored[sym], "stop_loss": new_sl}
        return {**pos, "stop_loss": new_sl}
