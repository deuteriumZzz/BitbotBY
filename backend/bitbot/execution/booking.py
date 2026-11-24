"""Учёт закрытия позиции: баланс paper, история сделок, funding, итог в журнале решений.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from bitbot.config.settings import Config
from bitbot.domain.types import PositionRecord
from bitbot.execution.monitor_base import MonitorBase

logger = logging.getLogger(__name__)


class BookingMixin(MonitorBase):
    async def _book_close(
        self, sym: str, pos: PositionRecord, price: float, reason: str
    ) -> None:
        """Учёт аварийного закрытия: баланс Paper, история сделок, приложение.

        Без цены выхода P&L не посчитать — сделку не трогаем, только лог.
        """
        if price <= 0:
            logger.error("Close of %s not booked: unknown exit price", sym)
            return
        side = pos.get("side", "buy")
        qty = pos.get("qty", 0.0)
        close_side = "sell" if side == "buy" else "buy"
        await self._portfolio_manager.update_portfolio(
            sym, close_side, qty, price, entry=pos["entry"], lev=pos.get("lev", 1.0)
        )
        if self._on_position_closed:
            self._on_position_closed(sym, {"reason": reason, "price": price})
        trade_id = pos.get("trade_id")
        pnl = None
        if trade_id:
            try:
                pnl = await self._trade_history.record_close(
                    trade_id=trade_id,
                    exit_price=price,
                    commission=qty * price * Config.COMMISSION_RATE,
                    reason=reason,
                    funding=await self._funding_effect(sym, pos, price),
                )
            except Exception as exc:
                logger.error("record_close for trade %s failed: %s", trade_id, exc)
        self._ledger_outcome(sym, pos, price, reason, pnl)

    async def _funding_effect(self, sym: str, pos: Any, price: float) -> float:
        """Вклад funding в итог сделки (минус — заплатили). 0, пока `FUNDING_IN_PNL` выключен.

        paper — оценка по ставке и сроку удержания, live — выплаты с биржи; аномальные значения
        и недоступность данных дают 0 (закрытие позиции от этого не зависит).
        """
        if not Config.FUNDING_IN_PNL:
            return 0.0
        try:
            import time as _t  # noqa: PLC0415

            from bitbot.execution import funding as fd  # noqa: PLC0415

            opened = float(pos.get("opened_at") or 0)
            if not opened:
                return 0.0
            qty, entry = float(pos.get("qty") or 0), float(pos.get("entry") or 0)
            margin = qty * entry / max(float(pos.get("lev") or 1.0), 1.0)
            if Config.PAPER_TRADING:
                eff = fd.estimate_effect(
                    pos.get("side", "buy"),
                    qty,
                    entry,
                    price,
                    _t.time() - opened,
                    Config.PAPER_FUNDING_RATE_8H,
                )
            else:
                live = await fd.live_effect(self._api, sym, int(opened * 1000))
                if live is None:
                    return 0.0
                eff = live
            if not fd.is_sane(eff, margin):
                logger.warning(
                    "Funding %.4f for %s отклонён как аномалия (маржа %.2f)",
                    eff,
                    sym,
                    margin,
                )
                return 0.0
            return eff
        except (
            Exception
        ) as exc:  # noqa: BLE001 — учёт funding не должен ломать закрытие
            logger.debug("Funding effect failed for %s: %s", sym, exc)
            return 0.0

    def _ledger_outcome(
        self, sym: str, pos: Any, price: float, reason: str, pnl: Optional[float]
    ) -> None:
        """Пишет исход сделки в Decision Ledger (`outcome`). Только запись; сбой гасится."""
        ledger = getattr(self, "ledger", None)
        if ledger is None:
            return
        try:
            import time as _time  # noqa: PLC0415

            opened = pos.get("opened_at")
            ledger.write(
                "outcome",
                decision_id=pos.get("decision_id"),
                symbol=sym,
                side=pos.get("side"),
                entry=pos.get("entry"),
                exit=price,
                qty=pos.get("qty"),
                reason=reason,
                pnl=pnl,  # net за вычетом комиссий; None — сделка не записана в БД
                recorded=pnl is not None,
                hold_s=round(_time.time() - opened) if opened else None,
                strategy=pos.get("strategy"),
                confidence=pos.get("confidence"),
                lev=pos.get("lev"),
            )
        except Exception as exc:  # журнал не должен ломать закрытие позиции
            logger.debug("Ledger outcome failed for %s: %s", sym, exc)
