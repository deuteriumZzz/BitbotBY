"""Биржа для paper-режима: те же торговые вызовы, что у Bybit, исполнение — симуляция.

BybitAPI в paper передаёт сюда четыре торговых вызова: рыночный ордер,
биржевые SL/TP, отмену и плечо. Поэтому исполнитель и монитор позиций в paper
идут тем же кодом, что в live, — ошибки закрытия и защиты видны уже в paper.
Рыночные данные (цены, свечи, стакан) по-прежнему настоящие, с биржи.

Рыночный ордер исполняется по текущей цене биржи с поправкой на проскальзывание
против нас. Стопы лежат в книге этого объекта; срабатывание по цене — забота
монитора позиций, как и в live (биржевой SL там — страховка на случай падения
бота). Баланс paper ведёт PortfolioManager, здесь его нет.

Ручки реализма — Config.PAPER_SLIPPAGE_PCT и PAPER_LATENCY_S; по умолчанию 0:
paper ведёт себя как раньше, калибровка — по первым live-сделкам
(backtest_live_divergence).
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from bitbot.domain.order_states import OrderOutcome, Placed, Rejected

logger = logging.getLogger(__name__)

PriceSource = Callable[[str], Awaitable[Optional[float]]]


@dataclass(frozen=True)
class Realism:
    slippage_pct: float = 0.0  # доля цены, на которую исполнение хуже текущей цены
    latency_s: float = 0.0  # задержка между отправкой и исполнением


class PaperExchange:
    def __init__(self, price: PriceSource, realism: Realism = Realism()) -> None:
        self._price = price
        self.realism = realism
        self.stops: Dict[str, Dict[str, Any]] = {}  # id → стоп-ордер в книге
        self._ids = itertools.count(1)

    def _next_id(self, kind: str) -> str:
        return f"paper-{kind}-{next(self._ids)}"

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        return True

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        lock_suffix: str = "trade",
        reduce_only: bool = False,
    ) -> OrderOutcome:
        if self.realism.latency_s > 0:
            await asyncio.sleep(self.realism.latency_s)
        market = await self._price(symbol)
        if not market or market <= 0:
            return Rejected("нет цены для симуляции исполнения")
        sign = 1.0 if side == "buy" else -1.0
        fill = market * (1.0 + sign * self.realism.slippage_pct)
        order = {
            "id": self._next_id("order"),
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "amount": amount,
            "filled": amount,
            "average": fill,
            "price": fill,
            "status": "closed",
            "reduceOnly": reduce_only,
            # Комиссию paper считает сам по COMMISSION_RATE (fee 0 → её не меняют).
            "fee": {"cost": 0.0},
        }
        logger.info("[PAPER] %s %.6f %s @ %.4f", side.upper(), amount, symbol, fill)
        return Placed(order)

    async def place_exchange_sl_tp(
        self, symbol: str, close_side: str, qty: float, sl_price: float, tp_price: float
    ) -> Tuple[Optional[str], Optional[str]]:
        ids: list[Optional[str]] = []
        for kind, px in (("sl", sl_price), ("tp", tp_price)):
            if not px:
                ids.append(None)
                continue
            oid = self._next_id(kind)
            self.stops[oid] = {
                "symbol": symbol,
                "side": close_side,
                "qty": qty,
                "price": px,
                "kind": kind,
            }
            ids.append(oid)
        return ids[0], ids[1]

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        return self.stops.pop(order_id, None) is not None
