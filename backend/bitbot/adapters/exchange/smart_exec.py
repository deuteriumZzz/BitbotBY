"""Умное исполнение (ADR-002 п.3–4, реестр идей): TWAP, Post-Only chase, частичные исполнения, резерв маржи.

Чистая логика без сети: решения принимают функции, ордера отправляет вызывающий код. В бот
НЕ подключено — сегодня вход рыночный. Включать только после замера проскальзывания на Paper
(ТЗ §4/5): без данных неизвестно, есть ли что экономить на лимитных входах. Пороги — предложения.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ── TWAP ──────────────────────────────────────────────────────────────────────
def twap_plan(
    total_qty: float, slices: int, duration_s: float, step: float, min_qty: float
) -> List[Dict[str, float]]:
    """Нарезка ордера: равные доли, округлённые вниз до `step`, остаток — в последнюю; доли < `min_qty` сливаются.

    Сумма срезов равна `total_qty` с точностью до шага. Время — равномерно в течение `duration_s`.
    """
    if total_qty <= 0 or slices < 1 or step <= 0 or duration_s < 0:
        raise ValueError("некорректные параметры TWAP")
    total = math.floor(total_qty / step + 1e-9) * step
    if total < min_qty:
        raise ValueError("объём меньше минимального ордера")
    n = max(
        1, min(slices, int(total / max(min_qty, step) + 1e-9))
    )  # не дробим ниже минимума
    base = math.floor(total / n / step + 1e-9) * step
    qtys = [base] * n
    qtys[-1] = round(total - base * (n - 1), 12)
    gap = duration_s / n if n > 1 else 0.0
    return [
        {"at_s": round(i * gap, 6), "qty": round(q, 12)} for i, q in enumerate(qtys)
    ]


# ── Post-Only chase ───────────────────────────────────────────────────────────
@dataclass
class ChaseState:
    started_at: float
    order_price: Optional[float] = None
    reprices: int = 0


@dataclass(frozen=True)
class ChaseParams:
    reprice_ticks: int = 2  # цена ушла от нашей заявки на столько тиков — переставляем
    max_reprices: int = 5
    max_wait_s: float = 30.0
    allow_market_fallback: bool = True


def chase_next(
    side: str,
    st: ChaseState,
    best_bid: float,
    best_ask: float,
    tick: float,
    now: float,
    prm: ChaseParams = ChaseParams(),
) -> Dict[str, Any]:
    """Следующее действие: place / hold / reprice / market_fallback / give_up.

    Покупаем по лучшему bid, продаём по лучшему ask (Post-Only: не пересекаем стакан).
    Исчерпан лимит переставлений или ожидания — рыночный ордер (если разрешён) или отказ.
    """
    if side not in ("buy", "sell"):
        raise ValueError("side buy|sell")
    target = best_bid if side == "buy" else best_ask
    if st.order_price is None:
        st.order_price = target
        return {"action": "place", "price": target}
    if now - st.started_at >= prm.max_wait_s or st.reprices >= prm.max_reprices:
        return {
            "action": "market_fallback" if prm.allow_market_fallback else "give_up",
            "price": None,
        }
    away = (
        (target - st.order_price) / tick
        if side == "buy"
        else (st.order_price - target) / tick
    )
    if away >= prm.reprice_ticks:  # рынок убежал от нашей заявки
        st.order_price, st.reprices = target, st.reprices + 1
        return {"action": "reprice", "price": target}
    return {"action": "hold", "price": st.order_price}


# ── Частичные исполнения ──────────────────────────────────────────────────────
TERMINAL = {"FILLED", "CANCELED", "REJECTED", "EXPIRED"}


def partial_fill_decision(
    target_qty: float,
    filled_qty: float,
    status: str,
    age_s: float,
    max_age_s: float = 60.0,
    min_fill_ratio: float = 0.5,
) -> Dict[str, Any]:
    """Что делать с частично исполненным ордером. `protect_qty` — объём, который обязан быть под SL/TP.

    Принцип: защита всегда равна ФАКТИЧЕСКИ исполненному объёму, а не запрошенному.
    """
    if target_qty <= 0 or filled_qty < 0:
        raise ValueError("некорректные объёмы")
    ratio = filled_qty / target_qty
    if ratio >= 0.999 or status == "FILLED":
        return {"action": "done", "protect_qty": filled_qty, "cancel_remainder": False}
    if status in TERMINAL:  # закрыт биржей, исполнено частично
        return {
            "action": "accept_partial" if filled_qty > 0 else "nothing_filled",
            "protect_qty": filled_qty,
            "cancel_remainder": False,
        }
    if age_s < max_age_s:
        return {"action": "wait", "protect_qty": filled_qty, "cancel_remainder": False}
    if ratio >= min_fill_ratio:
        return {
            "action": "accept_partial",
            "protect_qty": filled_qty,
            "cancel_remainder": True,
        }
    return {
        "action": "cancel_and_close_filled",
        "protect_qty": filled_qty,
        "cancel_remainder": True,
    }


# ── Резерв маржи под ордера «в полёте» ────────────────────────────────────────
class InflightMargin:
    """Маржа, зарезервированная под отправленные, но не исполненные ордера (лимитные входы).

    Нужна при лимитном входе: свободный баланс до исполнения ещё не уменьшился, и без резерва два
    входа подряд могут вместе превысить доступную маржу. Идемпотентна по `order_id`.
    """

    def __init__(self) -> None:
        self._r: Dict[str, float] = {}
        self._lock = threading.Lock()

    def reserve(self, order_id: str, notional: float, leverage: float = 1.0) -> float:
        if notional < 0 or leverage <= 0:
            raise ValueError("notional ≥ 0, leverage > 0")
        with self._lock:
            self._r[order_id] = (
                notional / leverage
            )  # повтор с тем же id — заменяет, не суммирует
            return self._r[order_id]

    def release(self, order_id: str) -> None:
        with self._lock:
            self._r.pop(order_id, None)

    def reserved_total(self) -> float:
        with self._lock:
            return sum(self._r.values())

    def available(self, balance: float) -> float:
        return max(0.0, balance - self.reserved_total())

    def can_open(self, balance: float, notional: float, leverage: float = 1.0) -> bool:
        return notional / max(leverage, 1e-9) <= self.available(balance)
