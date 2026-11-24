"""Жизненный цикл ордера: состояния, допустимые переходы, разбор ответа ccxt.

Событие, нарушающее переходы (повтор уведомления, откат из финального
состояния), отбрасывается — оно не должно менять состояние ордера. Чистая
логика без сети: ей пользуются адаптер биржи, приватный WS и сверка исполнения.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Set, Tuple


class OrderState(str, Enum):
    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"


TERMINAL = frozenset(
    {OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED, OrderState.EXPIRED}
)

_TRANSITIONS: Dict[OrderState, Set[OrderState]] = {
    OrderState.CREATED: {OrderState.SUBMITTED, OrderState.REJECTED},
    OrderState.SUBMITTED: {
        OrderState.PARTIALLY_FILLED,
        OrderState.FILLED,
        OrderState.CANCELED,
        OrderState.REJECTED,
        OrderState.EXPIRED,
    },
    OrderState.PARTIALLY_FILLED: {
        OrderState.PARTIALLY_FILLED,  # ещё одно частичное исполнение
        OrderState.FILLED,
        OrderState.CANCELED,  # остаток отменён
        OrderState.EXPIRED,
    },
}


def can_transition(current: OrderState, incoming: OrderState) -> bool:
    return incoming in _TRANSITIONS.get(current, set())


def next_state(current: OrderState, incoming: OrderState) -> Tuple[OrderState, bool]:
    """(состояние, принято ли событие). Недопустимое событие не меняет состояние."""
    if can_transition(current, incoming):
        return incoming, True
    return current, False


def classify_order(order: Optional[Dict[str, Any]]) -> OrderState:
    """Состояние по ответу ccxt (`status`, `filled`)."""
    if not order:
        return OrderState.UNKNOWN
    status = str(order.get("status") or "").lower()
    try:
        filled = float(order.get("filled") or 0)
    except (TypeError, ValueError):
        filled = 0.0
    if status == "closed":
        return OrderState.FILLED
    if status in ("canceled", "cancelled"):
        return OrderState.CANCELED
    if status == "rejected":
        return OrderState.REJECTED
    if status == "expired":
        return OrderState.EXPIRED
    if status == "open":
        return OrderState.PARTIALLY_FILLED if filled > 0 else OrderState.SUBMITTED
    return OrderState.UNKNOWN
