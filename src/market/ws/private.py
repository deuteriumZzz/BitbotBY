"""Нормализация приватных событий (Phase 4b): ордера и исполнения без дублей и отката.

ТОЛЬКО обработка событий: подключения к приватному потоку здесь нет — оно требует
подписанной аутентификации (ключи — только из зашифрованного хранилища, изменение
безопасности — по явному решению владельца).

Ордер: статус Bybit → `OrderState`; ордер, о котором события раньше не было,
считается отправленным (`SUBMITTED`), потому что его создал бот. Событие
отбрасывается, если оно дубль (тот же ордер, статус и время обновления), пришло
позже более нового (`updatedTime` меньше) или нарушает переходы машины состояний
(например, `PartiallyFilled` после `Filled`). Исполнение: дедупликация по `execId`
в ограниченном множестве (старые id вытесняются).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

from src.core.execution_reliability import OrderState, next_state

_STATUS = {
    "Created": OrderState.SUBMITTED,
    "New": OrderState.SUBMITTED,
    "Untriggered": OrderState.SUBMITTED,
    "Triggered": OrderState.SUBMITTED,
    "PartiallyFilled": OrderState.PARTIALLY_FILLED,
    "Filled": OrderState.FILLED,
    "Cancelled": OrderState.CANCELED,
    "PartiallyFilledCanceled": OrderState.CANCELED,
    "Deactivated": OrderState.CANCELED,
    "Rejected": OrderState.REJECTED,
}


class OrderEvents:
    def __init__(self, max_orders: int = 5000) -> None:
        self._orders: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.max_orders = max_orders
        self.dropped: Dict[str, int] = {
            "duplicate": 0, "out_of_order": 0, "invalid_transition": 0, "unknown_status": 0,
        }

    def state_of(self, order_id: str) -> Optional[OrderState]:
        o = self._orders.get(order_id)
        return o["state"] if o else None

    def on_event(self, ev: Dict[str, Any]) -> Tuple[Optional[OrderState], str]:
        """(состояние после события, причина отбрасывания или 'accepted')."""
        incoming = _STATUS.get(str(ev.get("orderStatus")))
        if incoming is None:
            self.dropped["unknown_status"] += 1
            return None, "unknown_status"
        oid = str(ev["orderId"])
        t = int(ev.get("updatedTime") or 0)
        cur = self._orders.get(oid)
        if cur is None:
            cur = {"state": OrderState.SUBMITTED, "t": 0, "keys": set()}
            self._orders[oid] = cur
            while len(self._orders) > self.max_orders:
                self._orders.popitem(last=False)
        key = (str(ev.get("orderStatus")), t)
        if key in cur["keys"]:
            self.dropped["duplicate"] += 1
            return cur["state"], "duplicate"
        if t and t < cur["t"]:
            self.dropped["out_of_order"] += 1
            return cur["state"], "out_of_order"
        state, ok = next_state(cur["state"], incoming)
        if not ok and incoming != cur["state"]:
            self.dropped["invalid_transition"] += 1
            return cur["state"], "invalid_transition"
        cur["keys"].add(key)
        cur["state"], cur["t"] = state, max(cur["t"], t)
        return state, "accepted"


class ExecutionDedup:
    def __init__(self, capacity: int = 10_000) -> None:
        self._seen: "OrderedDict[str, None]" = OrderedDict()
        self.capacity = capacity
        self.duplicates = 0

    def is_new(self, exec_id: str) -> bool:
        if exec_id in self._seen:
            self.duplicates += 1
            return False
        self._seen[exec_id] = None
        while len(self._seen) > self.capacity:
            self._seen.popitem(last=False)
        return True
