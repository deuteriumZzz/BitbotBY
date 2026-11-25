"""Надёжность исполнения: чистая логика без сети.

Машина состояний ордера — в `bitbot.domain.order_states`.

1. Сравнение позиций бота и биржи (`diff_positions`): позиции без наблюдения
   (нет мониторинга SL/TP — самый опасный случай), «призраки», расхождения
   объёма и стороны. Символы с ордером «в полёте» (плейсхолдер `None`) не
   сравниваются: исполнитель ещё не закончил.
2. `ConfirmTracker`: проблема считается подтверждённой, только если видна на двух
   проверках подряд, — иначе гонка с открытием/закрытием даёт ложные тревоги.

Модуль ничего не отправляет и не меняет: решения принимает вызывающий код.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Set, Tuple

# ── Сверка позиций ────────────────────────────────────────────────────────────

_SIDE = {"long": "buy", "short": "sell", "buy": "buy", "sell": "sell"}


def normalize_symbol(symbol: str) -> str:
    """'BTC/USDT:USDT' → 'BTC/USDT'."""
    return str(symbol).split(":")[0]


@dataclass(frozen=True)
class ReconcileDiff:
    # на бирже есть, бот не наблюдает (нет мониторинга SL/TP)
    untracked: Tuple[str, ...] = ()
    stale: Tuple[str, ...] = ()  # бот наблюдает, на бирже нет
    qty_mismatch: Tuple[Tuple[str, float, float], ...] = ()  # (символ, бот, биржа)
    side_mismatch: Tuple[str, ...] = ()

    def keys(self) -> Set[str]:
        """Устойчивые ключи проблем — для подтверждения на нескольких проверках."""
        out = {f"untracked:{s}" for s in self.untracked}
        out |= {f"stale:{s}" for s in self.stale}
        out |= {f"qty:{s}" for s, _b, _e in self.qty_mismatch}
        out |= {f"side:{s}" for s in self.side_mismatch}
        return out

    @property
    def empty(self) -> bool:
        return not self.keys()


def diff_positions(
    monitored: Dict[str, Any],
    exchange_positions: Iterable[Dict[str, Any]],
    qty_tolerance: float = 0.02,
) -> ReconcileDiff:
    """Сравнивает позиции бота (`_monitored`) с позициями биржи (ccxt)."""
    active: Dict[str, Dict[str, Any]] = {}
    for p in exchange_positions:
        try:
            qty = float(p.get("contracts") or 0)
        except (TypeError, ValueError):
            continue
        if qty > 0:
            active[normalize_symbol(p.get("symbol", ""))] = p

    in_flight = {s for s, pos in monitored.items() if pos is None}
    tracked = {s: pos for s, pos in monitored.items() if pos is not None}

    untracked = sorted(s for s in active if s not in monitored)
    stale = sorted(s for s in tracked if s not in active)
    qty_bad: List[Tuple[str, float, float]] = []
    side_bad: List[str] = []
    for s in sorted((tracked.keys() & active.keys()) - in_flight):
        bot_qty = float(tracked[s].get("qty") or 0)
        ex_qty = float(active[s].get("contracts") or 0)
        biggest = max(bot_qty, ex_qty)
        if biggest > 0 and abs(bot_qty - ex_qty) / biggest > qty_tolerance:
            qty_bad.append((s, bot_qty, ex_qty))
        bot_side = _SIDE.get(str(tracked[s].get("side", "")).lower())
        ex_side = _SIDE.get(str(active[s].get("side", "")).lower())
        if bot_side and ex_side and bot_side != ex_side:
            side_bad.append(s)
    return ReconcileDiff(
        tuple(untracked), tuple(stale), tuple(qty_bad), tuple(side_bad)
    )


class ConfirmTracker:
    """Подтверждает проблему, увидев её на двух проверках подряд."""

    def __init__(self) -> None:
        self._previous: Set[str] = set()

    def update(self, current_keys: Set[str]) -> Set[str]:
        confirmed = current_keys & self._previous
        self._previous = set(current_keys)
        return confirmed
