"""Мультибиржевая устойчивость (Phase 12, MASTER_SPEC §11 п.38): чистая логика без сети и ключей.

Две части, не требующие решений владельца о второй бирже:

1. `VenueHealth` — здоровье площадки по исходам API-вызовов (доля ошибок, p95
   задержки, серия ошибок). `policy()` советует только ЗАЩИТНОЕ действие:
   `block_entries` при падении. Автоматического переключения торговли на другую
   площадку здесь нет и быть не должно: у второй биржи другие цена, ликвидность,
   маржа и состояние позиций — это отдельное решение человека.
2. `hedge_plan` — плановый расчёт хеджа на второй площадке с ограничениями:
   доля капитала под хедж, доля видимой глубины стакана, потолок базисного риска.
   Это расчёт, а не ордер: исполнение хеджа в бот не подключено.

Все пороги — ПРЕДЛОЖЕНИЯ. Резервный API-ключ — вопрос безопасности и хранения ключей
(только по явному решению владельца), здесь не рассматривается.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Tuple

HEALTHY, DEGRADED, DOWN = "healthy", "degraded", "down"


class VenueHealth:
    def __init__(
        self,
        window: int = 100,
        degraded_error_rate: float = 0.10,
        down_error_rate: float = 0.50,
        down_consecutive: int = 5,
        slow_p95_ms: float = 3000.0,
        min_calls: int = 10,
    ) -> None:
        self._calls: Deque[Tuple[bool, float]] = deque(maxlen=window)
        self.consecutive_errors = 0
        self.degraded_error_rate, self.down_error_rate = (
            degraded_error_rate,
            down_error_rate,
        )
        self.down_consecutive, self.slow_p95_ms, self.min_calls = (
            down_consecutive,
            slow_p95_ms,
            min_calls,
        )

    def record(self, ok: bool, latency_ms: float = 0.0) -> None:
        self._calls.append((bool(ok), float(latency_ms)))
        self.consecutive_errors = 0 if ok else self.consecutive_errors + 1

    def stats(self) -> Dict[str, Any]:
        n = len(self._calls)
        lat = sorted(l for ok, l in self._calls if ok)
        return {
            "calls": n,
            "error_rate": (
                round(sum(1 for ok, _ in self._calls if not ok) / n, 4) if n else 0.0
            ),
            "p95_ms": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None,
            "consecutive_errors": self.consecutive_errors,
        }

    def state(self) -> str:
        s = self.stats()
        if self.consecutive_errors >= self.down_consecutive:
            return DOWN  # серия подряд — сигнал даже при малом числе вызовов
        if s["calls"] < self.min_calls:
            return HEALTHY  # мало данных: не объявляем деградацию по шуму
        if s["error_rate"] >= self.down_error_rate:
            return DOWN
        if (
            s["error_rate"] >= self.degraded_error_rate
            or (s["p95_ms"] or 0) >= self.slow_p95_ms
        ):
            return DEGRADED
        return HEALTHY

    def policy(self) -> Dict[str, Any]:
        """Защитное действие; никогда не «переключиться на другую биржу»."""
        st = self.state()
        return {
            "state": st,
            "block_entries": st == DOWN,
            "widen_caution": st == DEGRADED,  # напр. не повышать размер, чаще сверять
            "auto_failover": False,
        }


@dataclass(frozen=True)
class HedgeLimits:
    max_capital_fraction: float = 0.20  # доля капитала второй площадки под маржу хеджа
    max_depth_share: float = 0.10  # доля видимой глубины, которую не превышаем
    max_basis_bps: float = 50.0  # выше — хедж не рассматриваем (базисный риск)
    min_notional: float = 10.0


def hedge_plan(
    primary_net_notional: float,
    hedge_ratio: float,
    venue_equity: float,
    venue_leverage: float,
    depth_notional: float,
    basis_bps_p95: float,
    fee_bps: float,
    limits: HedgeLimits = HedgeLimits(),
) -> Dict[str, Any]:
    """План хеджа: знак противоположен основной экспозиции; ограничения перечислены в `capped_by`."""
    if not 0 <= hedge_ratio <= 1:
        raise ValueError("hedge_ratio ∈ [0, 1]")
    if venue_equity < 0 or venue_leverage <= 0 or depth_notional < 0:
        raise ValueError("некорректные параметры площадки")
    want = abs(primary_net_notional) * hedge_ratio
    caps: List[Tuple[str, float]] = [
        ("capital", limits.max_capital_fraction * venue_equity * venue_leverage),
        ("depth", limits.max_depth_share * depth_notional),
    ]
    notional = want
    capped_by: List[str] = []
    for name, cap in caps:
        if cap < notional:
            notional, capped_by = cap, capped_by + [name]
    reasons: List[str] = []
    if basis_bps_p95 > limits.max_basis_bps:
        reasons.append("basis_risk_too_high")
    if notional < limits.min_notional:
        reasons.append("below_min_notional")
    viable = not reasons and want > 0
    sign = -1.0 if primary_net_notional > 0 else 1.0
    return {
        "viable": viable,
        "hedge_notional": round(sign * notional, 4) if viable else 0.0,
        "coverage": (
            round(notional / abs(primary_net_notional), 4)
            if primary_net_notional
            else 0.0
        ),
        "capped_by": capped_by,
        "reasons": reasons,
        "margin_required": round(notional / venue_leverage, 4),
        "expected_cost_bps": round(2 * fee_bps, 4),  # вход + выход хеджа
        "residual_basis_risk": round(
            notional * basis_bps_p95 / 1e4, 4
        ),  # денег при базисе p95
    }
