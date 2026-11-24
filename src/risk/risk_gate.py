"""Единый Risk Gate (Phase 9, MASTER_SPEC §9.6): чистая функция, независимая от AI.

Принимает предложение входа и состояние портфеля, возвращает вердикт
allow/reduce/deny с кодами причин. Не знает ни о моделях, ни о режимах: любой
источник решения (AI, SAC, Hybrid, Telegram, trusted_trader) должен проходить
один и тот же шлюз. ОРДЕРА, УМЕНЬШАЮЩИЕ ПОЗИЦИЮ, ГЕЙТ НЕ БЛОКИРУЕТ — иначе
защита мешала бы выходу из риска.

Уровни circuit breaker вычисляются независимо (каждый — отдельный признак):
  1 запрет новых входов  · 2 отмена отложенных ордеров ·
  3 сокращение/закрытие по правилам · 4 полная остановка.
Пороги уровней 2–4 и лимиты `max_*` — ПРЕДЛОЖЕНИЯ владельцу, не выведены из данных.
Подключение к исполнителю не выполнено: это изменение торговой логики.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from src.risk.metrics import net_beta_exposure, stress_cvar


@dataclass(frozen=True)
class GateLimits:
    daily_loss_limit: float = 0.05  # доля капитала на начало дня
    max_drawdown: float = 0.20  # от пика
    max_consecutive_losses: int = 3
    max_correlation: float = 0.70
    max_symbol_exposure: float = 0.25  # доля капитала в одном символе
    max_gross_exposure: float = 1.00  # Σ|notional| / капитал
    max_net_beta_exposure: float = 0.75  # |Σ n·β| / капитал
    max_stress_cvar: float = 0.10  # Σ|n|·CVaR / капитал при корреляции → 1
    min_notional: float = 10.0  # меньше — не открываем (биржевые минимумы)
    # уровни по просадке: доли max_drawdown (предложение)
    cancel_orders_dd: float = 0.75
    reduce_dd: float = 1.00
    halt_dd: float = 1.25
    block_on_ood: bool = True

    @classmethod
    def from_config(cls, config: object) -> "GateLimits":
        """Берёт существующие пороги бота; новые остаются предложенными значениями."""
        return cls(
            daily_loss_limit=float(getattr(config, "DAILY_LOSS_LIMIT", 0.05)),
            max_drawdown=float(getattr(config, "MAX_DRAWDOWN_PERCENT", 0.20)),
            max_consecutive_losses=int(getattr(config, "CIRCUIT_BREAKER_LOSSES", 3)),
            max_correlation=float(getattr(config, "MAX_CORRELATION", 0.70)),
        )


@dataclass(frozen=True)
class Position:
    symbol: str
    notional: float  # знаковая: минус = шорт
    beta: float = 1.0  # бета к BTC
    cvar: float = 0.05  # CVaR актива, доля за горизонт удержания


@dataclass(frozen=True)
class Portfolio:
    equity: float
    peak_equity: float
    day_start_equity: float
    positions: Sequence[Position] = ()
    consecutive_losses: int = 0
    kill_switch: bool = False  # ручной/внешний аварийный стоп


@dataclass(frozen=True)
class Proposal:
    symbol: str
    notional: float  # знаковая
    beta: float = 1.0
    cvar: float = 0.05
    max_corr_with_open: float = 0.0
    reduce_only: bool = False
    ood: bool = False  # признак выхода за обучающее распределение (Phase 8)


@dataclass
class Verdict:
    action: str  # allow | reduce | deny
    approved_notional: float
    reasons: List[str] = field(default_factory=list)
    level: int = 0

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "approved_notional": self.approved_notional,
            "reasons": list(self.reasons),
            "level": self.level,
        }


def breaker_levels(pf: Portfolio, lim: GateLimits, ood: bool = False) -> dict:
    """Независимые признаки уровней 1–4 (могут быть истинны в любой комбинации)."""
    dd = 1 - pf.equity / pf.peak_equity if pf.peak_equity > 0 else 0.0
    day_loss = (
        (pf.day_start_equity - pf.equity) / pf.day_start_equity
        if pf.day_start_equity > 0
        else 0.0
    )
    return {
        1: bool(
            pf.kill_switch
            or day_loss >= lim.daily_loss_limit
            or pf.consecutive_losses >= lim.max_consecutive_losses
            or dd >= lim.max_drawdown * lim.cancel_orders_dd
            or (ood and lim.block_on_ood)
        ),
        2: bool(pf.kill_switch or dd >= lim.max_drawdown * lim.cancel_orders_dd),
        3: bool(pf.kill_switch or dd >= lim.max_drawdown * lim.reduce_dd),
        4: bool(pf.kill_switch or dd >= lim.max_drawdown * lim.halt_dd),
    }


def evaluate(p: Proposal, pf: Portfolio, lim: GateLimits) -> Verdict:
    if p.reduce_only:
        return Verdict("allow", p.notional, ["reduce_only"])
    if pf.equity <= 0:
        return Verdict("deny", 0.0, ["no_equity"], 4)

    lv = breaker_levels(pf, lim, p.ood)
    level = max((k for k, v in lv.items() if v), default=0)
    if level >= 1:
        breaker = ["circuit_breaker_l%d" % level]
        if p.ood and lim.block_on_ood:
            breaker.append("ood")
        return Verdict("deny", 0.0, breaker, level)

    if p.max_corr_with_open > lim.max_correlation:
        return Verdict("deny", 0.0, ["correlation"], 0)

    sign = 1.0 if p.notional >= 0 else -1.0
    want = abs(p.notional)
    cap = want
    reasons: List[str] = []

    def limit(room: float, code: str) -> None:
        nonlocal cap
        room = max(room, 0.0)
        if room < cap:
            cap = room
            reasons.append(code)

    cur = {x.symbol: x.notional for x in pf.positions}
    limit(
        lim.max_symbol_exposure * pf.equity - abs(cur.get(p.symbol, 0.0)),
        "symbol_exposure",
    )
    gross = sum(abs(x.notional) for x in pf.positions)
    limit(lim.max_gross_exposure * pf.equity - gross, "gross_exposure")

    # бета: нужный запас до лимита |Σ n·β| с учётом направления новой позиции
    net_beta = net_beta_exposure(
        [x.notional for x in pf.positions], [x.beta for x in pf.positions]
    )
    per_unit = sign * p.beta
    if per_unit != 0:
        limit_money = lim.max_net_beta_exposure * pf.equity
        room_money = limit_money - net_beta if per_unit > 0 else limit_money + net_beta
        limit(room_money / abs(per_unit), "net_beta")

    # стресс-CVaR: Σ|n|·cvar растёт линейно с размером новой позиции
    base = stress_cvar(
        [x.notional for x in pf.positions], [x.cvar for x in pf.positions]
    )
    if p.cvar > 0:
        limit((lim.max_stress_cvar * pf.equity - base) / p.cvar, "stress_cvar")

    if cap < lim.min_notional:
        return Verdict("deny", 0.0, reasons or ["below_min_notional"], 0)
    if cap < want:
        return Verdict("reduce", sign * cap, reasons, 0)
    return Verdict("allow", sign * want, [], 0)
