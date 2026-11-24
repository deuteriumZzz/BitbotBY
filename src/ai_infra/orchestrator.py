"""AI Orchestrator (MASTER_SPEC §13.1): какие компоненты запускать, в каком порядке и когда хватит.

Уровни:
  L1  Signal Intelligence + Meta-Model        — всегда (дёшево, быстро)
  L2  + SAC и прогноз (Chronos/TimesFM)        — если L1 не дал уверенного ответа
  L3  + LLM и RAG                              — если остаётся неопределённость
  L4  разрешение конфликта (перепроверка)      — если конфликт политик не снят
Нет пригодных данных → безопасный отказ (L0): ни одного шага, новых входов нет.

`plan` — чистая функция (решает по контексту и бюджету времени), `run_plan` — выполнение:
параллельно внутри уровня, таймаут на компонент, отмена, дубли запросов не дублируются,
частичный сбой не роняет остальных. Сбой компонента → политика из `policy.py`
(нейтрально / без голоса), «нет ответа» никогда не подтверждает сделку.
Пороги и оценки стоимости — ПРЕДЛОЖЕНИЯ. В цикл бота не подключён: сегодня роль
маршрутизатора выполняет `SignalCombiner.combine` по режиму.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from src.ai_infra.policy import never_confirms_trade, policy_for

# Оценка стоимости компонента, мс (предложение; уточняется замерами src/ops).
COST_MS = {
    "signal_intel": 5, "meta_model": 10, "sac": 20, "forecast": 1500,
    "rag": 50, "llm": 8000, "recheck": 3000,
}
FAILURE_KEY = {
    "llm": "llm_unavailable", "forecast": "forecaster_unavailable", "sac": "sac_failure",
    "rag": "rag_failure", "meta_model": "database_failure", "signal_intel": "stale_or_invalid_data",
    "recheck": "llm_unavailable",
}
DECISIVE_HIGH, DECISIVE_LOW = 0.70, 0.30  # вероятность Meta-Model: «уверенно за / против»
MIN_GROUPS = 2  # независимых групп факторов, ниже которых L1 недостаточно


@dataclass(frozen=True)
class OrchContext:
    data_ok: bool = True
    n_independent_groups: int = 0
    has_conflict: bool = False
    meta_prob: Optional[float] = None  # калиброванная вероятность, только если Meta-Model usable
    ai_available: bool = True
    sac_ready: bool = False
    forecaster_ready: bool = False
    budget_ms: float = 15_000.0


@dataclass
class Plan:
    level: str  # L0..L4
    stages: List[List[str]] = field(default_factory=list)  # внутри стадии — параллельно
    skipped: Dict[str, str] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    est_ms: float = 0.0


def _l1_sufficient(c: OrchContext) -> bool:
    decisive = c.meta_prob is not None and (c.meta_prob >= DECISIVE_HIGH or c.meta_prob <= DECISIVE_LOW)
    return decisive and c.n_independent_groups >= MIN_GROUPS and not c.has_conflict


def plan(c: OrchContext) -> Plan:
    if not c.data_ok:
        return Plan("L0", reasons=["insufficient_data"])  # безопасный отказ
    p = Plan("L1", stages=[["signal_intel"] + (["meta_model"] if c.meta_prob is not None else [])])
    spent = max(COST_MS[x] for x in p.stages[0])
    if _l1_sufficient(c):
        p.reasons.append("l1_sufficient")
        p.est_ms = spent
        return p

    def add_stage(level: str, comps: List[str]) -> bool:
        nonlocal spent
        cost = max(COST_MS[x] for x in comps)
        if spent + cost > c.budget_ms:
            for x in comps:
                p.skipped[x] = "budget"
            return False
        p.stages.append(comps)
        p.level, spent = level, spent + cost
        return True

    l2 = [x for x, ok in (("sac", c.sac_ready), ("forecast", c.forecaster_ready)) if ok]
    for x, ok in (("sac", c.sac_ready), ("forecast", c.forecaster_ready)):
        if not ok:
            p.skipped[x] = "not_ready"
    if l2:
        add_stage("L2", l2)
    # L1 не дал уверенного ответа (иначе вернулись бы выше) → лестница идёт до LLM; ранняя остановка
    # между стадиях — через `stop_after` в `run_plan`, если L2 уже решил вопрос.
    if not c.ai_available:
        p.skipped["llm"] = "ai_unavailable"
        p.reasons.append("llm_skipped_unavailable")
    else:
        add_stage("L3", ["llm", "rag"])
    if c.has_conflict and p.level == "L3":
        add_stage("L4", ["recheck"])
    p.est_ms = spent
    return p


@dataclass
class Result:
    ok: bool
    value: Any = None
    error: Optional[str] = None
    policy: Optional[str] = None  # действие политики при сбое


Runner = Callable[[], Awaitable[Any]]


async def run_plan(
    pl: Plan,
    runners: Dict[str, Runner],
    *,
    timeouts: Optional[Dict[str, float]] = None,
    default_timeout: float = 30.0,
    max_concurrency: int = 4,
    stop_after: Optional[Callable[[Dict[str, Result]], bool]] = None,
) -> Dict[str, Result]:
    """Выполняет план по стадиям; сбой компонента — `Result(ok=False)` с политикой, а не исключение.

    `stop_after(results)` вызывается после каждой стадии: True — решение уже достаточно уверенное,
    дорогие следующие стадии (LLM, перепроверка) не запускаются.
    """
    timeouts = timeouts or {}
    sem = asyncio.Semaphore(max_concurrency)
    inflight: Dict[str, "asyncio.Future[Result]"] = {}
    out: Dict[str, Result] = {}

    async def one(name: str) -> Result:
        runner = runners.get(name)
        key = FAILURE_KEY.get(name, "stale_or_invalid_data")
        if runner is None:
            return Result(False, error="no_runner", policy=policy_for(key).action)
        try:
            async with sem:
                value = await asyncio.wait_for(runner(), timeouts.get(name, default_timeout))
            return Result(True, value=value)
        except asyncio.TimeoutError:
            return Result(False, error="timeout", policy=policy_for(key).action)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — частичный сбой не роняет остальных
            return Result(False, error=f"{type(exc).__name__}: {exc}", policy=policy_for(key).action)

    def start(name: str) -> "asyncio.Future[Result]":
        if name not in inflight:  # дубль запроса: тот же компонент не запускаем дважды
            inflight[name] = asyncio.ensure_future(one(name))
        return inflight[name]

    try:
        for stage in pl.stages:
            results = await asyncio.gather(*(start(n) for n in stage))
            out.update(dict(zip(stage, results)))
            if stop_after is not None and stop_after(out):
                break
    except asyncio.CancelledError:
        for f in inflight.values():
            f.cancel()
        raise
    return out


def failures_never_confirm(results: Dict[str, Result]) -> bool:
    """Инвариант: ни один сбой не трактуется как подтверждение сделки."""
    return all(
        never_confirms_trade(FAILURE_KEY.get(n, "stale_or_invalid_data"))
        for n, r in results.items()
        if not r.ok
    )
