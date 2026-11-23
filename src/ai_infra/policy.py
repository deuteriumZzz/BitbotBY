"""Политика отказов по компонентам (Phase 10, MASTER_SPEC §13.4) в виде данных.

Одна таблица вместо разбросанных `try/except`: что делает система при сбое
каждого компонента. Принцип: отсутствие ответа ≠ подтверждение сделки —
деградация никогда не открывает вход и не повышает уверенность.
Таблица описывает целевое поведение; соответствие кода таблице проверяется
тестами по мере подключения (сейчас сверен только контракт самой таблицы).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

# Действия при сбое
NEUTRAL = "neutral"  # вклад компонента = нейтральный/пропущен, цикл идёт дальше
NO_VOTE = "no_vote"  # компонент не голосует
BLOCK_ENTRIES = "block_entries"  # новых входов нет
DEGRADE = "degrade"  # переход на упрощённый режим (внутри компонента)
REJECT = "reject"  # результат отброшен


@dataclass(frozen=True)
class FailurePolicy:
    action: str
    flag_in_ledger: bool
    note: str


POLICIES: Dict[str, FailurePolicy] = {
    "llm_unavailable": FailurePolicy(NEUTRAL, True, "поправка AI пропускается; цикл не блокируется"),
    "llm_invalid_output": FailurePolicy(REJECT, True, "ответ отбрасывается, повтор ≤N, затем нет голоса"),
    "forecaster_unavailable": FailurePolicy(NEUTRAL, True, "прогнозная поправка нейтральна; Hybrid+Forecasting → Hybrid"),
    "sac_failure": FailurePolicy(NO_VOTE, True, "SAC-вклад отсутствует"),
    "sac_failure_standalone": FailurePolicy(BLOCK_ENTRIES, True, "режим SAC Standalone без SAC: новых входов нет"),
    "rag_failure": FailurePolicy(DEGRADE, True, "решение без RAG, контекст пуст, пометка"),
    "stale_or_invalid_data": FailurePolicy(BLOCK_ENTRIES, True, "INSUFFICIENT_DATA, новых входов нет"),
    "database_failure": FailurePolicy(BLOCK_ENTRIES, True, "позиции ведёт Risk Engine по последнему состоянию + reconcile"),
    "redis_failure": FailurePolicy(DEGRADE, True, "in-memory/БД, критичное состояние не в Redis"),
    "risk_validation_failure": FailurePolicy(REJECT, True, "отказ, причина в Ledger"),
}

# Ни одно из этих действий не подтверждает сделку.
_NEVER_OPENS = {NEUTRAL, NO_VOTE, BLOCK_ENTRIES, DEGRADE, REJECT}


def policy_for(failure: str) -> FailurePolicy:
    """Политика сбоя; неизвестный сбой — самый безопасный вариант (запрет входов)."""
    return POLICIES.get(failure, FailurePolicy(BLOCK_ENTRIES, True, "неизвестный сбой: безопасный отказ"))


def never_confirms_trade(failure: str) -> bool:
    return policy_for(failure).action in _NEVER_OPENS
