"""Trading Mode Manager (Phase 1): таблица режимов и их готовность.

Один источник правды: какие политики активны в режиме и что ему нужно для
работы. Использует `SignalCombiner` (не торговать SAC-режимами без модели) и
состояние бота для приложения (почему режим недоступен). Сами режимы по-прежнему
выбираются через RuntimeConfig; переключение и политика перехода при открытых
позициях — отдельная задача (TARGET_ARCHITECTURE §3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

REASON_SAC_MISSING = "sac_model_missing"
REASON_AI_UNAVAILABLE = "ai_unavailable"
REASON_UNKNOWN_MODE = "unknown_mode"


@dataclass(frozen=True)
class ModeSpec:
    name: str
    label: str
    policies: Tuple[str, ...]
    needs_sac: bool
    needs_ai: bool


# «context» — прямые сигналы из рыночного контекста: combiner генерирует их
# в любом режиме, кроме local.
MODE_SPECS: Dict[str, ModeSpec] = {
    "local": ModeSpec("local", "Локальные стратегии", ("local",), False, False),
    "ai": ModeSpec("ai", "AI", ("ai", "context"), False, True),
    "dqn": ModeSpec("dqn", "SAC", ("sac", "context"), True, False),
    "hybrid": ModeSpec("hybrid", "Hybrid AI+SAC", ("ai", "sac", "context"), True, True),
}


def requires_sac(mode: str) -> bool:
    spec = MODE_SPECS.get(mode)
    return bool(spec and spec.needs_sac)


def readiness(
    mode: str, *, sac_loaded: bool, ai_available: bool = True
) -> Tuple[bool, str]:
    """(готов, причина). Причина пустая, если режим может принимать решения."""
    spec = MODE_SPECS.get(mode)
    if spec is None:
        return False, REASON_UNKNOWN_MODE
    if spec.needs_sac and not sac_loaded:
        return False, REASON_SAC_MISSING
    if spec.needs_ai and not ai_available:
        return False, REASON_AI_UNAVAILABLE
    return True, ""


def all_readiness(
    *, sac_loaded: bool, ai_available: bool = True
) -> Dict[str, Dict[str, object]]:
    """Готовность всех режимов — для состояния бота и приложения."""
    out: Dict[str, Dict[str, object]] = {}
    for name in MODE_SPECS:
        ready, reason = readiness(name, sac_loaded=sac_loaded, ai_available=ai_available)
        out[name] = {"ready": ready, "reason": reason}
    return out
