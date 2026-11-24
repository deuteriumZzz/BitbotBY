"""Стратегии объединения оценок AI и SAC (Phase 6, MASTER_SPEC §12, §6 «Hybrid»).

Чистые функции без побочных эффектов. Сейчас торгует только `current` — точная
копия правила из `SignalCombiner._hybrid` (тест сверяет их по сетке входов, чтобы
эталон не разъехался с боевым кодом). Остальные четыре — гипотезы из ТЗ
(Agreement, Weighted, Contextual, Risk-aware) с явными параметрами: они работают
**в тени**, их вердикты пишутся в журнал рядом с боевым решением, торговлю не
меняют. Ни одна не объявлена лучшей: сравнивать их можно только на исходах
сделок после издержек, а этих данных по Hybrid пока нет (SAC не обучена).

Вход каждой стратегии — независимые предложения политик (`Views`): AI и SAC
оценивают рынок отдельно, друг друга не видят.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

# Веса (sac, ai) по режиму рынка — копия `signal_combiner._REGIME_WEIGHTS`
# (тест проверяет совпадение).
REGIME_WEIGHTS: Dict[str, Tuple[float, float]] = {
    "trending_up": (0.5, 0.5),
    "trending_down": (0.3, 0.7),
    "ranging": (0.4, 0.6),
    "unknown": (0.4, 0.6),
}
DEFAULT_WEIGHTS = (0.4, 0.6)
SAC_SOLO_MIN = 0.80  # AI молчит — SAC входит один только при такой уверенности
RISK_AWARE_GAP_PENALTY = 0.5  # доля разрыва уверенностей, вычитаемая в risk_aware

BUY, SELL, HOLD = "buy", "sell", "hold"


@dataclass(frozen=True)
class Proposal:
    action: str  # buy | sell | hold
    confidence: float


@dataclass(frozen=True)
class Views:
    """Независимые предложения политик по одному символу."""

    ai: Optional[Proposal]  # None — AI промолчал (или ниже порога парсера)
    sac: Optional[Proposal]
    regime: str
    min_conf: float


@dataclass(frozen=True)
class Verdict:
    action: str  # buy | sell | hold
    confidence: float
    reasons: Tuple[str, ...] = ()


def _hold(*reasons: str) -> Verdict:
    return Verdict(HOLD, 0.0, tuple(reasons))


def _weights(regime: str) -> Tuple[float, float]:
    return REGIME_WEIGHTS.get(regime, DEFAULT_WEIGHTS)


def _sign(action: str) -> int:
    return {BUY: 1, SELL: -1}.get(action, 0)


def _agree(v: Views) -> bool:
    return (
        v.ai is not None
        and v.sac is not None
        and v.sac.action != HOLD
        and v.ai.action == v.sac.action
    )


# ── Эталон: боевое правило ────────────────────────────────────────────────────


def current(v: Views) -> Verdict:
    """Копия `_hybrid`: согласие AI и SAC, режимные веса, SAC solo при ≥ 0.80.

    SAC hold — вето; AI молчит — SAC входит один только при высокой уверенности;
    расхождение действий — отказ; смесь ниже порога — отказ. Chronos не входит.
    """
    if v.sac is None or v.sac.action == HOLD:
        return _hold("sac_hold")
    if v.ai is None:
        if v.sac.confidence >= SAC_SOLO_MIN:
            return Verdict(v.sac.action, v.sac.confidence, ("sac_solo",))
        return _hold("ai_silent_sac_below_solo")
    if v.ai.action != v.sac.action:
        return _hold("disagree")
    w_sac, w_ai = _weights(v.regime)
    combined = round(v.sac.confidence * w_sac + v.ai.confidence * w_ai, 3)
    if combined < v.min_conf:
        return _hold("below_min_conf")
    return Verdict(v.sac.action, combined, ("agree",))


# ── Гипотезы (только в тени) ──────────────────────────────────────────────────


def agreement(v: Views) -> Verdict:
    """Agreement: оба обязаны согласиться; уверенность — консервативная (минимум)."""
    if not _agree(v):
        return _hold("no_agreement")
    conf = round(min(v.ai.confidence, v.sac.confidence), 3)  # type: ignore[union-attr]
    if conf < v.min_conf:
        return _hold("below_min_conf")
    return Verdict(v.sac.action, conf, ("agree_min",))  # type: ignore[union-attr]


def weighted(v: Views) -> Verdict:
    """Weighted: знаковая смесь с режимными весами; несогласие гасит друг друга.

    Молчащая политика не участвует (веса нормируются на доступные); SAC hold
    участвует как нулевой голос и разбавляет оценку.
    """
    w_sac, w_ai = _weights(v.regime)
    items = []
    if v.sac is not None:
        items.append((w_sac, _sign(v.sac.action) * v.sac.confidence))
    if v.ai is not None:
        items.append((w_ai, _sign(v.ai.action) * v.ai.confidence))
    if not items:
        return _hold("no_inputs")
    total = sum(w for w, _ in items)
    score = sum(w * s for w, s in items) / total
    if abs(score) < v.min_conf:
        return _hold("score_below_min")
    return Verdict(BUY if score > 0 else SELL, round(abs(score), 3), ("weighted_vote",))


def contextual(v: Views) -> Verdict:
    """Contextual: правило зависит от режима.

    trending_up — SAC ведёт: входит и один при ≥ 0.80, при согласии — режимные
    веса. Остальные режимы — только согласие, уверенность — минимум.
    """
    if v.regime == "trending_up":
        if _agree(v):
            return current(v)
        if (
            v.ai is None
            and v.sac is not None
            and v.sac.action != HOLD
            and v.sac.confidence >= SAC_SOLO_MIN
        ):
            return Verdict(v.sac.action, v.sac.confidence, ("trend_sac_lead",))
        return _hold("no_agreement")
    return agreement(v)


def risk_aware(v: Views) -> Verdict:
    """Risk-aware: решение `current`, но с вычетом за разброс оценок политик.

    Чем сильнее расходятся уверенности AI и SAC, тем ниже итог. Экспозиция
    портфеля здесь недоступна (решается позже, в Risk Gate).
    """
    base = current(v)
    if base.action == HOLD:
        return base
    if v.ai is not None and v.sac is not None:
        gap = abs(v.ai.confidence - v.sac.confidence)
        conf = round(base.confidence - RISK_AWARE_GAP_PENALTY * gap, 3)
    else:
        conf = base.confidence
    if conf < v.min_conf:
        return _hold("uncertainty_penalty")
    return Verdict(base.action, conf, base.reasons + ("gap_penalty",))


STRATEGIES: Dict[str, Callable[[Views], Verdict]] = {
    "current": current,
    "agreement": agreement,
    "weighted": weighted,
    "contextual": contextual,
    "risk_aware": risk_aware,
}


def evaluate_all(v: Views) -> Dict[str, Verdict]:
    """Вердикты всех стратегий по одному набору предложений."""
    return {name: fn(v) for name, fn in STRATEGIES.items()}
