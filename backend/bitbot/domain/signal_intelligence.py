"""Signal Intelligence: понять, на чём стоит решение.

Только описание и подсчёт, **confidence не меняется**: менять уверенность можно лишь
после измерения по Decision Ledger и A/B на истории. Здесь:

- группы независимости факторов и семейства стратегий — конфигурируемые таблицы;
  принадлежность к разным группам НЕ доказывает независимость,
  это рабочая гипотеза, которую проверяет отчёт по журналу;
- число независимых групп за решением;
- «избыточные каналы»: группа факторов, влияющая на одно решение двумя и более
  способами (вход SAC + фильтр + прямой сигнал + merge). Это риск повторного
  учёта; направление влияния по каналам журнал пока не хранит;
- конфликты политик (LONG против SHORT) как отдельные объекты, без усреднения.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# Имена факторов — как в context_flags / fired в SignalCombiner.
FACTOR_GROUP: Dict[str, str] = {
    "funding": "funding",
    "basis": "funding",  # премия фьючерса — тот же перекос плеча, что и funding
    "oi": "open_interest",
    "liquidation": "order_flow",
    "ob": "order_flow",
    "pcr": "options",
    "iv": "options",
    "fng": "sentiment",
    "trends": "sentiment",
    "reddit": "sentiment",
    "macro": "flows",
    "etf": "flows",
    "stablecoin": "flows",
    "chronos": "forecasting",
}
META_FLAGS = frozenset({"profile", "market_type"})  # теги, не факторы

# Факторы, которые SAC получает во входах (rl_env/dqn_signal obs[14..20]).
SAC_OBS_FACTORS = frozenset({"funding", "ob", "pcr", "fng", "iv", "basis", "trends"})

# Группы политик-источников.
KIND_GROUP = {"ai": "llm", "sac": "rl", "forecast": "forecasting"}

# Семейство стратегии по ключевым словам; первое совпадение выигрывает.
_FAMILY_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "mean_reversion",
        ("reversion", "bollinger", "keltner", "spike_reversal", "poc", "bounce"),
    ),
    (
        "volume",
        ("volume", "obv", "klinger", "elder_force", "ease_of", "mfi", "cvd", "chaikin"),
    ),
    (
        "structure",
        ("support", "fibonacci", "liquidity", "fair_value", "amd", "breakout"),
    ),
    ("volatility", ("burst", "squeeze", "waddah", "dual_thrust", "scalping", "pump")),
    ("sentiment", ("sentiment",)),
    ("pattern", ("engulfing", "shooting_star", "heikin")),
    (
        "momentum",
        (
            "rsi",
            "stoch",
            "kdj",
            "williams",
            "fisher",
            "kst",
            "rvi",
            "ifr2",
            "momentum",
            "cci",
            "awesome",
        ),
    ),
    (
        "trend",
        (
            "ema",
            "macd",
            "trend",
            "sar",
            "ichimoku",
            "halftrend",
            "zlsma",
            "gtaa",
            "ssl",
            "vortex",
            "swing",
            "adx",
            "chandelier",
            "hedge",
        ),
    ),
)

# Прямой контекстный сигнал → группа по тексту причины (см. _generate_context_signals).
_CONTEXT_REASON_GROUP: Tuple[Tuple[str, str], ...] = (
    ("funding", "funding"),
    ("basis", "funding"),
    ("liquidation", "order_flow"),
    ("squeeze", "order_flow"),
    ("orderbook", "order_flow"),
    ("pcr", "options"),
    ("fear", "sentiment"),
    ("etf", "flows"),
)


def strategy_family(name: str) -> str:
    """Семейство стратегии по имени; неизвестные → 'other'."""
    low = str(name).lower()
    for family, keywords in _FAMILY_RULES:
        if any(k in low for k in keywords):
            return family
    return "other"


def context_signal_group(reasoning: str) -> str:
    """Группа прямого контекстного сигнала по тексту причины."""
    low = str(reasoning).lower()
    for needle, group in _CONTEXT_REASON_GROUP:
        if needle in low:
            return group
    return "context"


@dataclass(frozen=True)
class SignalAnalysis:
    independent_groups: Tuple[str, ...] = ()
    redundant_groups: Tuple[str, ...] = ()
    channels: Optional[Dict[str, Tuple[str, ...]]] = None
    conflicts: Tuple[Dict[str, Any], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _fired_flags(rec: Dict[str, Any]) -> List[str]:
    raw = rec.get("context_flags")
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return []
    return [k for k in data if k not in META_FLAGS] if isinstance(data, dict) else []


def analyze_signal(
    rec: Dict[str, Any], outputs: Iterable[Tuple[str, str, str, str]]
) -> SignalAnalysis:
    """Анализ одного решения.

    :param rec: рекомендация combiner'а (читается `strategy`, `context_flags`,
        `context_boost`, `reasoning`).
    :param outputs: `(kind, direction, model_id, reasoning)` каждой политики.
    """
    outs = list(outputs)
    groups: Set[str] = set()
    channels: Dict[str, List[str]] = {}

    def add(group: str, channel: str) -> None:
        channels.setdefault(group, [])
        if channel not in channels[group]:
            channels[group].append(channel)

    has_sac = False
    for kind, _direction, model_id, reasoning in outs:
        if kind in KIND_GROUP:
            groups.add(KIND_GROUP[kind])
        elif kind == "local":
            groups.add(strategy_family(model_id.split(":", 1)[-1]))
        elif kind == "context":
            group = context_signal_group(reasoning)
            groups.add(group)
            add(group, "generated")
        has_sac = has_sac or kind == "sac"

    if has_sac:  # SAC видит эти факторы во входах независимо от фильтров
        for factor in SAC_OBS_FACTORS:
            add(FACTOR_GROUP[factor], "sac_obs")
    for factor in _fired_flags(rec):
        fired = FACTOR_GROUP.get(factor)
        if fired:
            add(fired, "filter")
            groups.add(fired)
    if rec.get("context_boost"):
        add("context", "merge")

    # избыточность считаем только по реальным факторам (без merge-метки)
    redundant = sorted(
        g for g, ch in channels.items() if g != "context" and len(ch) >= 2
    )
    directional = [(m, d) for _k, d, m, _r in outs if d in ("buy", "sell")]
    conflicts: List[Dict[str, Any]] = []
    for i, (m_a, d_a) in enumerate(directional):
        for m_b, d_b in directional[i + 1 :]:
            if d_a != d_b:
                conflicts.append({"a": m_a, "b": m_b, "kind": "direction"})
    return SignalAnalysis(
        independent_groups=tuple(sorted(groups)),
        redundant_groups=tuple(redundant),
        channels={g: tuple(c) for g, c in sorted(channels.items())},
        conflicts=tuple(conflicts),
    )
