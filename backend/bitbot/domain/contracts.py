"""Единые контракты решений.

Политики (AI, SAC, Chronos, контекст, локальная стратегия) только **предлагают**;
здесь описано, как выглядит предложение (`ModelOutput`) и итоговое решение по
символу (`TradingDecision`). Контракты строятся адаптером из существующих
словарей рекомендаций (`decision_from_rec`) — поток данных и значения
confidence не меняются, контракты нужны журналу решений.

Главное: у каждого confidence явно указано, **чем оно является**
(`confidence_kind`). Самооценка LLM, величина действия SAC и вручную заданная
константа контекстного сигнала — не вероятность прибыли и не сравнимы между
собой.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from bitbot.domain.signal_intelligence import analyze_signal

# Сигнал сгенерирован локальной стратегией, без AI. Старые записи в истории
# опыта помечены английским «Local analysis» — их тоже считаем локальными.
LOCAL_REASONING_PREFIX = "Локальная стратегия"


def is_local_reasoning(reasoning: str) -> bool:
    return LOCAL_REASONING_PREFIX in reasoning or "Local analysis" in reasoning


# Статусы решения.
LONG = "LONG"
SHORT = "SHORT"
NO_TRADE = "NO_TRADE"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

# Природа числа confidence.
CK_LLM = "llm_self_report"  # самооценка LLM, не откалибрована
CK_SAC = "sac_action_magnitude"  # |a| действия SAC, не вероятность
CK_HEURISTIC = "heuristic_constant"  # задано вручную в коде (контекстные сигналы)
CK_STRATEGY = "strategy_score"  # внутренний скор локальной стратегии
CK_COMBINED = "combined_heuristic"  # линейная смесь выше перечисленных
CK_DIRECTION = "direction_only"  # прогноз без числа уверенности (Chronos)

_DIRECTION = {"buy": "buy", "sell": "sell", "up": "buy", "down": "sell"}
_STATUS = {"buy": LONG, "sell": SHORT}


@dataclass(frozen=True)
class ModelOutput:
    """Предложение одной политики."""

    model_id: str  # "ai", "sac", "forecast:chronos", "context", "local:<strategy>"
    kind: str  # ai | sac | forecast | context | local
    direction: str  # buy | sell | hold
    confidence: Optional[float]
    confidence_kind: str
    provider: str = ""
    model_version: str = ""
    latency_ms: Optional[float] = None
    error_status: str = ""
    rationale: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TradingDecision:
    """Итоговое решение по символу и всё, из чего оно получено."""

    decision_id: str
    trace_id: str
    mode: str
    symbol: str
    status: str
    confidence: float
    confidence_kind: str
    strategy: str
    entry: float
    stop_loss: float
    take_profit: float
    regime: str
    outputs: Tuple[ModelOutput, ...]
    conf_trail: Tuple[Tuple[str, float], ...]
    reason_codes: Tuple[str, ...]
    # Signal Intelligence: описание опоры решения, confidence не трогает.
    independent_groups: Tuple[str, ...] = ()
    redundant_groups: Tuple[str, ...] = ()
    channels: Optional[Dict[str, Tuple[str, ...]]] = None
    conflicts: Tuple[Dict[str, Any], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_decision_id(trace_id: str, symbol: str, action: str) -> str:
    """Детерминированный id: тот же цикл/символ/направление → тот же id."""
    raw = f"{trace_id}|{symbol}|{action}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _trail_conf(trail: List[Any], stage: str) -> Optional[float]:
    for name, value in trail:
        if name == stage:
            return float(value)
    return None


def _reason_codes(rec: Dict[str, Any]) -> Tuple[str, ...]:
    codes: List[str] = []
    if rec.get("context_boost"):
        codes.append("context_boost")
    penalty, raw = rec.get("ctx_penalty"), rec.get("ctx_penalty_raw")
    if penalty is not None and raw is not None and raw > penalty:
        codes.append("ctx_penalty_capped")
    flags = rec.get("context_flags")
    if flags:
        try:
            for name, value in json.loads(flags).items():
                codes.append(f"ctx:{name}={value}")
        except (ValueError, AttributeError):
            codes.append("ctx:unparsed")
    return tuple(codes)


def _outputs(rec: Dict[str, Any], trail: List[Any]) -> List[ModelOutput]:
    strategy = str(rec.get("strategy", ""))
    action = str(rec.get("action", "hold"))
    conf = float(rec.get("confidence", 0.0) or 0.0)
    provider = str(rec.get("ai_provider", "") or "")
    model = str(rec.get("ai_model", "") or "")

    def ai_out(c: Optional[float]) -> ModelOutput:
        return ModelOutput(
            "ai", "ai", action, c, CK_LLM, provider=provider, model_version=model
        )

    def sac_out(c: Optional[float]) -> ModelOutput:
        return ModelOutput("sac", "sac", action, c, CK_SAC)

    if strategy.startswith("hybrid("):
        outs = [
            ai_out(_trail_conf(trail, "ai")),
            sac_out(_trail_conf(trail, "sac")),
        ]
        chronos = rec.get("chronos_signal")
        if chronos not in (None, "off"):
            outs.append(
                ModelOutput(
                    "forecast:chronos",
                    "forecast",
                    _DIRECTION.get(str(chronos), "hold"),
                    None,
                    CK_DIRECTION,
                    rationale={"signal": chronos},
                )
            )
        return outs
    if strategy == "sac":
        return [sac_out(_trail_conf(trail, "sac") or conf)]
    if strategy == "context_signal":
        return [
            ModelOutput(
                "context",
                "context",
                action,
                conf,
                CK_HEURISTIC,
                rationale={"reasoning": str(rec.get("reasoning", ""))[:200]},
            )
        ]
    if is_local_reasoning(str(rec.get("reasoning", ""))):
        return [ModelOutput(f"local:{strategy}", "local", action, conf, CK_STRATEGY)]
    source = trail[0][1] if trail else conf
    return [ai_out(float(source))]


def decision_from_rec(
    rec: Dict[str, Any], mode: str, trace_id: str = ""
) -> TradingDecision:
    """Строит TradingDecision из рекомендации combiner'а. Ничего не меняет в rec."""
    action = str(rec.get("action", "hold"))
    trail = [list(x) for x in (rec.get("conf_trail") or [])]
    outputs = tuple(_outputs(rec, trail))
    combined = len(outputs) > 1
    kind = CK_COMBINED if combined else outputs[0].confidence_kind
    symbol = str(rec.get("symbol", ""))
    analysis = analyze_signal(
        rec,
        [
            (o.kind, o.direction, o.model_id, str(o.rationale.get("reasoning", "")))
            for o in outputs
        ],
    )
    return TradingDecision(
        decision_id=make_decision_id(trace_id, symbol, action),
        trace_id=trace_id,
        mode=mode,
        symbol=symbol,
        status=_STATUS.get(action, NO_TRADE),
        confidence=float(rec.get("confidence", 0.0) or 0.0),
        confidence_kind=kind,
        strategy=str(rec.get("strategy", "")),
        entry=float(rec.get("entry", 0.0) or 0.0),
        stop_loss=float(rec.get("stop_loss", 0.0) or 0.0),
        take_profit=float(rec.get("take_profit", 0.0) or 0.0),
        regime=str(rec.get("_regime", "") or ""),
        outputs=outputs,
        conf_trail=tuple((str(s), float(v)) for s, v in trail),
        reason_codes=_reason_codes(rec),
        independent_groups=analysis.independent_groups,
        redundant_groups=analysis.redundant_groups,
        channels=analysis.channels,
        conflicts=analysis.conflicts,
    )
