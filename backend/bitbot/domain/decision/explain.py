"""Объяснение решения: почему бот решил именно так.

Берёт `TradingDecision` в виде словаря (так он лежит в Decision Ledger) и
составляет читаемый разбор: кто участвовал и чем является их confidence, как
менялась уверенность по стадиям, где политики спорят, на какие группы факторов
опирается решение и какие из них влияют несколькими каналами сразу. Ничего не
пересчитывает и не оценивает качество — только пересказывает записанное.

CLI:  python -m bitbot.domain.decision.explain <decision_id> [путь_к_журналу]
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

_STATUS_RU = {
    "LONG": "ЛОНГ",
    "SHORT": "ШОРТ",
    "NO_TRADE": "без сделки",
    "INSUFFICIENT_DATA": "недостаточно данных",
}
_KIND_RU = {
    "llm_self_report": "самооценка LLM, не вероятность",
    "sac_action_magnitude": "величина действия SAC, не вероятность",
    "heuristic_constant": "константа, заданная в коде",
    "strategy_score": "внутренний скор стратегии",
    "combined_heuristic": "линейная смесь выше перечисленных",
    "direction_only": "только направление",
}


def _fmt(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.2f}"


def explain_decision(d: Dict[str, Any]) -> str:
    """Читаемое объяснение одного решения (словарь из журнала или `to_dict()`)."""
    status = _STATUS_RU.get(d.get("status", ""), str(d.get("status")))
    lines: List[str] = [
        f"Решение {status} по {d.get('symbol')} "
        f"(id {d.get('decision_id')}, режим {d.get('mode')}, "
        f"режим рынка {d.get('regime') or '—'})",
        f"Итоговая уверенность {_fmt(d.get('confidence'))} — "
        f"{_KIND_RU.get(d.get('confidence_kind', ''), d.get('confidence_kind'))}.",
        f"Стратегия: {d.get('strategy')}; вход {d.get('entry')}, "
        f"SL {d.get('stop_loss')}, TP {d.get('take_profit')}.",
        "",
        "Участники:",
    ]
    for o in d.get("outputs") or []:
        who = o.get("model_id")
        provenance = "/".join(
            x for x in (o.get("provider"), o.get("model_version")) if x
        )
        if provenance:
            who = f"{who} ({provenance})"
        kind = _KIND_RU.get(o.get("confidence_kind", ""), o.get("confidence_kind"))
        lines.append(
            f"  • {who}: {o.get('direction')} {_fmt(o.get('confidence'))} [{kind}]"
        )
    trail = d.get("conf_trail") or []
    if trail:
        chain = " → ".join(f"{s} {v:.2f}" for s, v in trail)
        lines += ["", f"Как менялась уверенность: {chain}"]
    conflicts = d.get("conflicts") or []
    if conflicts:
        pairs = "; ".join(f"{c['a']} против {c['b']}" for c in conflicts)
        lines += ["", f"Конфликты политик (не усреднялись): {pairs}"]
    groups = d.get("independent_groups") or []
    if groups:
        lines += [
            "",
            f"Независимых групп за решением: {len(groups)} ({', '.join(groups)}).",
        ]
    redundant = d.get("redundant_groups") or []
    channels = d.get("channels") or {}
    if redundant:
        detail = "; ".join(f"{g}: {' + '.join(channels.get(g, []))}" for g in redundant)
        lines.append(
            f"Один фактор влияет несколькими каналами (риск повторного учёта): {detail}."
        )
    codes = d.get("reason_codes") or []
    if codes:
        lines += ["", "Коды причин: " + ", ".join(codes)]
    lines += [
        "",
        "Уверенность — внутренняя оценка бота, а не вероятность прибыли сделки.",
    ]
    return "\n".join(lines)


def find_decision(
    records: List[Dict[str, Any]], decision_id: str
) -> Optional[Dict[str, Any]]:
    """Решение и итог его исполнения из записей журнала (последнее совпадение)."""
    found: Optional[Dict[str, Any]] = None
    outcome: Optional[Dict[str, Any]] = None
    for rec in records:
        if rec.get("kind") == "signals":
            for d in rec.get("decisions", []):
                if d.get("decision_id") == decision_id:
                    found = d
        elif rec.get("kind") == "execution" and rec.get("decision_id") == decision_id:
            outcome = rec
    if found is None:
        return None
    return {"decision": found, "execution": outcome}


def _load(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in (path + ".1", path):
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print(
            "Использование: python -m bitbot.domain.decision.explain <decision_id> [журнал]"
        )
        return 2
    path = (
        args[1]
        if len(args) > 1
        else os.getenv("DECISION_LEDGER_PATH", "data/decision_ledger.jsonl")
    )
    hit = find_decision(_load(path), args[0])
    if hit is None:
        print(f"Решение {args[0]} не найдено в {path}")
        return 1
    print(explain_decision(hit["decision"]))
    ex = hit["execution"]
    print()
    if ex is None:
        print("Исполнение: записи нет (решение не дошло до исполнителя).")
    elif ex.get("opened"):
        print("Исполнение: позиция открыта.")
    else:
        print(f"Исполнение: не открыто — {ex.get('reject') or 'причина не записана'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
