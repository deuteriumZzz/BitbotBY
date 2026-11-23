"""Отчёт по Decision Ledger: на чём стоят решения бота (Phase 3).

Запуск:  python -m src.research.signal_intelligence_report [путь_к_журналу]

Отвечает на вопросы, без ответов на которые менять confidence нельзя
(AUDIT_REPORT C-1/C-2): как меняется уверенность по стадиям, как часто один
фактор действует на решение несколькими каналами, сколько независимых групп
стоит за решением, как часто политики спорят, как работает цепочка AI-провайдеров.
Ничего не пишет и не меняет; читает журнал, как есть. Это описательная
статистика, а не доказательство наличия или отсутствия преимущества.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

_DEFAULT_PATH = "data/decision_ledger.jsonl"


def load_records(path: str) -> List[Dict[str, Any]]:
    """Читает журнал (сначала ротированный `.1`, затем текущий); битые строки пропускает."""
    records: List[Dict[str, Any]] = []
    for p in (path + ".1", path):
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except ValueError:
                    continue
    return records


def _stats(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    n = len(values)
    return {
        "n": n,
        "mean": round(sum(values) / n, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "share_up": round(sum(v > 1e-9 for v in values) / n, 3),
        "share_down": round(sum(v < -1e-9 for v in values) / n, 3),
    }


def _p95(values: List[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 1)


def _signals(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    decisions = [
        d
        for r in records
        if r.get("kind") == "signals"
        for d in r.get("decisions", [])
    ]
    n = len(decisions)
    out: Dict[str, Any] = {"n_decisions": n}
    if not n:
        return out
    deltas: Dict[str, List[float]] = {}
    redundant: Counter = Counter()
    conflicts: Counter = Counter()
    groups_count: List[int] = []
    for d in decisions:
        trail = d.get("conf_trail") or []
        for (s0, v0), (s1, v1) in zip(trail, trail[1:]):
            deltas.setdefault(f"{s0}→{s1}", []).append(v1 - v0)
        groups_count.append(len(d.get("independent_groups") or []))
        for g in d.get("redundant_groups") or []:
            redundant[g] += 1
        for c in d.get("conflicts") or []:
            conflicts[f"{c.get('a')} vs {c.get('b')}"] += 1
    out.update(
        {
            "by_status": dict(Counter(d.get("status") for d in decisions)),
            "by_mode": dict(Counter(d.get("mode") for d in decisions)),
            "by_source": dict(
                Counter(
                    "+".join(sorted({o.get("kind") for o in d.get("outputs", [])}))
                    for d in decisions
                )
            ),
            "by_confidence_kind": dict(
                Counter(d.get("confidence_kind") for d in decisions)
            ),
            "stage_deltas": {k: _stats(v) for k, v in sorted(deltas.items())},
            "mean_independent_groups": round(sum(groups_count) / n, 2),
            "single_group_share": round(sum(g <= 1 for g in groups_count) / n, 3),
            "with_redundant_channels": sum(
                1 for d in decisions if d.get("redundant_groups")
            ),
            "redundant_by_group": dict(redundant.most_common()),
            "with_context_boost": sum(
                1 for d in decisions if "context_boost" in (d.get("reason_codes") or [])
            ),
            "with_capped_penalty": sum(
                1
                for d in decisions
                if "ctx_penalty_capped" in (d.get("reason_codes") or [])
            ),
            "conflicts": dict(conflicts.most_common()),
        }
    )
    return out


def _ai(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    calls = [r for r in records if r.get("kind") == "ai_call"]
    ok = [r for r in calls if r.get("status") == "ok"]
    latencies = [float(r["latency_ms"]) for r in ok if r.get("latency_ms") is not None]
    statuses = Counter(a.get("status") for r in calls for a in (r.get("attempts") or []))
    return {
        "n_calls": len(calls),
        "ok": len(ok),
        "exhausted": sum(1 for r in calls if r.get("status") == "exhausted"),
        "answered_by": dict(Counter(r.get("provider") for r in ok).most_common()),
        "fallback_share": (
            round(sum(1 for r in ok if r.get("attempts")) / len(ok), 3) if ok else None
        ),
        "failed_attempts_by_status": dict(statuses.most_common()),
        "latency_ms_mean": (
            round(sum(latencies) / len(latencies), 1) if latencies else None
        ),
        "latency_ms_p95": _p95(latencies),
    }


def _data_quality(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    total: Counter = Counter()
    cycles = 0
    for r in records:
        if r.get("kind") == "data_quality":
            cycles += 1
            total.update(r.get("counts") or {})
    return {"cycles": cycles, "snapshots_by_status": dict(total)}


def _execution(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    ex = [r for r in records if r.get("kind") == "execution"]
    rejects = Counter(
        r.get("reject") for r in ex if not r.get("opened") and r.get("reject")
    )
    return {
        "n": len(ex),
        "opened": sum(1 for r in ex if r.get("opened")),
        "top_rejects": dict(rejects.most_common(5)),
    }


def build_report(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "records": len(records),
        "signals": _signals(records),
        "ai": _ai(records),
        "data_quality": _data_quality(records),
        "execution": _execution(records),
    }


def format_report(rep: Dict[str, Any]) -> str:
    s, ai, dq, ex = rep["signals"], rep["ai"], rep["data_quality"], rep["execution"]
    lines = [f"Журнал решений: {rep['records']} записей", ""]
    lines.append(f"РЕШЕНИЯ: {s.get('n_decisions', 0)}")
    if s.get("n_decisions"):
        lines += [
            f"  по статусу: {s['by_status']}   по режиму: {s['by_mode']}",
            f"  по источнику: {s['by_source']}",
            f"  природа confidence: {s['by_confidence_kind']}",
            f"  независимых групп на решение: {s['mean_independent_groups']} "
            f"(решений с ≤1 группой: {s['single_group_share']:.0%})",
            f"  решений с избыточными каналами факторов: "
            f"{s['with_redundant_channels']} {s['redundant_by_group']}",
            f"  с context_boost: {s['with_context_boost']}; "
            f"со срезанным потолком штрафом: {s['with_capped_penalty']}",
            f"  конфликты политик: {s['conflicts'] or 'нет'}",
            "  изменение confidence по стадиям (mean / доля вверх / доля вниз):",
        ]
        for stage, st in s["stage_deltas"].items():
            lines.append(
                f"    {stage}: n={st['n']} mean={st['mean']:+.3f} "
                f"↑{st['share_up']:.0%} ↓{st['share_down']:.0%}"
            )
    lines += [
        "",
        f"AI-ВЫЗОВЫ: {ai['n_calls']} (ответили: {ai['ok']}, "
        f"цепочка исчерпана: {ai['exhausted']})",
        f"  ответили: {ai['answered_by']}; доля вызовов с fallback: "
        f"{ai['fallback_share']}",
        f"  отказы по статусам: {ai['failed_attempts_by_status']}",
        f"  задержка мс: mean={ai['latency_ms_mean']} p95={ai['latency_ms_p95']}",
        "",
        f"КАЧЕСТВО ДАННЫХ: циклов {dq['cycles']}, "
        f"снэпшоты по статусам {dq['snapshots_by_status']}",
        f"ИСПОЛНЕНИЕ: попыток {ex['n']}, открыто {ex['opened']}, "
        f"частые отказы {ex['top_rejects']}",
        "",
        "Описательная статистика: из неё не следует ни наличие, ни отсутствие "
        "торгового преимущества.",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = args[0] if args else os.getenv("DECISION_LEDGER_PATH", _DEFAULT_PATH)
    records = load_records(path)
    if not records:
        print(f"Журнал пуст или не найден: {path}")
        return 1
    print(format_report(build_report(records)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
