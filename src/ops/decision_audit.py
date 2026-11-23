"""Decision Audit (Phase 11): сквозная цепочка Data → … → Outcome по одному решению.

Склеивает записи Decision Ledger одного цикла (`trace_id`) вокруг решения
(`decision_id`) и показывает, какие стадии есть, а каких нет. Пробел — это
находка, а не ошибка отчёта: он говорит, чего журнал не сохранил (например,
исходы сделок в ledger не пишутся — их передают из БД сделок). Ничего не
пересчитывает.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

STAGES = (
    "data_quality",
    "replay_input",
    "ai_call",
    "signals",
    "rejections",
    "shadow_decisions",
    "execution",
    "outcome",
)
# Стадии, без которых цепочку нельзя считать воспроизводимой.
REQUIRED = ("data_quality", "signals", "execution")


def audit_decision(
    records: Iterable[Dict[str, Any]],
    decision_id: str,
    outcome: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Цепочка по `decision_id`; None, если решения нет в журнале.

    `outcome` — итог сделки из БД (PnL, причина выхода): в ledger его нет.
    """
    recs = list(records)
    decision: Optional[Dict[str, Any]] = None
    trace_id = None
    for r in recs:
        if r.get("kind") == "signals":
            for d in r.get("decisions", []):
                if d.get("decision_id") == decision_id:
                    decision, trace_id = d, r.get("trace_id")
    if decision is None:
        return None

    chain: Dict[str, Any] = {"signals": decision}
    for r in recs:
        kind = r.get("kind")
        if kind == "execution" and r.get("decision_id") == decision_id:
            chain["execution"] = r
        elif r.get("trace_id") == trace_id and kind in STAGES and kind not in ("signals", "execution"):
            chain.setdefault(kind, r)  # первая запись цикла этого типа
    if outcome is not None:
        chain["outcome"] = outcome

    gaps: List[str] = [s for s in REQUIRED if s not in chain]
    if "execution" in chain and chain["execution"].get("opened") and "outcome" not in chain:
        gaps.append("outcome")  # позиция открыта, исхода нет
    return {
        "decision_id": decision_id,
        "trace_id": trace_id,
        "symbol": decision.get("symbol"),
        "stages": [s for s in STAGES if s in chain],
        "gaps": gaps,
        "complete": not gaps,
        "chain": chain,
    }


def format_audit(a: Dict[str, Any]) -> str:
    lines = [
        f"Аудит решения {a['decision_id']} ({a['symbol']}), цикл {a['trace_id']}",
        "Стадии: " + " → ".join(a["stages"]),
    ]
    ex = a["chain"].get("execution")
    if ex:
        lines.append(
            "Исполнение: открыто" if ex.get("opened") else f"Исполнение: нет — {ex.get('reject') or '?'}"
        )
    out = a["chain"].get("outcome")
    if out:
        lines.append(f"Исход: {out}")
    lines.append("Пробелы: " + (", ".join(a["gaps"]) if a["gaps"] else "нет"))
    return "\n".join(lines)
