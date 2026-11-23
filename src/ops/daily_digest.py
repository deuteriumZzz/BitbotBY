"""Daily Health Digest (Phase 11): компактная сводка суток по Decision Ledger.

Строится из записей журнала за окно (по умолчанию 24 ч) и использует уже
существующие расчёты (`signal_intelligence_report`, `ai_infra.health`), не
дублируя их. Пороги тревог — предложения. Это описательная сводка: она не
оценивает прибыльность.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

from src.ai_infra.health import provider_health
from src.research.signal_intelligence_report import build_report

AI_EXHAUSTED_ALERT = 0.5  # доля вызовов с исчерпанной цепочкой
BAD_DATA_ALERT = 0.2  # доля снимков не в статусе VALID
RECONCILE_ALERT = 1  # любое расхождение с биржей заслуживает внимания


def _parse(ts: Any) -> Optional[datetime]:
    try:
        t = datetime.fromisoformat(str(ts))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def in_window(
    records: Iterable[Dict[str, Any]], hours: float = 24.0, now: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)
    out = []
    for r in records:
        t = _parse(r.get("ts"))
        if t is not None and start <= t <= now:
            out.append(r)
    return out


def build_digest(
    records: Iterable[Dict[str, Any]], hours: float = 24.0, now: Optional[datetime] = None
) -> Dict[str, Any]:
    recs = in_window(records, hours, now)
    rep = build_report(recs)
    dq = rep["data_quality"]["snapshots_by_status"]
    total_snaps = sum(dq.values())
    bad = total_snaps - dq.get("VALID", 0)
    ai = rep["ai"]
    reconciles = sum(1 for r in recs if r.get("kind") in ("reconcile", "reconcile_order"))
    cycles = len({r.get("trace_id") for r in recs if r.get("trace_id")})

    alerts: List[str] = []
    if ai["n_calls"] and ai["exhausted"] / ai["n_calls"] > AI_EXHAUSTED_ALERT:
        alerts.append(f"AI: цепочка провайдеров исчерпана в {ai['exhausted']}/{ai['n_calls']} вызовов")
    if total_snaps and bad / total_snaps > BAD_DATA_ALERT:
        alerts.append(f"Данные: {bad}/{total_snaps} снимков не VALID")
    if reconciles >= RECONCILE_ALERT:
        alerts.append(f"Биржа: {reconciles} событий сверки (расхождение/неизвестный исход)")
    if not recs:
        alerts.append("Журнал за окно пуст: бот не писал решений")

    return {
        "window_hours": hours,
        "records": len(recs),
        "cycles": cycles,
        "decisions": rep["signals"].get("n_decisions", 0),
        "executions": rep["execution"],
        "rejections": rep["rejections"].get("n", 0),
        "ai": ai,
        "providers": provider_health(recs),
        "data_quality": dq,
        "reconcile_events": reconciles,
        "alerts": alerts,
    }


def format_digest(d: Dict[str, Any]) -> str:
    ex = d["executions"]
    lines = [
        f"Сводка за {d['window_hours']:g} ч: циклов {d['cycles']}, решений {d['decisions']}, "
        f"попыток входа {ex['n']} (открыто {ex['opened']}), отклонено возможностей {d['rejections']}",
        f"AI: вызовов {d['ai']['n_calls']}, ответили {d['ai']['ok']}, цепочка исчерпана {d['ai']['exhausted']}",
    ]
    for name, p in d["providers"].items():
        lines.append(f"  {name}: ответов {p['answered']}, успех {p['success_rate']}, p95 {p['latency_ms_p95']} мс")
    lines.append(f"Данные: {d['data_quality'] or 'нет записей'}")
    lines.append("Тревоги: " + ("; ".join(d["alerts"]) if d["alerts"] else "нет"))
    return "\n".join(lines)
