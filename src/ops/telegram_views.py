"""Тексты для Telegram-команд /digest, /audit, /dbcheck (только чтение, без Markdown).

Блокирующие функции (чтение журнала до 50 МБ, проверка БД) — вызывать через
`asyncio.to_thread`. Ничего не меняют: ни журнал, ни БД, ни состояние бота.
Telegram-лимит сообщения 4096 символов — длинный текст обрезается.
"""

from __future__ import annotations

import os
from typing import List, Optional

from src.infra import db_health
from src.ops.daily_digest import build_digest, format_digest
from src.ops.decision_audit import audit_decision, format_audit
from src.research.signal_intelligence_report import load_records

LIMIT = 3900
LEDGER_DEFAULT = "data/decision_ledger.jsonl"


def _ledger_path() -> str:
    return os.getenv("DECISION_LEDGER_PATH", LEDGER_DEFAULT)


def _cut(text: str) -> str:
    return text if len(text) <= LIMIT else text[: LIMIT - 20] + "\n…(обрезано)"


def parse_hours(arg: Optional[str], default: float = 24.0) -> float:
    """Окно сводки в часах из аргумента команды: 1–168, мусор → значение по умолчанию."""
    if arg is None:
        return default
    try:
        return min(168.0, max(1.0, float(arg)))
    except ValueError:
        return default


def digest_text(hours: float = 24.0, ledger_path: Optional[str] = None) -> str:
    recs = load_records(ledger_path or _ledger_path())
    if not recs:
        return "Журнал решений пуст или не найден: сводить нечего."
    return _cut(format_digest(build_digest(recs, hours)))


def audit_text(decision_id: str, ledger_path: Optional[str] = None) -> str:
    decision_id = (decision_id or "").strip()
    if not decision_id:
        return "Использование: /audit <decision_id>"
    recs = load_records(ledger_path or _ledger_path())
    hit = audit_decision(recs, decision_id)
    if hit is None:
        return f"Решение {decision_id} не найдено в журнале."
    return _cut(format_audit(hit))


def dbcheck_text(paths: Optional[List[str]] = None) -> str:
    reports = db_health.check_all(paths, quick=True)
    if not reports:
        return "Файлов БД сделок не найдено."
    lines = []
    for r in reports:
        mark = "✅" if r["ok"] else "⚠️"
        if r["ok"]:
            detail = ", ".join(f"{t}={n}" for t, n in r["tables"].items())
        else:
            detail = r["error"] or "; ".join(r["messages"])
        lines.append(
            f"{mark} {os.path.basename(r['path'])}: {r.get('size_mb')} МБ, WAL {r.get('wal_mb')} МБ — {detail}"
        )
    lines.append("Проверка по снимку; боевой файл не затронут.")
    return _cut("\n".join(lines))
