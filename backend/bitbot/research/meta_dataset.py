"""Данные для мета-модели: журнал решений → строки «что сказали источники → чем кончилось».

Решение (`signals`) и итог сделки (`outcome`) связаны `decision_id`. Строка — только
для закрытой и записанной в базу сделки (`recorded`), у которой нашлось решение.
Сделка без решения в журнале (доверенный канал, запись до журнала) идёт в `unmatched`:
счётчик в приложении показывает её отдельно, чтобы пробел в данных был виден сразу.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterator, List, Tuple

from bitbot.storage.decision_ledger import ledger_files

LEDGER_DEFAULT = "data/decision_ledger.jsonl"
NEED = 500  # предложение: 200 на калибровку (meta_model.MIN_CALIBRATION_SAMPLES) + обучение и проверка
_TTL_S = 600.0
_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def ledger_path() -> str:
    return os.getenv("DECISION_LEDGER_PATH", LEDGER_DEFAULT)


def _scan(files: List[str], marker: str) -> Iterator[Dict[str, Any]]:
    """Записи одного вида; по подстроке, чтобы не разбирать JSON всего журнала."""
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if marker in line:
                        try:
                            yield json.loads(line)
                        except ValueError:
                            continue  # обрезанная строка при падении — пропускаем
        except OSError:
            continue


def build(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Строки для обучения и счётчики пробелов."""
    files = ledger_files(path)
    outcomes: Dict[str, Dict[str, Any]] = {}
    no_id = not_recorded = 0
    for rec in _scan(files, '"kind": "outcome"'):
        did = rec.get("decision_id")
        if not did:
            no_id += 1
        elif not rec.get("recorded"):
            not_recorded += 1
        else:
            outcomes[did] = rec
    decisions: Dict[str, Dict[str, Any]] = {}
    for rec in _scan(files, '"kind": "signals"'):
        for d in rec.get("decisions") or []:
            did = d.get("decision_id")
            if did in outcomes:
                decisions[did] = d
    rows = []
    for did, out in outcomes.items():
        d = decisions.get(did)
        if d is None:
            continue
        pnl = float(out.get("pnl") or 0.0)
        rows.append(
            {
                "decision_id": did,
                "ts": out.get("ts"),
                "symbol": out.get("symbol"),
                "mode": d.get("mode"),
                "regime": d.get("regime"),
                "strategy": d.get("strategy"),
                "confidence": d.get("confidence"),
                "outputs": d.get("outputs") or [],
                "conf_trail": d.get("conf_trail") or [],
                "independent_groups": d.get("independent_groups") or [],
                "has_conflict": bool(d.get("conflicts")),
                "pnl": pnl,
                "win": pnl > 0,
            }
        )
    rows.sort(key=lambda r: str(r["ts"]))
    stats = {
        "unmatched": len(outcomes) - len(rows) + no_id,
        "not_recorded": not_recorded,
    }
    return rows, stats


def status(path: str = "") -> Dict[str, Any]:
    """Счётчик для приложения: сколько готовых примеров из нужных `NEED`."""
    path = path or ledger_path()
    hit = _cache.get(path)
    if hit and time.monotonic() - hit[0] < _TTL_S:
        return hit[1]
    rows, stats = build(path)
    res = {
        "have": len(rows),
        "need": NEED,
        "wins": sum(r["win"] for r in rows),
        "last": rows[-1]["ts"] if rows else None,
        **stats,
    }
    # ponytail: кэш в памяти процесса на 10 минут; полный проход по журналу — если счётчик нужен точнее
    _cache[path] = (time.monotonic(), res)
    return res
