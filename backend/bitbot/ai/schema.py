"""Строгая валидация рекомендаций LLM.

LLM не источник фактов: любое число в ответе сверяется со snapshot. Строгие
правила: неизвестные поля — отказ, strict enum, конечные числа, символ обязан
быть среди переданных, вход близок к цене snapshot, стоп/тейк стоят по нужную
сторону от входа. Результат — вердикт с кодами причин; сама проверка ничего не
меняет в торговле. В боте она подключена ТОЛЬКО как теневой аудит (пишется в
Ledger), потому что строгий режим отбросил бы ответы, которые прежний парсер
пропускает.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

ALLOWED_KEYS = frozenset(
    {
        "symbol",
        "action",
        "strategy",
        "confidence",
        "entry",
        "stop_loss",
        "take_profit",
        "reasoning",
        "p_win",  # запрос v2: вероятность, что тейк сработает раньше стопа
        "case_against",  # запрос v2: самый сильный довод против сделки
    }
)
REQUIRED_KEYS = ("symbol", "action", "strategy", "confidence")
ACTIONS = ("buy", "sell", "hold")
PRICE_TOLERANCE = 0.02  # |entry/price − 1| ≤ 2% — предложение
MAX_REASONING = 600


def _num(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def validate_recommendation(
    rec: Any,
    snapshots: Dict[str, Dict[str, Any]],
    *,
    price_tol: float = PRICE_TOLERANCE,
    strict_keys: bool = True,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """(очищенная рекомендация | None, коды причин). Пустые причины — рекомендация чистая."""
    if not isinstance(rec, dict):
        return None, ["not_object"]
    missing = [k for k in REQUIRED_KEYS if k not in rec]
    if missing:
        return None, ["missing:" + ",".join(missing)]
    reasons: List[str] = []
    extra = sorted(set(rec) - ALLOWED_KEYS)
    if extra and strict_keys:
        reasons.append("unknown_fields:" + ",".join(extra))
    if rec["action"] not in ACTIONS:
        reasons.append("bad_action")
    conf = _num(rec["confidence"])
    if conf is None or not 0.0 <= conf <= 1.0:
        reasons.append("bad_confidence")
    snap = snapshots.get(rec["symbol"]) if isinstance(rec["symbol"], str) else None
    if snap is None:
        reasons.append("unknown_symbol")

    entry, sl, tp = (_num(rec.get(k)) for k in ("entry", "stop_loss", "take_profit"))
    for key, val in (("entry", entry), ("stop_loss", sl), ("take_profit", tp)):
        if rec.get(key) is not None and (val is None or val <= 0):
            reasons.append(f"bad_{key}")
    price = _num(snap.get("price")) if snap else None
    if entry is not None and price and abs(entry / price - 1) > price_tol:
        reasons.append("entry_off_snapshot")  # число не из snapshot: LLM его придумал
    if None not in (entry, sl, tp) and rec["action"] in ("buy", "sell"):
        ok = sl < entry < tp if rec["action"] == "buy" else tp < entry < sl
        if not ok:
            reasons.append("sl_tp_wrong_side")
    reasoning = rec.get("reasoning")
    if reasoning is not None and (
        not isinstance(reasoning, str) or len(reasoning) > MAX_REASONING
    ):
        reasons.append("bad_reasoning")

    if reasons:
        return None, reasons
    clean = {k: rec[k] for k in rec if k in ALLOWED_KEYS}
    clean["confidence"] = conf
    return clean, []


def audit_response(
    recs: Iterable[Any], snapshots: List[Dict[str, Any]], **kw: Any
) -> Dict[str, Any]:
    """Сводка строгой проверки по ответу: сколько прошло бы и почему отклонено остальное."""
    by_symbol = {s.get("symbol"): s for s in snapshots}
    recs = list(recs)
    reasons: Counter = Counter()
    bad = 0
    for r in recs:
        clean, why = validate_recommendation(r, by_symbol, **kw)
        if clean is None:
            bad += 1
            reasons.update(x.split(":")[0] for x in why)
    return {"n": len(recs), "strict_rejected": bad, "reasons": dict(reasons)}
