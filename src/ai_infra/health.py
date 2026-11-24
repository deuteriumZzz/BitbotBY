"""Здоровье AI-провайдеров по записям Decision Ledger (Phase 10).

Читает записи `ai_call` — подключения к боту не требует. Для каждого провайдера:
сколько раз он отвечал, сколько раз отказал и по какой причине (по `attempts`
цепочки), доля успеха среди попыток, задержка (mean/p95), время последнего
успеха. Стоимость: в журнале нет токенов (`_call` возвращает только текст), поэтому
`estimate_cost` принимает токены от вызывающего и таблицу цен; без цены модель
получает None — число не выдумывается.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional


def _p95(values: List[float]) -> Optional[float]:
    if not values:
        return None
    v = sorted(values)
    return round(v[min(len(v) - 1, int(0.95 * len(v)))], 1)


def provider_health(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[Any, Dict[str, Any]] = defaultdict(
        lambda: {
            "answered": 0,
            "failed": Counter(),
            "latency": [],
            "last_ok_ts": None,
            "tokens": 0,
        }
    )
    for r in records:
        if r.get("kind") != "ai_call":
            continue
        for a in r.get("attempts") or []:  # отказы цепочки до ответившего
            stats[a.get("provider")]["failed"][a.get("status")] += 1
        if r.get("status") == "ok" and r.get("provider"):
            s = stats[r["provider"]]
            s["answered"] += 1
            s["tokens"] += int((r.get("tokens") or {}).get("total") or 0)
            if r.get("latency_ms") is not None:
                s["latency"].append(float(r["latency_ms"]))
            s["last_ok_ts"] = r.get("ts", s["last_ok_ts"])
    out: Dict[str, Dict[str, Any]] = {}
    for name, s in stats.items():
        failed = sum(s["failed"].values())
        tries = s["answered"] + failed
        lat = s["latency"]
        out[str(name)] = {
            "answered": s["answered"],
            "failed": dict(s["failed"]),
            "success_rate": round(s["answered"] / tries, 3) if tries else None,
            "latency_ms_mean": round(sum(lat) / len(lat), 1) if lat else None,
            "latency_ms_p95": _p95(lat),
            "last_ok_ts": s["last_ok_ts"],
            "tokens": s["tokens"],
        }
    return out


def tokens_by_model(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    """Токены по моделям из `ai_call.tokens.total` — вход для `estimate_cost` (модель без токенов не попадает)."""
    out: Dict[str, int] = defaultdict(int)
    for r in records:
        if r.get("kind") == "ai_call" and r.get("status") == "ok":
            total = int((r.get("tokens") or {}).get("total") or 0)
            if total:
                out[str(r.get("model"))] += total
    return dict(out)


def estimate_cost(
    tokens_by_model: Mapping[str, int], usd_per_mtok: Mapping[str, float]
) -> Dict[str, Optional[float]]:
    """Стоимость по моделям; модель без цены в таблице → None, а не 0 и не догадка."""
    return {
        m: (round(t / 1e6 * usd_per_mtok[m], 6) if m in usd_per_mtok else None)
        for m, t in tokens_by_model.items()
    }
