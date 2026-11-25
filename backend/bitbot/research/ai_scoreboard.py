"""Табло честности AI: сбываются ли прогнозы провайдера лучше, чем монетка.

Запрос v2 (флаг AI_FORECAST_PROMPT) просит у модели вероятности, которые можно проверить:
`p_up` — цена сначала пройдёт +ATR, а не −ATR, за `horizon` свечей; `p_win` у сделки —
тейк раньше стопа. Каждый прогноз из журнала (`ai_forecast`) сверяется со свечами после
него. Мера — Brier (средний квадрат ошибки вероятности) против «климата»: прогноза
«всегда доля исходов вверх». skill > 0 — модель лучше этого, ≤ 0 — нет.

Барьер не достигнут за горизонт или оба в одной свече — исход неизвестен, прогноз не
считается (но запоминается, чтобы не спрашивать биржу снова).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

from bitbot.storage.decision_ledger import ledger_files

BOARD_PATH = "data/ai_scoreboard.json"
MIN_N = 200  # предложение: столько проверенных прогнозов до вердикта
SKILL_MARGIN = 0.02  # предложение: «лучше монетки» — skill выше этого
MAX_FETCH = 150  # запросов свечей за один проход
_TF_S = {"m": 60, "h": 3600, "d": 86400}

Fetch = Callable[[str, str, int, int], Awaitable[List[List[float]]]]


def tf_seconds(tf: str) -> int:
    return int(tf[:-1]) * _TF_S[tf[-1]]


def first_touch(candles: Iterable[List[float]], up: float, down: float) -> str:
    """Какой барьер цена задела первой: up, down; both — в одной свече; none — ни один."""
    for c in candles:
        high, low = float(c[2]), float(c[3])
        hit_up, hit_down = high >= up, low <= down
        if hit_up and hit_down:
            return "both"
        if hit_up:
            return "up"
        if hit_down:
            return "down"
    return "none"


def _outcome(touch: str, win_on: str) -> Optional[int]:
    return None if touch in ("both", "none") else int(touch == win_on)


def resolve(
    rec: Dict[str, Any], candles: Dict[str, List[List[float]]]
) -> Dict[str, Dict[str, Any]]:
    """Исходы прогнозов одной записи `ai_forecast` по свечам после неё (ключ → строка)."""
    out: Dict[str, Dict[str, Any]] = {}
    m = float(rec.get("atr_mult") or 1.0)
    for it in rec.get("items") or []:
        bars, price, atr = candles.get(it["symbol"]), it.get("price"), it.get("atr")
        if bars is None or not price or not atr:
            continue
        touch = first_touch(bars, price + m * atr, price - m * atr)
        out[f"{rec['ts']}|{it['symbol']}|up"] = {
            "provider": rec.get("provider"),
            "kind": "forecast",
            "p": float(it["p_up"]),
            "y": _outcome(touch, "up"),
        }
    for tr in rec.get("trades") or []:
        bars = candles.get(tr["symbol"])
        sl, tp, p = tr.get("stop_loss"), tr.get("take_profit"), tr.get("p_win")
        if bars is None or not sl or not tp or p is None:
            continue
        buy = tr.get("action") == "buy"
        touch = first_touch(bars, tp if buy else sl, sl if buy else tp)
        out[f"{rec['ts']}|{tr['symbol']}|trade"] = {
            "provider": rec.get("provider"),
            "kind": "trade",
            "p": float(p),
            "y": _outcome(touch, "up" if buy else "down"),
        }
    return out


def score(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """По провайдерам: число проверок, Brier, Brier «климата», skill и вердикт словами."""
    groups: Dict[Tuple[str, str], List[Tuple[float, int]]] = {}
    for r in rows:
        if r.get("y") is not None:
            key = (str(r["provider"]), str(r["kind"]))
            groups.setdefault(key, []).append((float(r["p"]), int(r["y"])))
    out: Dict[str, Dict[str, Any]] = {}
    for (prov, kind), pairs in groups.items():
        n = len(pairs)
        base = sum(y for _, y in pairs) / n
        brier = sum((p - y) ** 2 for p, y in pairs) / n
        brier_base = base * (1 - base)
        skill = 1 - brier / brier_base if brier_base > 0 else 0.0
        if n < MIN_N:
            verdict = "мало данных"
        elif skill > SKILL_MARGIN:
            verdict = "лучше монетки"
        else:
            verdict = "не лучше монетки"
        # ponytail: порог skill без доверительного интервала; bootstrap — если вердикты начнут скакать
        out.setdefault(prov, {})[kind] = {
            "n": n,
            "hit_rate": round(base, 3),
            "brier": round(brier, 4),
            "brier_base": round(brier_base, 4),
            "skill": round(skill, 3),
            "verdict": verdict,
        }
    return out


def untrusted(summary: Dict[str, Dict[str, Any]]) -> List[str]:
    """Провайдеры, чьи прогнозы на MIN_N+ проверках не лучше монетки."""
    return sorted(
        p
        for p, kinds in summary.items()
        if (kinds.get("forecast") or {}).get("verdict") == "не лучше монетки"
    )


def _load(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            board = json.load(f)
        return board if isinstance(board, dict) else {}
    except (OSError, ValueError):
        return {}


def read(path: str = BOARD_PATH) -> Dict[str, Any]:
    """Итог для приложения: сводка по провайдерам и когда обновлено."""
    board = _load(path)
    return {
        "updated": board.get("updated"),
        "providers": board.get("summary") or {},
        "min_n": MIN_N,
    }


def _ts(rec: Dict[str, Any]) -> float:
    return datetime.fromisoformat(str(rec["ts"])).timestamp()


async def update(
    ledger_path: str, fetch: Fetch, path: str = BOARD_PATH
) -> Dict[str, Any]:
    """Проверяет созревшие прогнозы из журнала по свечам и пересчитывает сводку."""
    board = _load(path)
    resolved: Dict[str, Dict[str, Any]] = board.get("resolved") or {}
    now, fetched = time.time(), 0
    for f in ledger_files(ledger_path):
        with open(f, encoding="utf-8") as fh:
            lines = [ln for ln in fh if '"kind": "ai_forecast"' in ln]
        for line in lines:
            try:
                rec = json.loads(line)
                tf, horizon, at = str(rec["timeframe"]), int(rec["horizon"]), _ts(rec)
            except (ValueError, KeyError):
                continue
            ripe = at + tf_seconds(tf) * (horizon + 1) < now
            syms = {
                x["symbol"]
                for x in (rec.get("items") or []) + (rec.get("trades") or [])
                if f"{rec['ts']}|{x['symbol']}|up" not in resolved
                and f"{rec['ts']}|{x['symbol']}|trade" not in resolved
            }
            if not ripe or not syms or fetched >= MAX_FETCH:
                continue
            candles, since = {}, int(at * 1000)
            for sym in sorted(syms):
                fetched += 1
                try:
                    bars = await fetch(sym, tf, since, horizon + 1)
                except (
                    Exception
                ):  # noqa: BLE001 — сеть, монета ушла с биржи: попробуем позже
                    continue
                # только свечи, открытые после прогноза: без заглядывания вперёд
                candles[sym] = [b for b in bars if b[0] >= since][:horizon]
            resolved.update(resolve(rec, candles))
    summary = score(resolved.values())
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "resolved": resolved,
                "summary": summary,
                "updated": datetime.now(timezone.utc).isoformat(),
            },
            fh,
            ensure_ascii=False,
        )
    os.replace(tmp, path)
    return summary
