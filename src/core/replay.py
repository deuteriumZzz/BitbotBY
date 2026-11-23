"""Replay решений (Phase 4, MASTER_SPEC §3.4, §12.4): воспроизвести расчёт.

Берёт записанные входы одного вызова `SignalCombiner.combine` (ответ AI, сигналы
SAC, направления Chronos, снэпшоты, контекст рынка, настройки), прогоняет тот же
код комбинатора с подставными источниками и сравнивает результат с записанным.
Сеть, AI и модели не нужны — недетерминированные части (ответ LLM, инференс SAC,
прогноз Chronos) берутся из записи, поэтому проверяется именно детерминированная
логика: режимные веса, фильтры, контекстные сигналы, merge, пороги.

Сходимость «бит-в-бит» — по ключам `COMPARE_KEYS` (действие, confidence, трасса
стадий, контекстные флаги). Рассинхрон означает, что код комбинатора изменился
с момента записи или запись неполная.

CLI:  python -m src.core.replay [путь_к_журналу]
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

from src.core.signal_combiner import SignalCombiner

SCHEMA = 1
COMPARE_KEYS = (
    "symbol",
    "action",
    "strategy",
    "confidence",
    "conf_trail",
    "context_flags",
    "context_boost",
    "ctx_penalty",
    "ctx_penalty_raw",
    "chronos_signal",
)


def compact_expected(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Поля рекомендации, по которым сверяется воспроизведение."""
    return {k: copy.deepcopy(rec[k]) for k in COMPARE_KEYS if k in rec}


def build_replay_input(
    inputs: Dict[str, Any],
    snapshots: List[Dict[str, Any]],
    balance: float,
    market_context: Optional[Dict[str, Any]],
    recs: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Собирает запись для replay из `combiner.last_inputs` и результата combine.

    Берутся только символы, по которым есть рекомендации (расчёт по символам
    независим). None — нечего воспроизводить или входы не захвачены.
    """
    symbols = {r.get("symbol") for r in recs if r.get("symbol")}
    settings = inputs.get("settings") or {}
    if not symbols or not settings or settings.get("incomplete"):
        return None
    ctx = market_context or {}
    per_symbol = bool(ctx) and isinstance(next(iter(ctx.values())), dict)
    ctx_subset = {s: ctx[s] for s in symbols if s in ctx} if per_symbol else ctx
    return {
        "schema": SCHEMA,
        "settings": settings,
        "balance": balance,
        "snapshots": [
            copy.deepcopy(s) for s in snapshots if s.get("symbol") in symbols
        ],
        "market_context": copy.deepcopy(ctx_subset) or None,
        "ai_recs": [
            copy.deepcopy(r)
            for r in inputs.get("ai_recs", [])
            if r.get("symbol") in symbols
        ],
        "sac": {s: v for s, v in (inputs.get("sac") or {}).items() if s in symbols},
        "chronos": {
            s: v for s, v in (inputs.get("chronos") or {}).items() if s in symbols
        },
        "expected": [compact_expected(r) for r in recs],
    }


class _ReplayAI:
    unavailable = False

    def __init__(self, recs: List[Dict[str, Any]]) -> None:
        self._recs = recs

    async def analyze(self, snapshots: Any, balance: Any) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._recs)


class _ReplaySAC:
    loaded = True

    def __init__(self, signals: Dict[str, Dict[str, Any]]) -> None:
        self._signals = signals

    def get_signal(
        self,
        snap: Dict[str, Any],
        balance: Any,
        position: float = 0.0,
        entry: float = 0.0,
    ) -> Dict[str, Any]:
        sig = self._signals.get(snap.get("symbol"))
        if sig:
            return dict(sig)
        return {"action": "hold", "confidence": 0.0, "source": "sac"}


class _ReplayForecaster:
    """Направление Chronos по маркеру: первая цена в ряду — индекс символа."""

    def __init__(self, by_marker: Dict[float, str]) -> None:
        self._by_marker = by_marker

    def predict_direction(self, close_prices: Any, threshold_pct: float = 0.002) -> str:
        return self._by_marker.get(close_prices[0], "neutral")


class _ReplayRC:
    def __init__(self, settings: Dict[str, Any]) -> None:
        self._s = settings

    def get_mode(self) -> str:
        return self._s["mode"]

    def get_signal_confidence(self, paper: Any = None) -> float:
        return self._s["min_conf"]

    def get_market_type(self) -> str:
        return self._s["market_type"]

    def get_market_profile(self) -> str:
        return self._s.get("profile", "")

    def get_chronos_enabled(self) -> bool:
        return bool(self._s.get("chronos_enabled"))


async def replay_combine(inp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Повторяет `SignalCombiner.combine` на записанных входах."""
    if inp.get("schema") != SCHEMA:
        raise ValueError(f"неподдерживаемая схема replay: {inp.get('schema')!r}")
    st = inp["settings"]
    symbols = [s["symbol"] for s in inp["snapshots"]]
    markers = {sym: float(i) + 0.5 for i, sym in enumerate(symbols)}
    forecaster = _ReplayForecaster(
        {markers[s]: d for s, d in (inp.get("chronos") or {}).items() if s in markers}
    )
    market_data = {sym: pd.DataFrame({"close": [markers[sym]] * 3}) for sym in symbols}
    combiner = SignalCombiner(
        ai=_ReplayAI(inp.get("ai_recs") or []),
        rc=_ReplayRC(st),
        sac=_ReplaySAC(inp.get("sac") or {}),
        forecaster=forecaster,
    )
    return await combiner.combine(
        copy.deepcopy(inp["snapshots"]),
        inp["balance"],
        regime=st.get("regime", "unknown"),
        regimes=st.get("regimes") or {},
        market_context=copy.deepcopy(inp.get("market_context")),
        sentiment=None,
        market_data=market_data,
    )


def diff_recs(
    expected: List[Dict[str, Any]], actual: List[Dict[str, Any]]
) -> List[str]:
    """Расхождения между записанным и воспроизведённым результатом (пусто = совпало)."""

    def key(r: Dict[str, Any]) -> tuple:
        return (r.get("symbol"), r.get("action"))

    exp = {key(r): r for r in expected}
    act = {key(r): compact_expected(r) for r in actual}
    problems: List[str] = []
    for k in exp.keys() - act.keys():
        problems.append(f"{k}: записано, но не воспроизведено")
    for k in act.keys() - exp.keys():
        problems.append(f"{k}: воспроизведено, но не было записано")
    for k in exp.keys() & act.keys():
        for field in COMPARE_KEYS:
            if exp[k].get(field) != act[k].get(field):
                problems.append(
                    f"{k}.{field}: записано {exp[k].get(field)!r}, "
                    f"воспроизведено {act[k].get(field)!r}"
                )
    return sorted(problems)


async def replay_and_compare(inp: Dict[str, Any]) -> List[str]:
    return diff_recs(inp.get("expected") or [], await replay_combine(inp))


def load_replay_records(path: str) -> List[Dict[str, Any]]:
    """Записи `replay_input` из журнала (включая ротированный `.1`)."""
    out: List[Dict[str, Any]] = []
    for p in (path + ".1", path):
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("kind") == "replay_input" and rec.get("input"):
                    out.append(rec)
    return out


def _run(coro: Any) -> Any:
    """Запускает корутину в приватном цикле, не трогая текущий цикл потока
    (`asyncio.run` обнуляет его, что ломает код, вызывающий main() из тестов)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def main(argv: Optional[List[str]] = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = (
        args[0]
        if args
        else os.getenv("DECISION_LEDGER_PATH", "data/decision_ledger.jsonl")
    )
    records = load_replay_records(path)
    if not records:
        print(f"Нет записей replay_input в {path}")
        return 1
    bad = 0
    for rec in records:
        problems = _run(replay_and_compare(rec["input"]))
        status = "OK" if not problems else f"РАСХОЖДЕНИЕ ({len(problems)})"
        print(f"{rec.get('trace_id', '?')}: {status}")
        for p in problems:
            print(f"    {p}")
        bad += bool(problems)
    print(f"Итого: {len(records)} воспроизведено, расхождений: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
