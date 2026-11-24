"""Golden replay: эталон детерминированной торговой логики для рефакторинга.

Прогоняет настоящий код на фиксированных свечах (tests/replay/fixtures, BTC/ETH/SOL
15m) и собирает всё, что бот решает без сети и моделей:

1. индикаторы и снапшот рынка на каждом окне;
2. сигнал каждой из зарегистрированных стратегий;
3. локальный турнир стратегий (с исключением трендовых в боковике и без);
4. комбинатор в режимах ai / hybrid / dqn на сетке подставных ответов AI, SAC и
   Chronos (через bitbot.research.replay);
5. исполнение в paper: какая позиция откроется по рекомендациям комбинатора.

Рефакторинг не должен менять ни одного числа: test_golden сравнивает прогон с
tests/replay/golden.json. Намеренное изменение логики → перезаписать эталон и
разобрать дифф в PR:

    PYTHONPATH=. python3 -m tests.replay.golden --record
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd

HERE = Path(__file__).parent
GOLDEN_PATH = HERE / "golden.json"
SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT")
WINDOWS = tuple(range(220, 421, 20))  # конец окна: 11 окон по 220…420 свечей
# Поля, зависящие от текущего времени, — в эталон не входят.
VOLATILE = {"timestamp", "snapshot_utc", "snapshot_id", "candle_time", "data_quality"}
TRENDING = {
    "trend_following",
    "ema_crossover",
    "macd_crossover",
    "rsi_momentum",
    "swing_trading",
}


def _norm(value: Any) -> Any:
    """Числа до 8 значащих цифр, рекурсивно.

    Эталон пишется на одной версии numpy/pandas, а CI гоняет другую: хвост в
    9–10 знаке там может отличаться. Правка логики меняет числа намного сильнее.
    """
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        f = float(value)
        if not math.isfinite(f):
            return str(f)
        return float(f"{f:.8g}")
    if isinstance(value, dict):
        return {str(k): _norm(v) for k, v in sorted(value.items()) if k not in VOLATILE}
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    if hasattr(value, "item"):  # numpy-скаляр
        return _norm(value.item())
    return str(value)


def _frames() -> Dict[str, pd.DataFrame]:
    return {
        s: pd.read_csv(HERE / "fixtures" / f"{s.replace('/', '_')}_15m.csv")
        for s in SYMBOLS
    }


def _with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    from bitbot.market.data_loader import DataLoader

    quiet = SimpleNamespace(logger=logging.getLogger("golden"))
    return DataLoader.calculate_technical_indicators(quiet, df.copy())


def _rc(mode: str = "local") -> MagicMock:
    rc = MagicMock()
    rc.get_disabled_strategies.return_value = set()
    rc.get_market_type.return_value = "linear"
    rc.get_mode.return_value = mode
    return rc


def market_and_strategies(frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Слои 1–3: индикаторы, снапшоты, сигналы стратегий, локальный турнир."""
    from bitbot.ai.analyzer import AIAnalyzer
    from bitbot.domain.strategies import STRATEGY_REGISTRY, create_strategy
    from bitbot.market.market_scanner import MarketScanner

    scanner = object.__new__(MarketScanner)
    analyzer = AIAnalyzer(runtime_config=_rc())
    out: Dict[str, Any] = {}
    for sym, raw in frames.items():
        rows = []
        for end in WINDOWS:
            df = _with_indicators(raw.iloc[:end])
            signals = {}
            for name in STRATEGY_REGISTRY:
                try:
                    sig = create_strategy(name).generate_signal(df)
                    signals[name] = [sig.get("action"), sig.get("confidence")]
                except Exception as exc:  # падение стратегии — тоже часть поведения
                    signals[name] = ["error", type(exc).__name__]
            best = analyzer.recommend_strategy_local(df)
            best_ranging = analyzer.recommend_strategy_local(df, exclude=TRENDING)
            rows.append(
                {
                    "end": end,
                    "indicators": df.iloc[-1].drop("timestamp").to_dict(),
                    "snapshot": scanner.build_snapshot(sym, df, 0.0, []),
                    "signals": signals,
                    "local": [best[0], best[1], best[2].get("action")],
                    "local_ranging": [
                        best_ranging[0],
                        best_ranging[1],
                        best_ranging[2].get("action"),
                    ],
                }
            )
        out[sym] = rows
    return out


def _scenarios(snapshots: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Сетка входов комбинатора: режим × AI × SAC × Chronos × режим рынка."""
    out = []
    for sym, snap in snapshots.items():
        price, atr = float(snap["price"]), float(snap["atr"])
        for mode in ("ai", "hybrid", "dqn"):
            for ai_action in ("buy", "sell", None):
                for sac_action, sac_conf in (
                    ("buy", 0.8),
                    ("sell", 0.75),
                    ("hold", 0.0),
                ):
                    for chronos in ("up", "down", "neutral"):
                        for regime in ("trending_up", "ranging"):
                            ai_recs = []
                            if ai_action:
                                sign = 1 if ai_action == "buy" else -1
                                ai_recs.append(
                                    {
                                        "symbol": sym,
                                        "action": ai_action,
                                        "strategy": "ema_crossover",
                                        "confidence": 0.82,
                                        "entry": price,
                                        "stop_loss": price - sign * 2 * atr,
                                        "take_profit": price + sign * 4 * atr,
                                        "reasoning": "golden",
                                    }
                                )
                            out.append(
                                {
                                    "schema": 1,
                                    "settings": {
                                        "mode": mode,
                                        "min_conf": 0.55,
                                        "market_type": "linear",
                                        "profile": "bluechip",
                                        "chronos_enabled": True,
                                        "regime": regime,
                                        "regimes": {sym: regime},
                                    },
                                    "balance": 10_000.0,
                                    "snapshots": [dict(snap)],
                                    "market_context": None,
                                    "ai_recs": ai_recs,
                                    "sac": {
                                        sym: {
                                            "action": sac_action,
                                            "confidence": sac_conf,
                                        }
                                    },
                                    "chronos": {sym: chronos},
                                }
                            )
    return out


async def combiner(snapshots: Dict[str, Dict[str, Any]]) -> List[Any]:
    """Слой 4: комбинатор на сетке сценариев."""
    from bitbot.research.replay import replay_combine

    results = []
    for inp in _scenarios(snapshots):
        recs = await replay_combine(inp)
        st = inp["settings"]
        results.append(
            {
                "in": [
                    inp["snapshots"][0]["symbol"],
                    st["mode"],
                    inp["ai_recs"][0]["action"] if inp["ai_recs"] else None,
                    next(iter(inp["sac"].values()))["action"],
                    next(iter(inp["chronos"].values())),
                    st["regime"],
                ],
                "recs": [
                    {k: r.get(k) for k in sorted(r) if not k.startswith("_")}
                    for r in recs
                ],
            }
        )
    return results


async def execution(combined: List[Any]) -> List[Any]:
    """Слой 5: позиция, которую откроет paper-исполнитель по рекомендации."""
    from tests.core.test_order_executor_extra import (
        _BT_STATS,
        _make_cfg,
        _make_executor,
    )

    picked = [
        (c["in"], r)
        for c in combined
        if c["in"][1] == "hybrid"
        for r in c["recs"]
        if r.get("action") in ("buy", "sell")
    ][:24]
    results = []
    for scenario, rec in picked:
        executor, monitored, _ = _make_executor()
        price = float(rec["entry"])
        executor._api.exchange.fetch_ticker = AsyncMock(
            return_value={
                "bid": price * 0.9999,
                "ask": price * 1.0001,
                "quoteVolume": 5e7,
            }
        )
        with (
            patch("bitbot.execution.order_executor.Config", _make_cfg(paper=True)),
            patch("bitbot.execution.order_executor._ac_impact", return_value=0.0),
            patch(
                "bitbot.storage.trade_history.get_backtest_stats",
                return_value=_BT_STATS,
            ),
        ):
            await executor.execute(dict(rec, atr=rec.get("atr", 1.0)), {}, 10_000.0)
        pos = monitored.get(rec["symbol"])
        fields = {}
        if pos is not None:
            raw = pos if isinstance(pos, dict) else getattr(pos, "__dict__", {})
            fields = {
                k: v
                for k, v in raw.items()
                if not (
                    k.endswith("_at") or k.endswith("_time") or k in ("opened", "id")
                )
            }
        results.append({"in": scenario, "position": fields})
    return results


def _run(coro: Any) -> Any:
    """Свой loop вместо asyncio.run: тот в Python 3.9 оставляет поток без
    текущего loop, и следующие синхронные тесты падают."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def build() -> Dict[str, Any]:
    frames = _frames()
    market = market_and_strategies(frames)
    last_snaps = {s: rows[-1]["snapshot"] for s, rows in market.items()}
    combined = _run(combiner(last_snaps))
    executed = _run(execution(combined))
    return _norm(
        {"market": market, "combiner": combined, "execution": executed, "version": 1}
    )


def main(argv: List[str]) -> int:
    data = build()
    if "--record" in argv:
        GOLDEN_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"recorded {GOLDEN_PATH}")
        return 0
    print(json.dumps(data, ensure_ascii=False)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
