"""Golden replay: торговая логика после рефакторинга решает так же, как до него.

Эталон — tests/replay/golden.json (как пересобрать — в golden.py). Падение
значит, что изменилось решение бота; тест печатает первые расхождения по пути.
"""

from __future__ import annotations

import json
from typing import Any, List

from tests.replay import golden


def _diff(old: Any, new: Any, path: str, out: List[str], limit: int = 20) -> None:
    if len(out) >= limit:
        return
    if isinstance(old, dict) and isinstance(new, dict):
        for k in sorted(set(old) | set(new)):
            if k not in old or k not in new:
                out.append(f"{path}.{k}: {'добавлен' if k in new else 'пропал'}")
            else:
                _diff(old[k], new[k], f"{path}.{k}", out, limit)
    elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        for i, (a, b) in enumerate(zip(old, new)):
            _diff(a, b, f"{path}[{i}]", out, limit)
    elif old != new:
        out.append(
            f"{path}: было {json.dumps(old, ensure_ascii=False)[:120]}, "
            f"стало {json.dumps(new, ensure_ascii=False)[:120]}"
        )


def test_trading_logic_matches_golden() -> None:
    expected = json.loads(golden.GOLDEN_PATH.read_text(encoding="utf-8"))
    actual = json.loads(json.dumps(golden.build(), sort_keys=True))
    problems: List[str] = []
    _diff(expected, actual, "golden", problems)
    assert not problems, "Решения бота разошлись с эталоном:\n" + "\n".join(problems)


def test_golden_detects_a_changed_threshold(monkeypatch) -> None:
    """Страховка самой страховки: правка порога стратегии ловится."""
    from bitbot.domain.strategies import STRATEGY_REGISTRY

    name = next(iter(STRATEGY_REGISTRY))
    real = STRATEGY_REGISTRY[name].generate_signal

    def shifted(self, data):
        sig = dict(real(self, data))
        sig["confidence"] = float(sig.get("confidence") or 0.0) + 0.01
        return sig

    monkeypatch.setattr(STRATEGY_REGISTRY[name], "generate_signal", shifted)
    market = golden.market_and_strategies({"BTC/USDT": golden._frames()["BTC/USDT"]})
    expected = json.loads(golden.GOLDEN_PATH.read_text(encoding="utf-8"))
    got = golden._norm(market)["BTC/USDT"][0]["signals"][name]
    assert got != expected["market"]["BTC/USDT"][0]["signals"][name]
