"""MODE=local: сделки по локальным стратегиям, объяснение понятно человеку."""

from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from src.ai.experience_buffer import LOCAL_REASONING_PREFIX, is_local_reasoning
from src.core.trading_bot import TradingBot


def _bot(strategy=("breakout", 0.72, {"action": "buy"})):
    bot = TradingBot.__new__(TradingBot)  # без запуска всей инфраструктуры бота
    bot.combiner = MagicMock()
    bot.combiner.combine = AsyncMock(return_value=[])  # local: combiner молчит
    bot.combiner.ai.recommend_strategy_local.return_value = strategy
    bot._cycle = MagicMock()
    bot._cycle.optimize_allocation.side_effect = lambda recs, md: recs
    bot._current_regime = "ranging"
    rc = MagicMock()
    rc.get_signal_confidence.return_value = 0.6
    rc.get_sl_percent.return_value = 0.02
    rc.get_tp_multiplier.return_value = 3.0
    bot._runtime_config = rc
    return bot


async def _run(bot):
    snaps = [{"symbol": "SOL/USDT", "price": 100.0}]
    df = pd.DataFrame({"close": [1.0]})
    return await bot._generate_signals(
        snaps, 1000.0, {"SOL/USDT": "ranging"}, {"SOL/USDT": df}
    )


async def test_local_signal_explains_itself_in_russian():
    (rec,) = await _run(_bot())
    assert rec["strategy"] == "breakout" and rec["action"] == "buy"
    assert rec["reasoning"].startswith(LOCAL_REASONING_PREFIX)
    assert "«breakout»" in rec["reasoning"] and "0.72" in rec["reasoning"]
    assert "боковик" in rec["reasoning"]
    assert is_local_reasoning(rec["reasoning"])
    assert rec["stop_loss"] < rec["entry"] < rec["take_profit"]


async def test_below_min_confidence_makes_no_trade():
    assert await _run(_bot(("breakout", 0.5, {"action": "buy"}))) == []


@pytest.mark.parametrize(
    "mode, fallback, down, trades",
    [
        ("local", False, False, True),  # Local — основной путь, переключатель не влияет
        ("ai", True, True, True),  # AI-цепочка не ответила целиком — решают стратегии
        (
            "ai",
            True,
            False,
            False,
        ),  # AI ответил «входа нет» — стратегии его не перебивают
        ("ai", False, True, False),  # запасной путь выключен: AI лёг — сделок нет
        ("dqn", True, True, True),  # модель SAC не загружена
        ("dqn", True, False, False),  # SAC решил, что входа нет
        ("hybrid", True, True, True),
        ("hybrid", True, False, False),
    ],
)
async def test_local_fallback_only_when_decider_is_down(mode, fallback, down, trades):
    bot = _bot()
    bot._runtime_config.get_mode.return_value = mode
    bot._runtime_config.get_local_fallback.return_value = fallback
    bot.combiner.decider_unavailable.return_value = down
    assert bool(await _run(bot)) is trades


def test_decider_unavailable_by_mode():
    from types import SimpleNamespace

    from src.core.signal_combiner import SignalCombiner

    c = SignalCombiner.__new__(SignalCombiner)
    c.ai = SimpleNamespace(unavailable=True)
    c.sac = SimpleNamespace(loaded=True)
    assert c.decider_unavailable("ai") and c.decider_unavailable("hybrid")
    assert not c.decider_unavailable("dqn") and not c.decider_unavailable("local")
    c.ai.unavailable, c.sac.loaded = False, False
    assert c.decider_unavailable("dqn") and c.decider_unavailable("hybrid")
    assert not c.decider_unavailable("ai")
