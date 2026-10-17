"""MODE=local: сделки по локальным стратегиям, объяснение понятно человеку."""

from unittest.mock import AsyncMock, MagicMock

import pandas as pd

from src.experience_buffer import LOCAL_REASONING_PREFIX, is_local_reasoning
from src.trading_bot import TradingBot


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
