"""Регрессии C-8 и M-5 из AUDIT_REPORT: паузы в ветках защит и символ BTC."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.core.trading_bot import TradingBot


def _df(last_atr: float) -> pd.DataFrame:
    return pd.DataFrame({"atr": [1.0] * 25 + [last_atr]})


def _cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.ATR_SPIKE_LOOKBACK = 20
    cfg.ATR_SPIKE_MULT = 3.0
    cfg.SYMBOL = "X/USDT"  # нет в данных: BTC обязан находиться по своему ключу
    return cfg


def test_atr_spike_checks_btc_even_when_it_is_not_among_first_symbols() -> None:
    bot = TradingBot.__new__(TradingBot)
    data = {
        "A/USDT": _df(1.0),
        "B/USDT": _df(1.0),
        "C/USDT": _df(1.0),
        "BTC/USDT": _df(5.0),
    }
    with patch("src.core.trading_bot.Config", _cfg()):
        assert bot._is_atr_spike(data) is True


def test_atr_spike_false_when_calm() -> None:
    bot = TradingBot.__new__(TradingBot)
    data = {"BTC/USDT": _df(1.2), "A/USDT": _df(1.0)}
    with patch("src.core.trading_bot.Config", _cfg()):
        assert bot._is_atr_spike(data) is False


@pytest.mark.asyncio
async def test_sleep_rest_of_cycle_waits_for_remaining_interval() -> None:
    bot = TradingBot.__new__(TradingBot)
    bot._runtime_config = MagicMock()
    bot._runtime_config.get_trading_interval.return_value = 30
    t0 = asyncio.get_running_loop().time() - 10
    with patch("asyncio.sleep", new_callable=AsyncMock) as sleep:
        await bot._sleep_rest_of_cycle(t0)
    delay = sleep.await_args.args[0]
    assert 19.0 <= delay <= 20.0


@pytest.mark.asyncio
async def test_sleep_rest_of_cycle_never_negative() -> None:
    bot = TradingBot.__new__(TradingBot)
    bot._runtime_config = MagicMock()
    bot._runtime_config.get_trading_interval.return_value = 5
    t0 = asyncio.get_running_loop().time() - 100
    with patch("asyncio.sleep", new_callable=AsyncMock) as sleep:
        await bot._sleep_rest_of_cycle(t0)
    assert sleep.await_args.args[0] == 0.0
