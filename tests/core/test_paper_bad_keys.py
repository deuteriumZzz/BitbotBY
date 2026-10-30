"""Paper не должен падать из-за неверного или просроченного ключа Bybit."""

from unittest.mock import AsyncMock, MagicMock, patch

import ccxt
import pytest

from src.core.trading_bot import TradingBot


def _bot():
    bot = object.__new__(TradingBot)
    bot.api = MagicMock()
    bot.data_loader = MagicMock()
    return bot


@pytest.mark.parametrize("target", ["api", "data_loader"])
async def test_paper_falls_back_to_public_data_on_auth_error(target):
    bot = _bot()
    obj = getattr(bot, target)
    obj.initialize = AsyncMock(side_effect=[ccxt.AuthenticationError("expired"), None])
    with patch("src.core.trading_bot.Config") as cfg:
        cfg.PAPER_TRADING = True
        cfg.TESTNET = False
        cfg.BYBIT_API_KEY, cfg.BYBIT_API_SECRET = "k", "s"
        if target == "api":
            await bot._connect_exchange()
        else:
            await bot._connect_data_loader()
    assert obj.initialize.await_count == 2
    assert obj.initialize.await_args_list[1].args[:2] == ("", "")


@pytest.mark.parametrize("target", ["api", "data_loader"])
async def test_live_still_fails_on_auth_error(target):
    bot = _bot()
    obj = getattr(bot, target)
    obj.initialize = AsyncMock(side_effect=ccxt.AuthenticationError("expired"))
    with patch("src.core.trading_bot.Config") as cfg:
        cfg.PAPER_TRADING = False
        cfg.TESTNET = False
        cfg.BYBIT_API_KEY, cfg.BYBIT_API_SECRET = "k", "s"
        with pytest.raises(ccxt.AuthenticationError):
            if target == "api":
                await bot._connect_exchange()
            else:
                await bot._connect_data_loader()
    assert obj.initialize.await_count == 1
