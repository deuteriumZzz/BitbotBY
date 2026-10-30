"""
Тесты для TradingBot._check_funding_arb — подключение FundingArbDetector
к уже вычисленному funding_rate из MarketContext (без влияния на сделки).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_PATCH_KEYS = [
    "RedisClient",
    "BybitAPI",
    "DataLoader",
    "PortfolioManager",
    "RiskManager",
    "MarketScanner",
    "NewsAnalyzer",
    "AIAnalyzer",
    "SignalCombiner",
    "RegimeDetector",
    "PortfolioOptimizer",
    "CorrelationFilter",
    "TradeHistory",
    "TelegramNotifier",
    "TradingStrategy",
]


def _make_patches() -> dict:
    return {k: MagicMock() for k in _PATCH_KEYS}


def _cfg_defaults(cfg):
    cfg.PAPER_TRADING = False
    cfg.INITIAL_BALANCE = 1000.0
    cfg.RISK_PER_TRADE = 0.02
    cfg.CORRELATION_WINDOW = 50
    cfg.MAX_CORRELATION = 0.7
    cfg.MODE = "local"
    cfg.BYBIT_API_KEY = cfg.BYBIT_API_SECRET = ""
    cfg.ANTHROPIC_API_KEY = cfg.DEEPSEEK_API_KEY = ""
    cfg.OPENAI_API_KEY = ""
    cfg.TELEGRAM_BOT_TOKEN = ""
    cfg.SILENT_DEATH_HOURS = 6.0
    cfg.REGIME_CACHE_TTL = "300"
    cfg.STOP_LOSS_PERCENT = 0.05


def make_bot():
    with patch.multiple("src.core.trading_bot", **_make_patches()):
        with patch("src.core.trading_bot.Config") as cfg:
            _cfg_defaults(cfg)
            from src.core.trading_bot import TradingBot

            return TradingBot()


class TestCheckFundingArb:
    @pytest.mark.asyncio
    async def test_calls_check_and_alert_per_symbol(self):
        bot = make_bot()
        bot._funding_arb_detector.check_and_alert = AsyncMock()
        market_ctx_map = {
            "BTC/USDT": {"funding_rate": 0.0008},
            "ETH/USDT": {"funding_rate": -0.0012},
        }
        await bot._check_funding_arb(market_ctx_map)
        assert bot._funding_arb_detector.check_and_alert.await_count == 2
        bot._funding_arb_detector.check_and_alert.assert_any_await(
            "BTC/USDT", 0.0008, bot.telegram
        )
        bot._funding_arb_detector.check_and_alert.assert_any_await(
            "ETH/USDT", -0.0012, bot.telegram
        )

    @pytest.mark.asyncio
    async def test_skips_symbol_without_funding_rate(self):
        bot = make_bot()
        bot._funding_arb_detector.check_and_alert = AsyncMock()
        await bot._check_funding_arb({"BTC/USDT": {}})
        bot._funding_arb_detector.check_and_alert.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_map_is_noop(self):
        bot = make_bot()
        bot._funding_arb_detector.check_and_alert = AsyncMock()
        await bot._check_funding_arb({})
        bot._funding_arb_detector.check_and_alert.assert_not_called()
