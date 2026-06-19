"""
Tests for TradingBot._reconcile_positions.

Covers: stale removal, lost-position recovery, zero-qty skip,
symbol normalisation, API failure handling, paper-trading bypass.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exchange_pos(
    symbol: str,
    side: str = "buy",
    contracts: float = 1.0,
    entry: float = 50000.0,
) -> dict:
    return {"symbol": symbol, "side": side, "contracts": contracts, "entryPrice": entry}


_BOT_PATCHES = dict(
    RedisClient=MagicMock,
    BybitAPI=MagicMock,
    DataLoader=MagicMock,
    PortfolioManager=MagicMock,
    RiskManager=MagicMock,
    MarketScanner=MagicMock,
    NewsAnalyzer=MagicMock,
    AIAnalyzer=MagicMock,
    SignalCombiner=MagicMock,
    RegimeDetector=MagicMock,
    PortfolioOptimizer=MagicMock,
    CorrelationFilter=MagicMock,
    TradeHistory=MagicMock,
    TelegramNotifier=MagicMock,
    TradingStrategy=MagicMock,
)


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


def make_bot(paper: bool = False):
    with patch.multiple("src.trading_bot", **_BOT_PATCHES):
        with patch("src.trading_bot.Config") as cfg:
            _cfg_defaults(cfg)
            cfg.PAPER_TRADING = paper
            from src.trading_bot import TradingBot
            bot = TradingBot()
            bot.api.fetch_positions = AsyncMock(return_value=[])
            return bot


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReconcilePositions:

    @pytest.mark.asyncio
    async def test_paper_trading_skips_fetch(self):
        """In PAPER_TRADING mode reconcile must not call fetch_positions."""
        bot = make_bot()
        with patch("src.trading_bot.Config") as cfg:
            cfg.PAPER_TRADING = True
            await bot._reconcile_positions()
        bot.api.fetch_positions.assert_not_called()

    @pytest.mark.asyncio
    async def test_removes_stale_position(self):
        """Position in _monitored absent from exchange should be removed."""
        bot = make_bot()
        bot._monitored["BTC/USDT"] = {"qty": 0.01, "entry": 50000.0, "side": "buy"}
        bot.api.fetch_positions = AsyncMock(return_value=[])

        await bot._reconcile_positions()

        assert "BTC/USDT" not in bot._monitored

    @pytest.mark.asyncio
    async def test_recovers_lost_position(self):
        """Position on exchange but absent from _monitored should be restored."""
        bot = make_bot()
        bot.api.fetch_positions = AsyncMock(return_value=[
            _exchange_pos("BTC/USDT", side="buy", contracts=0.5, entry=45000.0)
        ])

        await bot._reconcile_positions()

        assert "BTC/USDT" in bot._monitored
        pos = bot._monitored["BTC/USDT"]
        assert pos["qty"] == 0.5
        assert pos["entry"] == 45000.0
        assert pos["side"] == "buy"
        assert pos["trade_id"] is None

    @pytest.mark.asyncio
    async def test_skips_zero_qty_position(self):
        """Exchange position with contracts=0 must not be added to _monitored."""
        bot = make_bot()
        bot.api.fetch_positions = AsyncMock(return_value=[
            _exchange_pos("ETH/USDT", contracts=0.0)
        ])

        await bot._reconcile_positions()

        assert "ETH/USDT" not in bot._monitored

    @pytest.mark.asyncio
    async def test_normalises_colon_symbol(self):
        """ccxt may return 'BTC/USDT:USDT' — must normalise to 'BTC/USDT'."""
        bot = make_bot()
        bot.api.fetch_positions = AsyncMock(return_value=[
            _exchange_pos("BTC/USDT:USDT", contracts=1.0, entry=60000.0)
        ])

        await bot._reconcile_positions()

        assert "BTC/USDT" in bot._monitored
        assert "BTC/USDT:USDT" not in bot._monitored

    @pytest.mark.asyncio
    async def test_api_failure_leaves_monitored_unchanged(self):
        """If fetch_positions raises, _monitored must stay intact."""
        bot = make_bot()
        bot._monitored["SOL/USDT"] = {"qty": 10.0, "entry": 150.0, "side": "buy"}
        bot.api.fetch_positions = AsyncMock(side_effect=Exception("network error"))

        await bot._reconcile_positions()

        assert "SOL/USDT" in bot._monitored

    @pytest.mark.asyncio
    async def test_stale_and_recovered_in_same_call(self):
        """Removes one symbol and recovers another simultaneously."""
        bot = make_bot()
        bot._monitored["OLD/USDT"] = {"qty": 1.0, "entry": 100.0, "side": "buy"}
        bot.api.fetch_positions = AsyncMock(return_value=[
            _exchange_pos("NEW/USDT", contracts=2.0, entry=200.0)
        ])

        await bot._reconcile_positions()

        assert "OLD/USDT" not in bot._monitored
        assert "NEW/USDT" in bot._monitored

    @pytest.mark.asyncio
    async def test_existing_monitored_not_overwritten(self):
        """Position already in _monitored and on exchange must not be touched."""
        bot = make_bot()
        bot._monitored["BTC/USDT"] = {
            "qty": 1.0, "entry": 50000.0, "side": "buy", "stop_loss": 48000.0
        }
        bot.api.fetch_positions = AsyncMock(return_value=[
            _exchange_pos("BTC/USDT", contracts=1.0, entry=51000.0)
        ])

        await bot._reconcile_positions()

        assert bot._monitored["BTC/USDT"]["entry"] == 50000.0
        assert bot._monitored["BTC/USDT"]["stop_loss"] == 48000.0

    @pytest.mark.asyncio
    async def test_negative_qty_skipped(self):
        """Negative contracts value must be treated as zero-size and skipped."""
        bot = make_bot()
        bot.api.fetch_positions = AsyncMock(return_value=[
            _exchange_pos("ADA/USDT", contracts=-0.5)
        ])

        await bot._reconcile_positions()

        assert "ADA/USDT" not in bot._monitored
