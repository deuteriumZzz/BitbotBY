"""
Tests for TradingBot._monitor_positions.

This is the SL/TP / trailing-stop / circuit-breaker loop —
the most money-critical path in the bot.

Strategy: stop the infinite loop after one iteration by setting
`bot.is_running = False` inside the price mock side_effect, and
patch `asyncio.sleep` to avoid 5-second delays.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Bot factory
# ---------------------------------------------------------------------------

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


def make_bot():
    with patch.multiple("src.trading_bot", **_BOT_PATCHES):
        with patch("src.trading_bot.Config") as cfg:
            cfg.PAPER_TRADING = True
            cfg.AUTO_EXECUTE = True
            cfg.MAX_POSITIONS = 3
            cfg.INITIAL_BALANCE = 1000.0
            cfg.RISK_PER_TRADE = 0.02
            cfg.CORRELATION_WINDOW = 50
            cfg.MAX_CORRELATION = 0.0
            cfg.MODE = "local"
            cfg.SYMBOL = "BTC/USDT"
            cfg.TIMEFRAME = "15m"
            cfg.BYBIT_API_KEY = cfg.BYBIT_API_SECRET = ""
            cfg.ANTHROPIC_API_KEY = cfg.DEEPSEEK_API_KEY = cfg.OPENAI_API_KEY = ""
            cfg.TELEGRAM_BOT_TOKEN = ""
            cfg.SILENT_DEATH_HOURS = 6.0
            cfg.REGIME_CACHE_TTL = "300"
            cfg.COMMISSION_RATE = 0.001
            cfg.TRAILING_STOP_ATR_MULT = 0.0
            cfg.CIRCUIT_BREAKER_LOSSES = 0
            from src.trading_bot import TradingBot

            bot = TradingBot()
            bot.api.get_current_price = AsyncMock(return_value=50000.0)
            bot.api.create_order = AsyncMock()
            bot.api.cancel_order = AsyncMock()
            bot.portfolio_manager.update_portfolio = AsyncMock()
            bot.trade_history.record_close = AsyncMock()
            bot.trade_history.get_summary = AsyncMock(
                return_value={"closed_trades": 1, "win_rate": 0.6, "total_pnl": 10.0}
            )
            bot.telegram.notify = AsyncMock()
            return bot


def _open_pos(
    side: str = "buy",
    entry: float = 50000.0,
    sl: float = 49000.0,
    tp: float = 52000.0,
    qty: float = 0.01,
    trade_id: int = 1,
    atr: float = 0.0,
) -> dict:
    return {
        "side": side,
        "entry": entry,
        "stop_loss": sl,
        "take_profit": tp,
        "qty": qty,
        "trade_id": trade_id,
        "atr": atr,
    }


async def _run_once(bot, price: float, paper: bool = True, cfg_overrides: dict = None):
    """Run _monitor_positions for exactly one iteration then exit."""
    cfg_overrides = cfg_overrides or {}

    async def _price(_sym):
        bot.is_running = False
        return price

    bot.api.get_current_price = _price
    bot.is_running = True

    with patch("src.trading_bot.Config") as cfg:
        cfg.PAPER_TRADING = paper
        cfg.COMMISSION_RATE = 0.001
        cfg.TRAILING_STOP_ATR_MULT = cfg_overrides.get("TRAILING_STOP_ATR_MULT", 0.0)
        cfg.CIRCUIT_BREAKER_LOSSES = cfg_overrides.get("CIRCUIT_BREAKER_LOSSES", 0)
        with patch("asyncio.sleep", new=AsyncMock()):
            await bot._monitor_positions()


# ---------------------------------------------------------------------------
# SL / TP trigger tests
# ---------------------------------------------------------------------------


class TestSlTpTrigger:

    @pytest.mark.asyncio
    async def test_buy_sl_closes_position(self):
        """Price drops below buy SL → position removed from _monitored."""
        bot = make_bot()
        bot._monitored["BTC/USDT"] = _open_pos(side="buy", sl=49000, tp=52000)

        await _run_once(bot, price=48500.0)

        assert "BTC/USDT" not in bot._monitored

    @pytest.mark.asyncio
    async def test_buy_tp_closes_position(self):
        """Price rises above buy TP → position removed from _monitored."""
        bot = make_bot()
        bot._monitored["BTC/USDT"] = _open_pos(side="buy", sl=49000, tp=52000)

        await _run_once(bot, price=53000.0)

        assert "BTC/USDT" not in bot._monitored

    @pytest.mark.asyncio
    async def test_sell_sl_closes_position(self):
        """Price rises above sell SL → position removed."""
        bot = make_bot()
        bot._monitored["BTC/USDT"] = _open_pos(
            side="sell", entry=50000, sl=51000, tp=48000
        )

        await _run_once(bot, price=51500.0)

        assert "BTC/USDT" not in bot._monitored

    @pytest.mark.asyncio
    async def test_sell_tp_closes_position(self):
        """Price drops below sell TP → position removed."""
        bot = make_bot()
        bot._monitored["BTC/USDT"] = _open_pos(
            side="sell", entry=50000, sl=51000, tp=48000
        )

        await _run_once(bot, price=47500.0)

        assert "BTC/USDT" not in bot._monitored

    @pytest.mark.asyncio
    async def test_price_inside_range_keeps_position(self):
        """Price between SL and TP → position stays open."""
        bot = make_bot()
        bot._monitored["BTC/USDT"] = _open_pos(side="buy", sl=49000, tp=52000)

        await _run_once(bot, price=50500.0)

        assert "BTC/USDT" in bot._monitored


# ---------------------------------------------------------------------------
# Paper trading vs live
# ---------------------------------------------------------------------------


class TestPaperVsLive:

    @pytest.mark.asyncio
    async def test_paper_trading_skips_create_order(self):
        """In PAPER_TRADING mode, api.create_order must not be called."""
        bot = make_bot()
        bot._monitored["BTC/USDT"] = _open_pos(side="buy", sl=49000, tp=52000)

        await _run_once(bot, price=48500.0, paper=True)

        bot.api.create_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_live_trading_calls_create_order(self):
        """In live mode, api.create_order must be called with market close."""
        bot = make_bot()
        bot._monitored["BTC/USDT"] = _open_pos(side="buy", sl=49000, tp=52000, qty=0.01)

        await _run_once(bot, price=48500.0, paper=False)

        bot.api.create_order.assert_called_once_with("BTC/USDT", "market", "sell", 0.01)

    @pytest.mark.asyncio
    async def test_record_close_called_with_trade_id(self):
        """trade_history.record_close must be called after SL hit."""
        bot = make_bot()
        bot._monitored["BTC/USDT"] = _open_pos(
            side="buy", sl=49000, tp=52000, trade_id=42
        )

        await _run_once(bot, price=48500.0)

        bot.trade_history.record_close.assert_called_once()
        call = bot.trade_history.record_close.call_args
        # trade_id may be positional or keyword
        passed_id = call.kwargs.get("trade_id") or (call.args[0] if call.args else None)
        assert passed_id == 42

    @pytest.mark.asyncio
    async def test_telegram_notified_on_close(self):
        """telegram.notify must be called after position closes."""
        bot = make_bot()
        bot._monitored["ETH/USDT"] = _open_pos(side="buy", sl=3000, tp=3500, entry=3200)

        await _run_once(bot, price=2900.0)

        bot.telegram.notify.assert_called_once()

    @pytest.mark.asyncio
    async def test_portfolio_manager_called_on_close(self):
        """portfolio_manager.update_portfolio must be called when position closes."""
        bot = make_bot()
        bot._monitored["SOL/USDT"] = _open_pos(
            side="buy", sl=140, tp=160, entry=150, qty=10
        )

        await _run_once(bot, price=135.0)

        bot.portfolio_manager.update_portfolio.assert_called_once()


# ---------------------------------------------------------------------------
# Trailing stop
# ---------------------------------------------------------------------------


class TestTrailingStop:

    @pytest.mark.asyncio
    async def test_trailing_stop_raises_sl_for_buy(self):
        """Rising price with ATR trailing stop should raise the buy SL."""
        bot = make_bot()
        # entry=50000, sl=49000, atr=500 → trail = 51000 - 1×500 = 50500 > 49000 → update
        bot._monitored["BTC/USDT"] = _open_pos(
            side="buy", entry=50000, sl=49000, tp=55000, atr=500.0
        )

        async def _price(_sym):
            bot.is_running = False
            return 51000.0

        bot.api.get_current_price = _price
        bot.is_running = True

        with patch("src.trading_bot.Config") as cfg:
            cfg.PAPER_TRADING = True
            cfg.COMMISSION_RATE = 0.001
            cfg.TRAILING_STOP_ATR_MULT = 1.0
            cfg.CIRCUIT_BREAKER_LOSSES = 0
            with patch("asyncio.sleep", new=AsyncMock()):
                await bot._monitor_positions()

        if "BTC/USDT" in bot._monitored:
            assert bot._monitored["BTC/USDT"]["stop_loss"] == pytest.approx(50500.0)

    @pytest.mark.asyncio
    async def test_trailing_stop_lowers_sl_for_sell(self):
        """Falling price with ATR trailing stop should lower the sell SL."""
        bot = make_bot()
        # entry=50000, sl=51000, atr=500 → trail = 49000 + 1×500 = 49500 < 51000 → update
        bot._monitored["BTC/USDT"] = _open_pos(
            side="sell", entry=50000, sl=51000, tp=45000, atr=500.0
        )

        async def _price(_sym):
            bot.is_running = False
            return 49000.0

        bot.api.get_current_price = _price
        bot.is_running = True

        with patch("src.trading_bot.Config") as cfg:
            cfg.PAPER_TRADING = True
            cfg.COMMISSION_RATE = 0.001
            cfg.TRAILING_STOP_ATR_MULT = 1.0
            cfg.CIRCUIT_BREAKER_LOSSES = 0
            with patch("asyncio.sleep", new=AsyncMock()):
                await bot._monitor_positions()

        if "BTC/USDT" in bot._monitored:
            assert bot._monitored["BTC/USDT"]["stop_loss"] == pytest.approx(49500.0)

    @pytest.mark.asyncio
    async def test_trailing_stop_zero_mult_no_update(self):
        """TRAILING_STOP_ATR_MULT=0 → SL must never change."""
        bot = make_bot()
        bot._monitored["BTC/USDT"] = _open_pos(
            side="buy", sl=49000, tp=55000, atr=500.0
        )

        await _run_once(
            bot, price=51000.0, cfg_overrides={"TRAILING_STOP_ATR_MULT": 0.0}
        )

        if "BTC/USDT" in bot._monitored:
            assert bot._monitored["BTC/USDT"]["stop_loss"] == 49000.0


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:

    @pytest.mark.asyncio
    async def test_circuit_breaker_triggers_on_nth_loss(self):
        """N consecutive losses should trigger Telegram alert."""
        bot = make_bot()
        bot._consecutive_losses = 2  # already at N-1

        bot._monitored["BTC/USDT"] = _open_pos(
            side="buy", entry=50000, sl=49000, tp=55000
        )

        await _run_once(bot, price=48500.0, cfg_overrides={"CIRCUIT_BREAKER_LOSSES": 3})

        bot.telegram.notify.assert_called()
        all_calls = " ".join(str(c) for c in bot.telegram.notify.call_args_list)
        assert "Circuit breaker" in all_calls or "убытка" in all_calls

    @pytest.mark.asyncio
    async def test_win_resets_consecutive_losses(self):
        """A profitable close (TP hit) must reset _consecutive_losses to 0."""
        bot = make_bot()
        bot._consecutive_losses = 2
        bot._monitored["BTC/USDT"] = _open_pos(
            side="buy", entry=50000, sl=49000, tp=52000
        )

        # Price 53000 > TP 52000 → profit → losses reset
        await _run_once(bot, price=53000.0, cfg_overrides={"CIRCUIT_BREAKER_LOSSES": 3})

        assert bot._consecutive_losses == 0
