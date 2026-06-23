"""
E2E-тесты для ключевых методов торгового цикла.

Всё внешнее I/O замокано. Тесты проверяют:
  - _filter_by_balance (чистая функция, без I/O)
  - _execute_top_rec (сигнал → поток ордеров)
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

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


def _make_cfg(paper: bool = True, auto: bool = True, max_pos: int = 3):
    cfg = MagicMock()
    cfg.PAPER_TRADING = paper
    cfg.AUTO_EXECUTE = auto
    cfg.MAX_POSITIONS = max_pos
    cfg.MAX_CORRELATION = 0.0  # отключаем защиту по корреляции в большинстве тестов
    cfg.RISK_PER_TRADE = 0.02
    cfg.INITIAL_BALANCE = 1000.0
    cfg.COMMISSION_RATE = 0.001
    cfg.TIMEFRAME = "15m"
    cfg.SYMBOL = "BTC/USDT"
    cfg.DEFAULT_STRATEGY = "ema_crossover"
    cfg.TELEGRAM_CONFIRM_TIMEOUT = 0
    cfg.SILENT_DEATH_HOURS = 6.0
    cfg.REGIME_CACHE_TTL = "300"
    cfg.CORRELATION_WINDOW = 50
    cfg.MODE = "local"
    cfg.BYBIT_API_KEY = cfg.BYBIT_API_SECRET = ""
    cfg.ANTHROPIC_API_KEY = cfg.DEEPSEEK_API_KEY = cfg.OPENAI_API_KEY = ""
    cfg.TELEGRAM_BOT_TOKEN = ""
    return cfg


def make_bot(balance: float = 1000.0):
    with patch.multiple("src.trading_bot", **_make_patches()):
        with patch("src.trading_bot.Config") as cfg, patch(
            "src.order_executor.Config", cfg
        ):
            cfg.PAPER_TRADING = True
            cfg.AUTO_EXECUTE = True
            cfg.MAX_POSITIONS = 3
            cfg.MAX_CORRELATION = 0.0
            cfg.RISK_PER_TRADE = 0.02
            cfg.INITIAL_BALANCE = balance
            cfg.COMMISSION_RATE = 0.001
            cfg.TIMEFRAME = "15m"
            cfg.SYMBOL = "BTC/USDT"
            cfg.DEFAULT_STRATEGY = "ema_crossover"
            cfg.TELEGRAM_CONFIRM_TIMEOUT = 0
            cfg.SILENT_DEATH_HOURS = 6.0
            cfg.REGIME_CACHE_TTL = "300"
            cfg.CORRELATION_WINDOW = 50
            cfg.MODE = "local"
            cfg.BYBIT_API_KEY = cfg.BYBIT_API_SECRET = ""
            cfg.ANTHROPIC_API_KEY = cfg.DEEPSEEK_API_KEY = cfg.OPENAI_API_KEY = ""
            cfg.TELEGRAM_BOT_TOKEN = ""
            from src.trading_bot import TradingBot

            bot = TradingBot()
            bot._paper_balance = balance
            # current_balance — единый регистр paper-баланса после рефакторинга
            bot.portfolio_manager.current_balance = balance
            bot.trade_history.get_win_rate = AsyncMock(return_value=0.6)
            bot.trade_history.get_trade_count = AsyncMock(return_value=0)
            bot.trade_history.get_expected_value = AsyncMock(return_value=5.0)
            bot.trade_history.record_open = AsyncMock(return_value="trade-001")
            bot.trade_history.record_close = AsyncMock()
            bot.telegram.ask_confirm = AsyncMock(return_value=True)
            bot.telegram.notify = AsyncMock()
            bot.portfolio_manager.update_portfolio = AsyncMock()
            bot.corr_filter.is_allowed = MagicMock(return_value=True)
            bot.corr_filter.max_correlation = MagicMock(return_value=0.0)
            bot.risk_manager.calculate_kelly_size = MagicMock(return_value=0.0)
            bot.api.round_quantity = MagicMock(
                side_effect=lambda sym, qty: round(qty, 6)
            )
            bot.api.set_leverage = AsyncMock(return_value=True)
            bot.api.get_current_price = AsyncMock(return_value=50000.0)
            return bot


@contextmanager
def _patch_cfg(cfg):
    """Патчим Config в trading_bot, order_executor и runtime_config."""
    with patch("src.trading_bot.Config", cfg):
        with patch("src.order_executor.Config", cfg):
            with patch("src.runtime_config.Config", cfg):
                yield cfg


def _buy_rec(sym: str = "BTC/USDT", entry: float = 50000.0, conf: float = 0.85) -> dict:
    return {
        "symbol": sym,
        "action": "buy",
        "strategy": "ema_crossover",
        "confidence": conf,
        "entry": entry,
        "stop_loss": entry * 0.97,
        "take_profit": entry * 1.06,
        "atr": entry * 0.01,
        "alloc_fraction": 0.02,
        "_snap": None,
    }


def _df(entry: float = 50000.0) -> pd.DataFrame:
    return pd.DataFrame({"close": [entry] * 20, "volume": [100.0] * 20})


# ---------------------------------------------------------------------------
# Тесты: _filter_by_balance
# ---------------------------------------------------------------------------


class TestFilterByBalance:
    """_filter_by_balance — чистая функция, без зависимости от Config."""

    def test_affordable_rec_passes(self):
        bot = make_bot(balance=1000.0)
        # 1000 >= 50000 * 0.001 = 50 → проходит
        recs = [_buy_rec(entry=50000.0)]
        result = bot._filter_by_balance(recs, balance=1000.0)
        assert len(result) == 1

    def test_underfunded_rec_filtered(self):
        bot = make_bot(balance=10.0)
        # 10 < 50000 * 0.001 = 50 → фильтруется
        recs = [_buy_rec(entry=50000.0)]
        result = bot._filter_by_balance(recs, balance=10.0)
        assert len(result) == 0

    def test_zero_entry_always_passes(self):
        bot = make_bot()
        recs = [{"symbol": "X/USDT", "action": "buy", "entry": 0}]
        result = bot._filter_by_balance(recs, balance=0.0)
        assert len(result) == 1

    def test_empty_input_returns_empty(self):
        bot = make_bot()
        assert bot._filter_by_balance([], balance=1000.0) == []

    def test_marks_affordable_flag(self):
        bot = make_bot(balance=1000.0)
        recs = [_buy_rec(entry=50000.0)]
        result = bot._filter_by_balance(recs, balance=1000.0)
        assert result[0].get("affordable") is True


# ---------------------------------------------------------------------------
# Тесты: _execute_top_rec
# ---------------------------------------------------------------------------


class TestExecuteTopRec:

    @pytest.mark.asyncio
    async def test_empty_filtered_does_nothing(self):
        bot = make_bot()
        with _patch_cfg(_make_cfg(auto=True)):
            await bot._execute_top_rec([], {})
        assert bot._monitored == {}

    @pytest.mark.asyncio
    async def test_auto_execute_false_shows_dialog_and_executes(self):
        # AUTO_EXECUTE=False → диалог с таймером, авто-исполнение если нет ответа
        bot = make_bot()
        bot.telegram.ask_confirm = AsyncMock(return_value=True)
        with _patch_cfg(_make_cfg(auto=False)):
            with patch(
                "src.trade_history.get_backtest_stats",
                return_value={"win_rate": 0.5, "total_trades": 0, "ev": 0.0},
            ):
                with patch("src.order_executor._ac_impact", return_value=0.0):
                    await bot._execute_top_rec([_buy_rec()], {})
        assert "BTC/USDT" in bot._monitored

    @pytest.mark.asyncio
    async def test_hold_action_skipped(self):
        bot = make_bot()
        rec = {**_buy_rec(), "action": "hold"}
        with _patch_cfg(_make_cfg(auto=True)):
            await bot._execute_top_rec([rec], {})
        assert bot._monitored == {}

    @pytest.mark.asyncio
    async def test_max_positions_guard(self):
        bot = make_bot()
        bot._monitored = {"A/USDT": {}, "B/USDT": {}, "C/USDT": {}}
        with _patch_cfg(_make_cfg(auto=True, max_pos=3)):
            await bot._execute_top_rec([_buy_rec(sym="D/USDT")], {})
        assert "D/USDT" not in bot._monitored

    @pytest.mark.asyncio
    async def test_duplicate_symbol_guard(self):
        bot = make_bot()
        bot._monitored["BTC/USDT"] = {"qty": 0.01, "side": "buy"}
        with _patch_cfg(_make_cfg(auto=True)):
            await bot._execute_top_rec([_buy_rec(sym="BTC/USDT")], {})
        assert len(bot._monitored) == 1

    @pytest.mark.asyncio
    async def test_paper_buy_opens_position(self):
        bot = make_bot(balance=1000.0)
        rec = _buy_rec(sym="BTC/USDT", entry=50000.0)

        with _patch_cfg(_make_cfg(auto=True)):
            with patch(
                "src.trade_history.get_backtest_stats",
                return_value={"win_rate": 0.55, "total_trades": 100, "ev": 3.0},
            ):
                with patch("src.order_executor._ac_impact", return_value=0.0):
                    await bot._execute_top_rec([rec], {"BTC/USDT": _df(50000.0)})

        assert "BTC/USDT" in bot._monitored
        pos = bot._monitored["BTC/USDT"]
        assert pos["side"] == "buy"
        assert pos["qty"] > 0

    @pytest.mark.asyncio
    async def test_telegram_rejection_aborts_trade(self):
        bot = make_bot()
        bot.telegram.ask_confirm = AsyncMock(return_value=False)
        rec = _buy_rec(sym="ETH/USDT", entry=3000.0)

        # AUTO_EXECUTE=False → диалог показывается, Skip отменяет сделку
        with _patch_cfg(_make_cfg(auto=False)):
            with patch(
                "src.trade_history.get_backtest_stats",
                return_value={"win_rate": 0.5, "total_trades": 0, "ev": 0.0},
            ):
                with patch("src.order_executor._ac_impact", return_value=0.0):
                    await bot._execute_top_rec([rec], {})

        assert "ETH/USDT" not in bot._monitored

    @pytest.mark.asyncio
    async def test_last_trade_at_set_on_paper_buy(self):
        bot = make_bot(balance=1000.0)
        assert bot._last_trade_at is None
        rec = _buy_rec(sym="BTC/USDT", entry=50000.0)

        with _patch_cfg(_make_cfg(auto=True)):
            with patch(
                "src.trade_history.get_backtest_stats",
                return_value={"win_rate": 0.55, "total_trades": 100, "ev": 3.0},
            ):
                with patch("src.order_executor._ac_impact", return_value=0.0):
                    await bot._execute_top_rec([rec], {"BTC/USDT": _df()})

        assert bot._last_trade_at is not None
