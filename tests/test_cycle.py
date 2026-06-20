"""
Тесты для src/cycle.py — CycleRunner.

Мокируются: NewsAnalyzer, MarketScanner, PortfolioOptimizer, TelegramNotifier.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from src.cycle import CycleRunner

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def make_runner(get_regime=None) -> CycleRunner:
    """Создаёт CycleRunner с замоканными зависимостями."""
    news = MagicMock()
    scanner = MagicMock()
    optimizer = MagicMock()
    telegram = MagicMock()
    telegram.notify = AsyncMock()
    regime_fn = get_regime or (lambda: "trending")

    return CycleRunner(
        news=news,
        scanner=scanner,
        portfolio_optimizer=optimizer,
        telegram=telegram,
        get_current_regime=regime_fn,
    )


def make_ohlcv(n: int = 60) -> pd.DataFrame:
    """Минимальный OHLCV DataFrame для тестов."""
    import numpy as np

    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "close": prices,
            "open": prices * 0.999,
            "high": prices * 1.002,
            "low": prices * 0.998,
            "volume": np.random.uniform(100, 500, n),
        }
    )


# ---------------------------------------------------------------------------
# collect_snapshots
# ---------------------------------------------------------------------------


class TestCollectSnapshots:
    """Тесты сбора снимков рыночных данных."""

    @pytest.mark.asyncio
    async def test_collect_snapshots_returns_list(self):
        """collect_snapshots возвращает список снимков."""
        runner = make_runner()
        runner._news.get_sentiment = AsyncMock(return_value=(0.5, ["headline"]))
        runner._scanner.build_snapshot = MagicMock(return_value={"symbol": "BTC/USDT"})

        df = make_ohlcv()
        result = await runner.collect_snapshots(["BTC/USDT"], {"BTC/USDT": df})
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["symbol"] == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_collect_snapshots_skips_missing_market_data(self):
        """Символ без market_data пропускается — не попадает в результат."""
        runner = make_runner()
        runner._news.get_sentiment = AsyncMock(return_value=(0.3, []))
        runner._scanner.build_snapshot = MagicMock(return_value={"symbol": "ETH/USDT"})

        result = await runner.collect_snapshots(["ETH/USDT"], {})
        assert result == []

    @pytest.mark.asyncio
    async def test_collect_snapshots_handles_news_exception(self):
        """Исключение в get_sentiment не прерывает обработку — sentiment=0.0."""
        runner = make_runner()
        runner._news.get_sentiment = AsyncMock(side_effect=Exception("API error"))
        snap = {"symbol": "BTC/USDT"}
        runner._scanner.build_snapshot = MagicMock(return_value=snap)

        df = make_ohlcv()
        await runner.collect_snapshots(["BTC/USDT"], {"BTC/USDT": df})
        # build_snapshot вызывается с sentiment=0.0 и пустыми headlines
        runner._scanner.build_snapshot.assert_called_once()
        call_args = runner._scanner.build_snapshot.call_args[0]
        assert call_args[2] == 0.0  # sentiment
        assert call_args[3] == []  # headlines

    @pytest.mark.asyncio
    async def test_collect_snapshots_skips_empty_dataframe(self):
        """Пустой DataFrame для символа пропускается."""
        runner = make_runner()
        runner._news.get_sentiment = AsyncMock(return_value=(0.1, []))
        runner._scanner.build_snapshot = MagicMock(return_value={"symbol": "BTC/USDT"})

        result = await runner.collect_snapshots(
            ["BTC/USDT"], {"BTC/USDT": pd.DataFrame()}
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_collect_snapshots_skips_none_from_build_snapshot(self):
        """Если build_snapshot вернул None — результат не добавляется."""
        runner = make_runner()
        runner._news.get_sentiment = AsyncMock(return_value=(0.5, ["h"]))
        runner._scanner.build_snapshot = MagicMock(return_value=None)

        df = make_ohlcv()
        result = await runner.collect_snapshots(["BTC/USDT"], {"BTC/USDT": df})
        assert result == []

    @pytest.mark.asyncio
    async def test_collect_snapshots_multiple_symbols(self):
        """collect_snapshots обрабатывает несколько символов параллельно."""
        runner = make_runner()
        runner._news.get_sentiment = AsyncMock(return_value=(0.5, []))
        runner._scanner.build_snapshot = MagicMock(
            side_effect=lambda sym, df, sent, hl: {"symbol": sym}
        )

        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        market_data = {s: make_ohlcv() for s in symbols}
        result = await runner.collect_snapshots(symbols, market_data)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# optimize_allocation
# ---------------------------------------------------------------------------


class TestOptimizeAllocation:
    """Тесты распределения позиций CVaR-оптимизатором."""

    def test_single_buy_uses_default_risk(self):
        """Один buy-сигнал получает alloc_fraction = RISK_PER_TRADE."""
        from config import Config

        runner = make_runner()
        recs = [{"symbol": "BTC/USDT", "action": "buy"}]
        result = runner.optimize_allocation(recs, {"BTC/USDT": make_ohlcv(60)})
        assert result[0]["alloc_fraction"] == Config.RISK_PER_TRADE

    def test_sell_signal_gets_default_alloc(self):
        """sell-сигнал получает alloc_fraction = RISK_PER_TRADE."""
        from config import Config

        runner = make_runner()
        recs = [{"symbol": "BTC/USDT", "action": "sell"}]
        result = runner.optimize_allocation(recs, {"BTC/USDT": make_ohlcv(60)})
        assert result[0]["alloc_fraction"] == Config.RISK_PER_TRADE

    def test_two_buys_calls_optimizer(self):
        """При >= 2 buy-сигналах вызывается PortfolioOptimizer.allocate()."""
        runner = make_runner()
        runner._optimizer.allocate = MagicMock(
            return_value={"BTC/USDT": 0.03, "ETH/USDT": 0.02}
        )
        recs = [
            {"symbol": "BTC/USDT", "action": "buy"},
            {"symbol": "ETH/USDT", "action": "buy"},
        ]
        market_data = {
            "BTC/USDT": make_ohlcv(60),
            "ETH/USDT": make_ohlcv(60),
        }
        result = runner.optimize_allocation(recs, market_data)
        runner._optimizer.allocate.assert_called_once()
        btc_rec = next(r for r in result if r["symbol"] == "BTC/USDT")
        assert "alloc_fraction" in btc_rec

    def test_alloc_fraction_capped_at_risk_per_trade_times_3(self):
        """alloc_fraction кэпается на RISK_PER_TRADE * 3."""
        from config import Config

        runner = make_runner()
        runner._optimizer.allocate = MagicMock(
            return_value={
                "BTC/USDT": 1.0,
                "ETH/USDT": 1.0,
            }  # нереально большое значение
        )
        recs = [
            {"symbol": "BTC/USDT", "action": "buy"},
            {"symbol": "ETH/USDT", "action": "buy"},
        ]
        market_data = {
            "BTC/USDT": make_ohlcv(60),
            "ETH/USDT": make_ohlcv(60),
        }
        result = runner.optimize_allocation(recs, market_data)
        for r in result:
            assert r["alloc_fraction"] <= Config.RISK_PER_TRADE * 3

    def test_short_df_falls_back_to_default(self):
        """Если данных < 30 свечей — PortfolioOptimizer не вызывается."""
        from config import Config

        runner = make_runner()
        recs = [
            {"symbol": "BTC/USDT", "action": "buy"},
            {"symbol": "ETH/USDT", "action": "buy"},
        ]
        market_data = {
            "BTC/USDT": make_ohlcv(10),  # слишком мало данных
            "ETH/USDT": make_ohlcv(10),
        }
        result = runner.optimize_allocation(recs, market_data)
        runner._optimizer.allocate.assert_not_called()
        for r in result:
            assert r["alloc_fraction"] == Config.RISK_PER_TRADE


# ---------------------------------------------------------------------------
# print_recommendations
# ---------------------------------------------------------------------------


class TestPrintRecommendations:
    """Тесты вывода рекомендаций в stdout."""

    def test_print_empty_recs_does_not_raise(self, capsys):
        """print_recommendations не падает на пустом списке."""
        CycleRunner.print_recommendations([], balance=10000.0, cycle=1)
        captured = capsys.readouterr()
        assert "No actionable" in captured.out

    def test_print_with_recs_shows_symbol(self, capsys):
        """print_recommendations выводит символ и action."""
        recs = [
            {
                "symbol": "BTC/USDT",
                "action": "buy",
                "confidence": 0.85,
                "strategy": "ema_crossover",
                "entry": 65000.0,
                "stop_loss": 63000.0,
                "take_profit": 69000.0,
                "reasoning": "Strong trend",
            }
        ]
        CycleRunner.print_recommendations(recs, balance=10000.0, cycle=5)
        captured = capsys.readouterr()
        assert "BTC/USDT" in captured.out
        assert "BUY" in captured.out

    def test_print_multiple_recs_does_not_raise(self, capsys):
        """print_recommendations выводит несколько записей без ошибок."""
        recs = [
            {
                "symbol": f"SYM{i}",
                "action": "sell",
                "confidence": 0.6,
                "strategy": "rsi",
            }
            for i in range(5)
        ]
        CycleRunner.print_recommendations(recs, balance=5000.0, cycle=10)
        captured = capsys.readouterr()
        assert "SYM0" in captured.out


# ---------------------------------------------------------------------------
# notify_new_signals
# ---------------------------------------------------------------------------


class TestNotifyNewSignals:
    """Тесты Telegram-уведомлений о новых сигналах."""

    @pytest.mark.asyncio
    async def test_notify_calls_telegram_for_new_buy_signal(self):
        """Новый buy-сигнал отправляется в Telegram."""
        runner = make_runner()
        recs = [
            {
                "symbol": "BTC/USDT",
                "action": "buy",
                "confidence": 0.9,
                "strategy": "ema",
            }
        ]
        await runner.notify_new_signals(recs, balance=10000.0, cycle=1)
        runner._telegram.notify.assert_called_once()

    @pytest.mark.asyncio
    async def test_notify_skips_hold_signals(self):
        """Hold-сигналы не отправляются в Telegram."""
        runner = make_runner()
        recs = [{"symbol": "BTC/USDT", "action": "hold", "confidence": 0.5}]
        await runner.notify_new_signals(recs, balance=10000.0, cycle=1)
        runner._telegram.notify.assert_not_called()

    @pytest.mark.asyncio
    async def test_notify_deduplicates_repeated_signals(self):
        """Повторный идентичный сигнал не отправляется повторно."""
        runner = make_runner()
        recs = [
            {
                "symbol": "BTC/USDT",
                "action": "buy",
                "confidence": 0.9,
                "strategy": "ema",
            }
        ]
        await runner.notify_new_signals(recs, balance=10000.0, cycle=1)
        await runner.notify_new_signals(recs, balance=10000.0, cycle=2)
        # Второй вызов не должен дублировать сообщение
        assert runner._telegram.notify.call_count == 1

    @pytest.mark.asyncio
    async def test_notify_sends_again_after_action_change(self):
        """После смены action (buy → sell) сигнал отправляется снова."""
        runner = make_runner()
        buy_recs = [
            {
                "symbol": "BTC/USDT",
                "action": "buy",
                "confidence": 0.9,
                "strategy": "ema",
            }
        ]
        sell_recs = [
            {
                "symbol": "BTC/USDT",
                "action": "sell",
                "confidence": 0.85,
                "strategy": "ema",
            }
        ]

        await runner.notify_new_signals(buy_recs, balance=10000.0, cycle=1)
        await runner.notify_new_signals(sell_recs, balance=10000.0, cycle=2)
        assert runner._telegram.notify.call_count == 2

    @pytest.mark.asyncio
    async def test_notify_evicts_symbol_when_absent_from_recs(self):
        """Символ, исчезнувший из рекомендаций, удаляется из dedup-кэша."""
        runner = make_runner()
        buy_recs = [
            {
                "symbol": "BTC/USDT",
                "action": "buy",
                "confidence": 0.9,
                "strategy": "ema",
            }
        ]
        await runner.notify_new_signals(buy_recs, balance=10000.0, cycle=1)
        assert "BTC/USDT" in runner._last_signals

        # Следующий цикл без BTC/USDT → должен быть evicted
        await runner.notify_new_signals([], balance=10000.0, cycle=2)
        assert "BTC/USDT" not in runner._last_signals

    @pytest.mark.asyncio
    async def test_notify_includes_entry_sl_tp_in_message(self):
        """Сообщение включает entry, SL, TP если указаны."""
        runner = make_runner()
        recs = [
            {
                "symbol": "ETH/USDT",
                "action": "buy",
                "confidence": 0.88,
                "strategy": "macd",
                "entry": 3500.0,
                "stop_loss": 3400.0,
                "take_profit": 3700.0,
            }
        ]
        await runner.notify_new_signals(recs, balance=10000.0, cycle=3)
        message = runner._telegram.notify.call_args[0][0]
        assert "3500" in message
        assert "3400" in message
        assert "3700" in message


# ---------------------------------------------------------------------------
# md_escape
# ---------------------------------------------------------------------------


class TestMdEscape:
    """Тесты экранирования Markdown-символов."""

    def test_escape_underscores(self):
        assert CycleRunner.md_escape("trend_up") == "trend\\_up"

    def test_escape_asterisks(self):
        assert CycleRunner.md_escape("**bold**") == "\\*\\*bold\\*\\*"

    def test_escape_backticks(self):
        assert CycleRunner.md_escape("`code`") == "\\`code\\`"

    def test_escape_brackets(self):
        assert CycleRunner.md_escape("[link]") == "\\[link]"

    def test_no_escape_needed(self):
        assert CycleRunner.md_escape("hello world") == "hello world"
