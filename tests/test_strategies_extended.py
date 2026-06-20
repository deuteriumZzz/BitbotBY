"""
Расширенные тесты src/strategies.py.

Покрывают ветки, не охваченные test_strategies.py:
- TradingStrategy: switch_strategy, get_signal, list_strategies,
  get_current_strategy_info
- get_all_strategies, create_strategy
- Конкретные сигналы для каждой стратегии в специфических рыночных условиях
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.indicators import calculate_technical_indicators
from src.strategies import (
    STRATEGY_REGISTRY,
    BollingerBandsStrategy,
    BreakoutStrategy,
    EMACrossoverStrategy,
    MeanReversionStrategy,
    RSIMomentumStrategy,
    ScalpingStrategy,
    SwingTradingStrategy,
    TradingStrategy,
    TrendFollowingStrategy,
    create_strategy,
    get_all_strategies,
)
from tests.conftest import make_ohlcv

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_redis():
    """Патч RedisClient — не требуется живой Redis."""
    with patch("src.strategies.RedisClient") as mock_redis_cls:
        instance = MagicMock()
        instance.save_trading_state = MagicMock(return_value=None)
        instance.publish_signal = MagicMock(return_value=None)
        mock_redis_cls.return_value = instance
        yield instance


def df_with_indicators(n: int = 100, trend: float = 0.002) -> pd.DataFrame:
    """OHLCV + все технические индикаторы."""
    return calculate_technical_indicators(make_ohlcv(n=n, trend=trend))


# ---------------------------------------------------------------------------
# create_strategy / get_all_strategies
# ---------------------------------------------------------------------------


class TestFactoryFunctions:
    """Тесты фабричных функций."""

    def test_create_strategy_known_name(self):
        """create_strategy возвращает экземпляр правильного класса."""
        strat = create_strategy("rsi_momentum")
        assert isinstance(strat, RSIMomentumStrategy)

    def test_create_strategy_unknown_raises_value_error(self):
        """create_strategy бросает ValueError для неизвестного имени."""
        with pytest.raises(ValueError, match="Неизвестная стратегия"):
            create_strategy("nonexistent_strategy")

    def test_get_all_strategies_returns_all(self):
        """get_all_strategies возвращает описания всех зарегистрированных стратегий."""
        strategies = get_all_strategies()
        assert len(strategies) == len(STRATEGY_REGISTRY)

    def test_get_all_strategies_have_required_keys(self):
        """Каждое описание стратегии содержит обязательные ключи."""
        for info in get_all_strategies():
            assert "name" in info
            assert "description" in info
            assert "risk_level" in info
            assert "market_type" in info
            assert "recommended_timeframes" in info


# ---------------------------------------------------------------------------
# BaseStrategy helpers
# ---------------------------------------------------------------------------


class TestBaseStrategyHelpers:
    """Тесты вспомогательных методов BaseStrategy."""

    def test_last_returns_default_for_missing_column(self):
        """_last возвращает default если колонки нет."""
        strat = EMACrossoverStrategy()
        df = pd.DataFrame({"close": [1.0, 2.0]})
        assert strat._last(df, "nonexistent", default=99.0) == 99.0

    def test_last_returns_last_value(self):
        """_last возвращает последнее значение колонки."""
        strat = EMACrossoverStrategy()
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        assert strat._last(df, "close") == 3.0

    def test_prev_returns_second_last(self):
        """_prev возвращает предпоследнее значение."""
        strat = EMACrossoverStrategy()
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        assert strat._prev(df, "close") == 2.0

    def test_prev2_returns_third_last(self):
        """_prev2 возвращает третье с конца значение."""
        strat = EMACrossoverStrategy()
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]})
        assert strat._prev2(df, "close") == 2.0

    def test_prev_returns_default_for_short_df(self):
        """_prev возвращает default для DataFrame из одной строки."""
        strat = EMACrossoverStrategy()
        df = pd.DataFrame({"close": [42.0]})
        assert strat._prev(df, "close", default=0.0) == 0.0

    def test_get_info_returns_correct_dict(self):
        """get_info возвращает корректный словарь метаданных."""
        strat = RSIMomentumStrategy()
        info = strat.get_info()
        assert info["name"] == "rsi_momentum"
        assert info["market_type"] == "ranging"
        assert info["risk_level"] == "medium"


# ---------------------------------------------------------------------------
# RSIMomentumStrategy — специфические сигналы
# ---------------------------------------------------------------------------


class TestRSIMomentumSignals:
    """Тесты сигналов RSIMomentumStrategy в разных рыночных условиях."""

    def _make_df_with_rsi(self, rsi_value: float) -> pd.DataFrame:
        """DataFrame с заданным RSI через прямую инъекцию."""
        df = df_with_indicators(100)
        df["rsi"] = rsi_value
        return df

    def test_strong_buy_signal_at_rsi_below_25(self):
        """RSI < 25 → buy с confidence 0.90."""
        strat = RSIMomentumStrategy()
        df = self._make_df_with_rsi(22.0)
        signal = strat.generate_signal(df)
        assert signal["action"] == "buy"
        assert signal["confidence"] == pytest.approx(0.90)

    def test_moderate_buy_signal_at_rsi_below_30(self):
        """RSI < 30 (но >= 25) → buy с confidence 0.78."""
        strat = RSIMomentumStrategy()
        df = self._make_df_with_rsi(27.0)
        signal = strat.generate_signal(df)
        assert signal["action"] == "buy"
        assert signal["confidence"] == pytest.approx(0.78)

    def test_hold_signal_at_rsi_30_to_40(self):
        """RSI 30-40 — нейтральная зона → hold
        (тир убран, нет торгового преимущества)."""
        strat = RSIMomentumStrategy()
        df = self._make_df_with_rsi(35.0)
        signal = strat.generate_signal(df)
        assert signal["action"] == "hold"

    def test_strong_sell_signal_at_rsi_above_75(self):
        """RSI > 75 → sell с confidence 0.90."""
        strat = RSIMomentumStrategy()
        df = self._make_df_with_rsi(78.0)
        signal = strat.generate_signal(df)
        assert signal["action"] == "sell"
        assert signal["confidence"] == pytest.approx(0.90)

    def test_moderate_sell_signal_at_rsi_above_70(self):
        """RSI > 70 (но <= 75) → sell с confidence 0.78."""
        strat = RSIMomentumStrategy()
        df = self._make_df_with_rsi(72.0)
        signal = strat.generate_signal(df)
        assert signal["action"] == "sell"
        assert signal["confidence"] == pytest.approx(0.78)

    def test_hold_signal_at_rsi_60_to_70(self):
        """RSI 60-70 — нейтральная зона → hold
        (тир убран, нет торгового преимущества)."""
        strat = RSIMomentumStrategy()
        df = self._make_df_with_rsi(65.0)
        signal = strat.generate_signal(df)
        assert signal["action"] == "hold"

    def test_hold_when_rsi_neutral(self):
        """RSI в нейтральном диапазоне (40-60) → hold."""
        strat = RSIMomentumStrategy()
        df = self._make_df_with_rsi(50.0)
        signal = strat.generate_signal(df)
        assert signal["action"] == "hold"


# ---------------------------------------------------------------------------
# BollingerBandsStrategy — специфические сигналы
# ---------------------------------------------------------------------------


class TestBollingerBandsSignals:
    """Тесты сигналов BollingerBandsStrategy."""

    def test_buy_at_lower_band_with_low_rsi(self):
        """Цена на нижней BB + RSI < 30 → buy 0.90."""
        strat = BollingerBandsStrategy()
        df = df_with_indicators(100)
        price = float(df["close"].iloc[-1])
        df["bb_lower"] = price + 1.0  # close <= bb_lower
        df["bb_upper"] = price * 1.05
        df["rsi"] = 28.0
        signal = strat.generate_signal(df)
        assert signal["action"] == "buy"
        assert signal["confidence"] == pytest.approx(0.90)

    def test_sell_at_upper_band_with_high_rsi(self):
        """Цена на верхней BB + RSI > 70 → sell 0.90."""
        strat = BollingerBandsStrategy()
        df = df_with_indicators(100)
        price = float(df["close"].iloc[-1])
        df["bb_upper"] = price - 1.0  # close >= bb_upper
        df["bb_lower"] = price * 0.95
        df["rsi"] = 72.0
        signal = strat.generate_signal(df)
        assert signal["action"] == "sell"
        assert signal["confidence"] == pytest.approx(0.90)

    def test_hold_when_bands_equal(self):
        """Нулевая ширина BB → hold."""
        strat = BollingerBandsStrategy()
        df = df_with_indicators(100)
        df["bb_upper"] = df["close"]
        df["bb_lower"] = df["close"]
        signal = strat.generate_signal(df)
        assert signal["action"] == "hold"


# ---------------------------------------------------------------------------
# BreakoutStrategy
# ---------------------------------------------------------------------------


class TestBreakoutStrategySignals:
    """Тесты сигналов BreakoutStrategy."""

    def test_hold_on_short_data(self):
        """Менее 21 свечи → hold."""
        strat = BreakoutStrategy()
        df = df_with_indicators(15)
        signal = strat.generate_signal(df)
        assert signal["action"] == "hold"

    def test_buy_on_resistance_breakout_with_high_volume(self):
        """Пробой сопротивления + высокий объём → buy 0.85."""
        strat = BreakoutStrategy()
        df = df_with_indicators(100, trend=0.0)
        # Устанавливаем текущую цену выше максимума предыдущих 20 свечей
        prev_max = float(df["high"].iloc[-21:-1].max())
        df.iloc[-1, df.columns.get_loc("close")] = prev_max * 1.05
        df["volume_ratio"] = 2.0  # высокий объём
        signal = strat.generate_signal(df)
        assert signal["action"] == "buy"
        assert signal["confidence"] == pytest.approx(0.85)

    def test_buy_on_resistance_breakout_with_low_volume(self):
        """Пробой сопротивления + низкий объём → buy 0.65."""
        strat = BreakoutStrategy()
        df = df_with_indicators(100, trend=0.0)
        prev_max = float(df["high"].iloc[-21:-1].max())
        df.iloc[-1, df.columns.get_loc("close")] = prev_max * 1.05
        df["volume_ratio"] = 1.0  # низкий объём
        signal = strat.generate_signal(df)
        assert signal["action"] == "buy"
        assert signal["confidence"] == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# MeanReversionStrategy
# ---------------------------------------------------------------------------


class TestMeanReversionSignals:
    """Тесты MeanReversionStrategy."""

    def test_strong_buy_extreme_low_with_positive_momentum(self):
        """Экстремально низкая цена + положительный momentum (разворот) → buy 0.88.

        mom > 0 при extreme_low = цена уже отскакивает вверх = сильный сигнал разворота.
        """
        strat = MeanReversionStrategy()
        df = df_with_indicators(100)
        price = float(df["close"].iloc[-1])
        df["bb_lower"] = price + 1.0
        df["bb_upper"] = price * 1.1
        df["rsi"] = 25.0
        df["momentum"] = 0.05  # отскок вверх
        signal = strat.generate_signal(df)
        assert signal["action"] == "buy"
        assert signal["confidence"] == pytest.approx(0.88)

    def test_moderate_buy_extreme_low_with_negative_momentum(self):
        """Экстремально низкая цена + отрицательный momentum
        (всё ещё падает) → buy 0.72."""
        strat = MeanReversionStrategy()
        df = df_with_indicators(100)
        price = float(df["close"].iloc[-1])
        df["bb_lower"] = price + 1.0
        df["bb_upper"] = price * 1.1
        df["rsi"] = 25.0
        df["momentum"] = -0.05  # ещё падает
        signal = strat.generate_signal(df)
        assert signal["action"] == "buy"
        assert signal["confidence"] == pytest.approx(0.72)

    def test_moderate_buy_below_lower_band(self):
        """Цена ниже нижней BB, но RSI >= 30 → buy 0.62."""
        strat = MeanReversionStrategy()
        df = df_with_indicators(100)
        price = float(df["close"].iloc[-1])
        df["bb_lower"] = price + 1.0
        df["bb_upper"] = price * 1.1
        df["rsi"] = 45.0
        signal = strat.generate_signal(df)
        assert signal["action"] == "buy"
        assert signal["confidence"] == pytest.approx(0.62)

    def test_buy_below_middle_band_with_low_rsi(self):
        """Цена ниже средней BB + RSI < 45 → buy 0.55."""
        strat = MeanReversionStrategy()
        df = df_with_indicators(100)
        price = float(df["close"].iloc[-1])
        df["bb_lower"] = price * 0.95
        df["bb_upper"] = price * 1.1
        df["bb_middle"] = price + 1.0  # close < bb_middle
        df["rsi"] = 40.0
        signal = strat.generate_signal(df)
        assert signal["action"] == "buy"
        assert signal["confidence"] == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# ScalpingStrategy
# ---------------------------------------------------------------------------


class TestScalpingSignals:
    """Тесты ScalpingStrategy."""

    def test_buy_with_high_volume_ratio(self):
        """RSI < 32, цена > EMA, MACD > Signal, volume_ratio > 1.5 → buy 0.75."""
        strat = ScalpingStrategy()
        df = df_with_indicators(100)
        price = float(df["close"].iloc[-1])
        df["rsi"] = 30.0
        df["ema_short"] = price * 0.99  # close > ema_short
        df["macd"] = 1.0
        df["macd_signal"] = 0.5
        df["volume_ratio"] = 2.0
        signal = strat.generate_signal(df)
        assert signal["action"] == "buy"
        assert signal["confidence"] == pytest.approx(0.75)

    def test_buy_with_low_volume_ratio(self):
        """Те же условия, но volume_ratio <= 1.5 → buy 0.58."""
        strat = ScalpingStrategy()
        df = df_with_indicators(100)
        price = float(df["close"].iloc[-1])
        df["rsi"] = 30.0
        df["ema_short"] = price * 0.99
        df["macd"] = 1.0
        df["macd_signal"] = 0.5
        df["volume_ratio"] = 1.0
        signal = strat.generate_signal(df)
        assert signal["action"] == "buy"
        assert signal["confidence"] == pytest.approx(0.58)

    def test_sell_signal(self):
        """RSI > 68, цена < EMA, MACD < Signal → sell."""
        strat = ScalpingStrategy()
        df = df_with_indicators(100)
        price = float(df["close"].iloc[-1])
        df["rsi"] = 70.0
        df["ema_short"] = price * 1.01  # close < ema_short
        df["macd"] = -1.0
        df["macd_signal"] = -0.5
        df["volume_ratio"] = 1.0
        signal = strat.generate_signal(df)
        assert signal["action"] == "sell"


# ---------------------------------------------------------------------------
# TrendFollowingStrategy
# ---------------------------------------------------------------------------


class TestTrendFollowingSignals:
    """Тесты TrendFollowingStrategy."""

    def test_strong_buy_with_rsi_in_range(self):
        """SMA20 > SMA50, EMA short > long, close > SMA20, RSI в [45, 70] → buy 0.80."""
        strat = TrendFollowingStrategy()
        df = df_with_indicators(100)
        price = float(df["close"].iloc[-1])
        df["sma_20"] = price * 0.99  # close > sma_20
        df["sma_50"] = price * 0.98  # sma_20 > sma_50
        df["ema_short"] = price * 0.99
        df["ema_long"] = price * 0.97
        df["rsi"] = 55.0
        signal = strat.generate_signal(df)
        assert signal["action"] == "buy"
        assert signal["confidence"] == pytest.approx(0.80)

    def test_weak_buy_without_rsi_filter(self):
        """strong_up но RSI вне [45, 70] → buy 0.62."""
        strat = TrendFollowingStrategy()
        df = df_with_indicators(100)
        price = float(df["close"].iloc[-1])
        df["sma_20"] = price * 0.99
        df["sma_50"] = price * 0.98
        df["ema_short"] = price * 0.99
        df["ema_long"] = price * 0.97
        df["rsi"] = 80.0  # вне диапазона
        signal = strat.generate_signal(df)
        assert signal["action"] == "buy"
        assert signal["confidence"] == pytest.approx(0.62)

    def test_sell_signals(self):
        """SMA20 < SMA50, EMA short < long, close < SMA20 → sell."""
        strat = TrendFollowingStrategy()
        df = df_with_indicators(100)
        price = float(df["close"].iloc[-1])
        # strong_down: sma_20 < sma_50 AND ema_short < ema_long AND close < sma_20
        df["sma_20"] = price * 1.02  # sma_20 > close
        df["sma_50"] = price * 1.05  # sma_50 > sma_20
        df["ema_short"] = price * 0.97
        df["ema_long"] = price * 0.99
        df["rsi"] = 40.0  # в (30, 55) → confidence 0.80
        signal = strat.generate_signal(df)
        assert signal["action"] == "sell"


# ---------------------------------------------------------------------------
# TradingStrategy (оркестратор)
# ---------------------------------------------------------------------------


class TestTradingStrategyOrchestrator:
    """Тесты класса TradingStrategy (оркестратор)."""

    def test_switch_strategy_changes_active_strategy(self):
        """switch_strategy меняет активную стратегию."""
        ts = TradingStrategy("ema_crossover")
        assert ts.strategy_name == "ema_crossover"
        ts.switch_strategy("rsi_momentum")
        assert ts.strategy_name == "rsi_momentum"
        assert isinstance(ts.strategy, RSIMomentumStrategy)

    def test_switch_strategy_invalid_raises(self):
        """switch_strategy с неизвестным именем бросает ValueError."""
        ts = TradingStrategy("ema_crossover")
        with pytest.raises(ValueError):
            ts.switch_strategy("unknown_strategy")

    @pytest.mark.asyncio
    async def test_get_signal_returns_valid_dict(self):
        """get_signal возвращает словарь с action и confidence."""
        ts = TradingStrategy("rsi_momentum")
        df = df_with_indicators(100)
        signal = await ts.get_signal(df)
        assert "action" in signal
        assert signal["action"] in ("buy", "sell", "hold")
        assert "confidence" in signal

    @pytest.mark.asyncio
    async def test_get_signal_calls_redis_save(self, mock_redis):
        """get_signal сохраняет состояние в Redis."""
        ts = TradingStrategy("ema_crossover")
        df = df_with_indicators(100)
        await ts.get_signal(df)
        mock_redis.save_trading_state.assert_called()

    @pytest.mark.asyncio
    async def test_get_signal_publishes_signal(self, mock_redis):
        """get_signal публикует сигнал через Redis publish."""
        ts = TradingStrategy("rsi_momentum")
        df = df_with_indicators(100)
        await ts.get_signal(df)
        mock_redis.publish_signal.assert_called()

    def test_list_strategies_returns_all(self):
        """list_strategies возвращает все зарегистрированные стратегии."""
        result = TradingStrategy.list_strategies()
        assert len(result) == len(STRATEGY_REGISTRY)

    def test_get_current_strategy_info(self):
        """get_current_strategy_info возвращает метаданные активной стратегии."""
        ts = TradingStrategy("macd_crossover")
        info = ts.get_current_strategy_info()
        assert info["name"] == "macd_crossover"

    @pytest.mark.asyncio
    async def test_initialize_does_not_raise(self):
        """initialize() не бросает исключений."""
        ts = TradingStrategy("ema_crossover")
        await ts.initialize()  # should complete without error


# ---------------------------------------------------------------------------
# SwingTradingStrategy
# ---------------------------------------------------------------------------


class TestSwingTradingSignals:
    """Тесты SwingTradingStrategy."""

    def test_strong_buy_signal(self):
        """Восходящий тренд + MACD вверх + RSI в [40, 65]
        + цена выше BB middle → buy 0.85."""
        strat = SwingTradingStrategy()
        df = df_with_indicators(100)
        price = float(df["close"].iloc[-1])
        df["ema_short"] = price * 1.01  # ema_short > ema_long
        df["ema_long"] = price * 0.99
        df["macd"] = 0.5
        df["macd_signal"] = 0.3
        df["rsi"] = 52.0
        df["bb_middle"] = price * 0.99  # close > bb_middle
        signal = strat.generate_signal(df)
        assert signal["action"] == "buy"
        assert signal["confidence"] == pytest.approx(0.85)

    def test_hold_when_no_trend(self):
        """Нет явного тренда → hold."""
        strat = SwingTradingStrategy()
        df = df_with_indicators(100)
        price = float(df["close"].iloc[-1])
        # EMA short ≈ long, MACD ≈ signal
        df["ema_short"] = price
        df["ema_long"] = price
        df["macd"] = 0.0
        df["macd_signal"] = 0.0
        df["rsi"] = 50.0
        signal = strat.generate_signal(df)
        assert signal["action"] == "hold"
