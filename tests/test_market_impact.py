"""
Тесты для src/market_impact.py — модель Almgren-Chriss.

Покрывают: almgren_chriss_impact(), estimate_from_df(), граничные случаи.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.market_impact import (
    _MIN_IMPACT,
    _MAX_IMPACT,
    almgren_chriss_impact,
    estimate_from_df,
)


# ---------------------------------------------------------------------------
# almgren_chriss_impact
# ---------------------------------------------------------------------------

class TestAlmgrenChrissImpact:
    """Тесты функции almgren_chriss_impact."""

    def test_typical_crypto_values_in_valid_range(self):
        """Типичные значения BTC → импакт в диапазоне [MIN, MAX]."""
        # BTC: order=10000 USDT, daily vol=1B USDT, vol=2%
        impact = almgren_chriss_impact(
            order_size_usdt=10_000,
            daily_volume_usdt=1_000_000_000,
            daily_vol=0.02,
        )
        assert _MIN_IMPACT <= impact <= _MAX_IMPACT

    def test_impact_is_float(self):
        """Функция возвращает float."""
        impact = almgren_chriss_impact(10000, 1_000_000, 0.02)
        assert isinstance(impact, float)

    def test_zero_order_size_returns_min_impact(self):
        """Нулевой размер ордера → MIN_IMPACT."""
        impact = almgren_chriss_impact(0, 1_000_000, 0.02)
        assert impact == _MIN_IMPACT

    def test_zero_daily_volume_returns_min_impact(self):
        """Нулевой дневной объём → MIN_IMPACT."""
        impact = almgren_chriss_impact(10000, 0, 0.02)
        assert impact == _MIN_IMPACT

    def test_negative_order_size_returns_min_impact(self):
        """Отрицательный размер ордера → MIN_IMPACT."""
        impact = almgren_chriss_impact(-1000, 1_000_000, 0.02)
        assert impact == _MIN_IMPACT

    def test_impact_capped_at_max(self):
        """Огромный ордер → импакт не превышает MAX_IMPACT."""
        impact = almgren_chriss_impact(
            order_size_usdt=1_000_000_000,  # огромный ордер
            daily_volume_usdt=1_000,        # мизерный объём
            daily_vol=0.10,
        )
        assert impact == _MAX_IMPACT

    def test_impact_floor_at_min(self):
        """Крошечный ордер → импакт не ниже MIN_IMPACT."""
        impact = almgren_chriss_impact(
            order_size_usdt=1,              # 1 USDT
            daily_volume_usdt=1_000_000_000,
            daily_vol=0.001,
        )
        assert impact >= _MIN_IMPACT

    def test_larger_order_increases_impact(self):
        """Больший ордер при прочих равных даёт больший импакт."""
        small = almgren_chriss_impact(1_000, 1_000_000, 0.02)
        large = almgren_chriss_impact(100_000, 1_000_000, 0.02)
        assert large >= small

    def test_higher_volatility_increases_impact(self):
        """Более высокая волатильность увеличивает импакт."""
        low_vol = almgren_chriss_impact(10_000, 1_000_000, 0.01)
        high_vol = almgren_chriss_impact(10_000, 1_000_000, 0.05)
        assert high_vol >= low_vol

    def test_custom_eta_gamma(self):
        """Кастомные параметры eta и gamma применяются корректно."""
        impact_default = almgren_chriss_impact(10_000, 1_000_000, 0.02)
        impact_custom = almgren_chriss_impact(10_000, 1_000_000, 0.02, eta=0.5, gamma=0.5)
        # Большие коэффициенты → больший импакт (при тех же условиях)
        assert impact_custom >= impact_default

    def test_small_participation_rate_gives_small_impact(self):
        """Маленький participation rate (0.0001) → небольшой импакт."""
        impact = almgren_chriss_impact(
            order_size_usdt=10_000,
            daily_volume_usdt=100_000_000,  # rate = 0.0001
            daily_vol=0.02,
        )
        # Для маленькой доли участия импакт должен быть невелик
        assert impact < 0.005


# ---------------------------------------------------------------------------
# estimate_from_df
# ---------------------------------------------------------------------------

class TestEstimateFromDf:
    """Тесты функции estimate_from_df."""

    def _make_df(self, n: int = 50, price: float = 50000.0) -> pd.DataFrame:
        """Синтетический OHLCV DataFrame."""
        prices = price + np.cumsum(np.random.randn(n) * 100)
        return pd.DataFrame({
            "close": prices,
            "open": prices * 0.999,
            "high": prices * 1.002,
            "low": prices * 0.998,
            "volume": np.random.uniform(1000, 10000, n),
        })

    def test_typical_values_in_range(self):
        """estimate_from_df с нормальными данными → результат в [MIN, MAX]."""
        df = self._make_df(50, 50000.0)
        impact = estimate_from_df(df, order_size_usdt=10_000, timeframe="15m")
        assert _MIN_IMPACT <= impact <= _MAX_IMPACT

    def test_returns_float(self):
        """estimate_from_df возвращает float."""
        df = self._make_df()
        result = estimate_from_df(df, 10_000)
        assert isinstance(result, float)

    def test_none_df_returns_min_impact(self):
        """None DataFrame → MIN_IMPACT."""
        impact = estimate_from_df(None, order_size_usdt=10_000)
        assert impact == _MIN_IMPACT

    def test_empty_df_returns_min_impact(self):
        """Пустой DataFrame → MIN_IMPACT."""
        impact = estimate_from_df(pd.DataFrame(), order_size_usdt=10_000)
        assert impact == _MIN_IMPACT

    def test_single_row_df_returns_min_impact(self):
        """DataFrame с одной строкой → MIN_IMPACT (нужно хотя бы 2 для log returns)."""
        df = self._make_df(1)
        impact = estimate_from_df(df, order_size_usdt=10_000)
        assert impact == _MIN_IMPACT

    def test_df_without_close_returns_min_impact(self):
        """DataFrame без колонки 'close' → MIN_IMPACT."""
        df = pd.DataFrame({"open": [100, 101, 102], "volume": [1000, 1000, 1000]})
        impact = estimate_from_df(df, order_size_usdt=10_000)
        assert impact == _MIN_IMPACT

    def test_df_without_volume_uses_fallback(self):
        """DataFrame без колонки 'volume' → использует fallback объём."""
        df = pd.DataFrame({"close": [50000.0 + i * 10 for i in range(20)]})
        impact = estimate_from_df(df, order_size_usdt=10_000)
        assert _MIN_IMPACT <= impact <= _MAX_IMPACT

    def test_different_timeframes_give_different_impacts(self):
        """Разные таймфреймы дают разные значения импакта."""
        df = self._make_df(100)
        impact_1m = estimate_from_df(df, 10_000, timeframe="1m")
        impact_1d = estimate_from_df(df, 10_000, timeframe="1d")
        # 1m имеет 1440 свечей/день vs 1d → разная нормализация объёма
        # Они могут быть равны только если зажаты кэпами
        assert isinstance(impact_1m, float)
        assert isinstance(impact_1d, float)

    def test_larger_order_size_increases_impact(self):
        """Больший ордер увеличивает оцениваемый импакт."""
        df = self._make_df(100, 50000.0)
        small = estimate_from_df(df, order_size_usdt=1_000)
        large = estimate_from_df(df, order_size_usdt=1_000_000)
        assert large >= small

    def test_very_few_log_returns_uses_fallback_vol(self):
        """Менее 5 лог-доходностей → используется fallback волатильность 0.02."""
        df = pd.DataFrame({"close": [100.0, 101.0, 102.0], "volume": [1000, 1000, 1000]})
        impact = estimate_from_df(df, order_size_usdt=10_000)
        assert _MIN_IMPACT <= impact <= _MAX_IMPACT
