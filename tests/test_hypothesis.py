"""
Property-based тесты (hypothesis) для критических инвариантов системы.

Гарантируют, что математические свойства алгоритмов выполняются
для любых допустимых входных данных.

hypothesis не в requirements.txt — устанавливается через pip install hypothesis.
Если пакет не установлен — тесты пропускаются.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    from hypothesis import HealthCheck, assume, given, settings
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ohlcv_df(n: int, start_price: float = 100.0) -> pd.DataFrame:
    """Создаёт синтетический OHLCV DataFrame из n строк."""
    rng = np.random.default_rng(seed=42 + n)
    prices = start_price * np.cumprod(1 + rng.normal(0, 0.005, n))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "open": prices * 0.999,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": rng.uniform(100, 1000, n),
        }
    ).set_index("timestamp")


# ---------------------------------------------------------------------------
# 1. RSI всегда в [0, 100]
# ---------------------------------------------------------------------------

if HYPOTHESIS_AVAILABLE:

    @given(
        st.lists(
            st.floats(
                min_value=0.1,
                max_value=100_000,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=20,
            max_size=200,
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_rsi_always_in_0_100_range(prices):
        """RSI всегда лежит в диапазоне [0, 100] для любой ценовой серии."""
        from src.indicators import calculate_rsi

        series = pd.Series(prices, dtype=float)
        rsi = calculate_rsi(series)
        valid = rsi.dropna()
        assert (valid >= 0).all(), f"RSI < 0: {valid[valid < 0].tolist()}"
        assert (valid <= 100).all(), f"RSI > 100: {valid[valid > 100].tolist()}"

    @given(st.integers(min_value=50, max_value=500))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_add_indicators_columns_stable(n_rows):
        """add_indicators добавляет одни и те же колонки независимо от размера."""
        from src.indicators import add_indicators

        df = make_ohlcv_df(n_rows)
        result = add_indicators(df)
        required = [
            "rsi",
            "macd",
            "macd_signal",
            "bb_upper",
            "bb_lower",
            "sma_20",
            "atr",
        ]
        for col in required:
            assert col in result.columns, f"Missing: {col} for n={n_rows}"

    @given(
        win_rate=st.floats(min_value=0.01, max_value=0.99),
        risk_reward=st.floats(min_value=0.1, max_value=10.0),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_kelly_fraction_always_non_negative(win_rate, risk_reward):
        """Размер позиции по Келли всегда >= 0 для любых win_rate и risk_reward."""
        assume(not np.isnan(win_rate) and not np.isnan(risk_reward))

        from unittest.mock import patch

        with patch("src.redis_client.RedisClient.__init__", return_value=None):
            from src.risk_management import RiskManager

            rm = RiskManager(10000, 0.02)
            rm.redis = None

        entry = 100.0
        sl = entry * (1 - 0.02)
        tp = entry * (1 + 0.02 * risk_reward)

        size = rm.calculate_kelly_size(entry, sl, tp, win_rate, 10000)
        assert size >= 0, (
            f"negative size={size} for win_rate={win_rate},"
            f" risk_reward={risk_reward}"
        )

    @given(
        order_size=st.floats(min_value=1.0, max_value=10_000_000),
        daily_volume=st.floats(min_value=1.0, max_value=100_000_000_000),
        daily_vol=st.floats(min_value=0.001, max_value=0.5),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_almgren_chriss_impact_in_bounds(order_size, daily_volume, daily_vol):
        """almgren_chriss_impact всегда в [MIN_IMPACT, MAX_IMPACT]."""
        assume(not np.isnan(order_size))
        assume(not np.isnan(daily_volume))
        assume(not np.isnan(daily_vol))

        from src.market_impact import (
            _MAX_IMPACT,
            _MIN_IMPACT,
            almgren_chriss_impact,
        )

        impact = almgren_chriss_impact(order_size, daily_volume, daily_vol)
        assert _MIN_IMPACT <= impact <= _MAX_IMPACT, (
            f"impact={impact} out of bounds for "
            f"order={order_size}, vol={daily_volume}, dv={daily_vol}"
        )

    @given(
        signal_action=st.sampled_from(["buy", "sell", "hold"]),
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_validate_signal_returns_bool(signal_action, confidence):
        """validate_signal всегда возвращает bool."""
        import asyncio
        from unittest.mock import patch

        with patch("src.redis_client.RedisClient.__init__", return_value=None):
            from src.risk_management import RiskManager

            rm = RiskManager(10000, 0.02)
            rm.redis = None

        result = asyncio.get_event_loop().run_until_complete(
            rm.validate_signal(
                {"action": signal_action, "confidence": confidence}, None
            )
        )
        assert isinstance(result, bool)
        if signal_action == "hold":
            assert result is False

    @given(n_recs=st.integers(min_value=0, max_value=20))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
    def test_print_recommendations_never_raises(n_recs, capsys):
        """print_recommendations никогда не бросает исключение."""
        from src.cycle import CycleRunner

        recs = [
            {
                "symbol": f"SYM{i}",
                "action": "buy",
                "confidence": 0.8,
                "strategy": "ema",
            }
            for i in range(n_recs)
        ]
        try:
            CycleRunner.print_recommendations(recs, balance=10000.0, cycle=1)
        except Exception as e:
            pytest.fail(f"print_recommendations raised: {e}")

    @given(
        text=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
            max_size=200,
        )
    )
    @settings(max_examples=100)
    def test_md_escape_never_raises(text):
        """md_escape никогда не бросает исключение."""
        from src.cycle import CycleRunner

        result = CycleRunner.md_escape(text)
        assert isinstance(result, str)

    @given(
        close_prices=st.lists(
            st.floats(
                min_value=1.0,
                max_value=10_000,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=5,
            max_size=200,
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_estimate_from_df_never_raises(close_prices):
        """estimate_from_df никогда не бросает исключение."""
        from src.market_impact import _MAX_IMPACT, _MIN_IMPACT, estimate_from_df

        df = pd.DataFrame(
            {
                "close": close_prices,
                "volume": [1000.0] * len(close_prices),
            }
        )
        result = estimate_from_df(df, order_size_usdt=10_000)
        assert isinstance(result, float)
        assert _MIN_IMPACT <= result <= _MAX_IMPACT
