import numpy as np
import pandas as pd

from src.correlation_filter import CorrelationFilter


def _df(prices):
    return pd.DataFrame(
        {"close": prices},
        index=pd.date_range("2024-01-01", periods=len(prices), freq="15min"),
    )


def _make_correlated(n: int = 60, strength: float = 0.95):
    """Два DataFrame с заданной корреляцией log-returns."""
    np.random.seed(0)
    common = np.random.normal(0, 0.01, n)
    r_a = common + np.random.normal(0, 0.001, n)
    r_b = common * strength + np.random.normal(0, 0.005 * (1 - strength), n)
    return _df(40000 * np.exp(np.cumsum(r_a))), _df(2500 * np.exp(np.cumsum(r_b)))


def _make_independent(n: int = 60):
    """Два DataFrame с ~нулевой корреляцией log-returns."""
    np.random.seed(1)
    a = 40000 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
    b = 100 * np.exp(np.cumsum(np.random.normal(0, 0.015, n)))
    return _df(a), _df(b)


class TestCorrelationFilter:
    def test_blocks_highly_correlated_pair(self):
        btc, eth = _make_correlated(strength=0.95)
        cf = CorrelationFilter(window=50, max_corr=0.7)
        cf.update_from_df("BTC/USDT", btc)
        cf.update_from_df("ETH/USDT", eth)
        assert not cf.is_allowed("ETH/USDT", ["BTC/USDT"])

    def test_allows_uncorrelated_pair(self):
        btc, sol = _make_independent()
        cf = CorrelationFilter(window=50, max_corr=0.7)
        cf.update_from_df("BTC/USDT", btc)
        cf.update_from_df("SOL/USDT", sol)
        assert cf.is_allowed("SOL/USDT", ["BTC/USDT"])

    def test_empty_open_positions_always_allowed(self):
        cf = CorrelationFilter()
        assert cf.is_allowed("BTC/USDT", []) is True

    def test_max_correlation_zero_when_no_open(self):
        cf = CorrelationFilter()
        assert cf.max_correlation("BTC/USDT", []) == 0.0

    def test_same_symbol_excluded_from_check(self):
        """BTC vs [BTC] — себя с собой не сравниваем → 0.0."""
        btc, _ = _make_correlated()
        cf = CorrelationFilter()
        cf.update_from_df("BTC/USDT", btc)
        assert cf.max_correlation("BTC/USDT", ["BTC/USDT"]) == 0.0

    def test_returns_none_when_insufficient_data(self):
        """Менее 10 свечей — не хватает данных → None."""
        cf = CorrelationFilter(window=50)
        cf.update_from_df("A/USDT", _df([1.0, 1.1, 1.2, 1.15, 1.3]))
        cf.update_from_df("B/USDT", _df([2.0, 2.1, 2.2, 2.15, 2.3]))
        assert cf.correlation("A/USDT", "B/USDT") is None

    def test_correlation_is_symmetric(self):
        btc, eth = _make_correlated()
        cf = CorrelationFilter(window=50, max_corr=0.7)
        cf.update_from_df("BTC/USDT", btc)
        cf.update_from_df("ETH/USDT", eth)
        c1 = cf.correlation("BTC/USDT", "ETH/USDT")
        c2 = cf.correlation("ETH/USDT", "BTC/USDT")
        assert c1 is not None and c2 is not None
        assert abs(c1 - c2) < 1e-10

    def test_empty_df_is_ignored(self):
        cf = CorrelationFilter()
        cf.update_from_df("X/USDT", pd.DataFrame())
        cf.update_from_df("Y/USDT", pd.DataFrame({"close": []}))
        assert cf.correlation("X/USDT", "Y/USDT") is None

    def test_threshold_above_actual_corr_allows_pair(self):
        """Порог выше реальной корреляции → пара проходит."""
        btc, eth = _make_correlated(strength=0.5)  # corr ≈ 0.5–0.7
        cf = CorrelationFilter(window=50, max_corr=0.95)
        cf.update_from_df("BTC/USDT", btc)
        cf.update_from_df("ETH/USDT", eth)
        assert cf.is_allowed("ETH/USDT", ["BTC/USDT"])

    def test_max_correlation_uses_absolute_value(self):
        """Отрицательная корреляция (шорт vs лонг) тоже блокирует."""
        np.random.seed(2)
        n = 60
        r = np.random.normal(0, 0.01, n)
        a = 100 * np.exp(np.cumsum(r))
        b = 100 * np.exp(np.cumsum(-r))  # инвертированный — corr ≈ -1
        cf = CorrelationFilter(window=50, max_corr=0.7)
        cf.update_from_df("A/USDT", _df(a))
        cf.update_from_df("B/USDT", _df(b))
        assert cf.max_correlation("B/USDT", ["A/USDT"]) >= 0.7
        assert not cf.is_allowed("B/USDT", ["A/USDT"])

    def test_no_data_for_new_symbol_allows(self):
        """Нет данных по новой монете → данных нет → разрешаем."""
        btc, _ = _make_correlated()
        cf = CorrelationFilter(window=50, max_corr=0.7)
        cf.update_from_df("BTC/USDT", btc)
        # SOL не обновлялся → corr = None → max_correlation = 0.0 → allowed
        assert cf.is_allowed("SOL/USDT", ["BTC/USDT"])
