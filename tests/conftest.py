import sys
from unittest.mock import MagicMock

# Мокаем redis до любых импортов из src, чтобы тесты запускались без живого Redis.
_redis_mock = MagicMock()
_redis_mock.StrictRedis = MagicMock
_redis_mock.Redis = MagicMock
_redis_mock.ConnectionError = ConnectionError
sys.modules.setdefault("redis", _redis_mock)

# Мокаем prometheus_client, чтобы тесты работали без установленного пакета.
_prom_mock = MagicMock()
_prom_mock.Counter = MagicMock(return_value=MagicMock())
_prom_mock.Gauge = MagicMock(return_value=MagicMock())
sys.modules.setdefault("prometheus_client", _prom_mock)

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(
    n: int = 100,
    start_price: float = 100.0,
    trend: float = 0.001,
) -> pd.DataFrame:
    """Синтетический OHLCV DataFrame для тестирования."""
    prices = [start_price]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + trend + np.random.normal(0, 0.005)))
    prices = np.array(prices)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "open": prices * 0.999,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": np.random.uniform(100, 1000, n),
        }
    )
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture
def ohlcv_uptrend():
    return make_ohlcv(n=100, trend=0.002)


@pytest.fixture
def ohlcv_flat():
    return make_ohlcv(n=100, trend=0.0)


@pytest.fixture
def ohlcv_downtrend():
    return make_ohlcv(n=100, trend=-0.002)
