import contextlib
import importlib
import os
import sys
from unittest.mock import MagicMock, patch

# TestClient шлёт Host: testserver; без ключа дашборд пускает только локальные
# адреса (защита от DNS-rebinding) — тестовый хост считаем своим.
os.environ.setdefault("DASHBOARD_ALLOWED_HOSTS", "testserver")

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

# Монитор позиций разложен по модулям; Config читается в каждом из них.
_MONITOR_MODULES = (
    "bitbot.execution.monitor_base",
    "bitbot.execution.booking",
    "bitbot.execution.closing",
    "bitbot.execution.levels",
)


# RuntimeConfig разложен по модулям пакета; Config читается в каждом из них.
_RUNTIME_MODULES = tuple(
    f"bitbot.config.runtime.{m}"
    for m in ("base", "trading", "risk", "models", "market", "presets", "active")
)


def _patch_config(modules, cfg):
    stack = contextlib.ExitStack()
    for name in modules:
        if hasattr(importlib.import_module(name), "Config"):
            stack.enter_context(patch(f"{name}.Config", cfg))
    return stack


def monitor_config(cfg):
    """Подменяет Config во всех модулях монитора позиций (контекстный менеджер)."""
    return _patch_config(_MONITOR_MODULES, cfg)


def runtime_config(cfg):
    """Подменяет Config во всех модулях RuntimeConfig (контекстный менеджер)."""
    return _patch_config(_RUNTIME_MODULES, cfg)


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


@pytest.fixture(autouse=True)
def _disable_decision_ledger(monkeypatch):
    """Тесты не пишут в реальный data/decision_ledger.jsonl."""
    monkeypatch.setenv("DECISION_LEDGER_ENABLED", "0")


@pytest.fixture
def ohlcv_uptrend():
    return make_ohlcv(n=100, trend=0.002)


@pytest.fixture
def ohlcv_flat():
    return make_ohlcv(n=100, trend=0.0)


@pytest.fixture
def ohlcv_downtrend():
    return make_ohlcv(n=100, trend=-0.002)
