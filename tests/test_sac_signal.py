"""
Тесты инференса SAC: _snap_to_obs() и DQNSignal.

Проверяют:
- форму и тип выходного вектора наблюдения
- корректное использование реальных OHLCV из snap["ohlcv"]
- fallback к snap["price"] при отсутствии ohlcv
- декодирование строкового MACD ("bullish"/"bearish")
- нормализацию через norm_stats
- поведение DQNSignal без модели (загрузки нет → hold)
"""

from __future__ import annotations

import numpy as np
import pytest

from reinforcement_learning.rl_env import OBS_DIM
from src.dqn_signal import DQNSignal, _snap_to_obs

# ── Фикстуры снэпшотов ────────────────────────────────

_SNAP_MINIMAL = {
    "symbol": "BTC/USDT",
    "price": 50000.0,
    "atr": 500.0,
    "volume_ratio": 1.2,
    "indicators": {
        "rsi": 55.0,
        "macd": "bullish",
        "bb_width": 0.04,
    },
}

_SNAP_WITH_OHLCV = {
    **_SNAP_MINIMAL,
    "ohlcv": {
        "open": 49800.0,
        "high": 50500.0,
        "low": 49500.0,
        "close": 50000.0,
        "volume": 1234.5,
        "macd": 150.0,
        "macd_signal": 120.0,
    },
}

_SNAP_BEARISH = {
    **_SNAP_MINIMAL,
    "indicators": {**_SNAP_MINIMAL["indicators"], "macd": "bearish"},
}


# ── Тесты _snap_to_obs ────────────────────────────────


def test_obs_shape_and_dtype():
    obs = _snap_to_obs(_SNAP_MINIMAL, balance=5000.0, norm_stats=None)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


def test_ohlcv_values_used_when_present():
    obs = _snap_to_obs(_SNAP_WITH_OHLCV, balance=5000.0, norm_stats=None)
    assert obs[0] == pytest.approx(49800.0)  # open
    assert obs[1] == pytest.approx(50500.0)  # high
    assert obs[2] == pytest.approx(49500.0)  # low
    assert obs[3] == pytest.approx(50000.0)  # close
    assert obs[4] == pytest.approx(1234.5)  # volume
    assert obs[6] == pytest.approx(150.0)  # macd (числовой)
    assert obs[7] == pytest.approx(120.0)  # macd_signal


def test_fallback_to_price_without_ohlcv():
    obs = _snap_to_obs(_SNAP_MINIMAL, balance=5000.0, norm_stats=None)
    price = 50000.0
    assert obs[0] == pytest.approx(price)  # open
    assert obs[1] == pytest.approx(price)  # high
    assert obs[2] == pytest.approx(price)  # low
    assert obs[3] == pytest.approx(price)  # close


def test_macd_string_bullish_maps_to_plus_one():
    obs = _snap_to_obs(_SNAP_MINIMAL, balance=5000.0, norm_stats=None)
    assert obs[6] == pytest.approx(1.0)


def test_macd_string_bearish_maps_to_minus_one():
    obs = _snap_to_obs(_SNAP_BEARISH, balance=5000.0, norm_stats=None)
    assert obs[6] == pytest.approx(-1.0)


def test_normalization_applied():
    norm_stats = {
        "close": [50000.0, 1000.0],
        "rsi": [50.0, 10.0],
    }
    obs = _snap_to_obs(_SNAP_WITH_OHLCV, balance=5000.0, norm_stats=norm_stats)
    # close: (50000 - 50000) / 1000 = 0.0
    assert obs[3] == pytest.approx(0.0)
    # rsi: (55 - 50) / 10 = 0.5
    assert obs[5] == pytest.approx(0.5)


def test_volume_fallback_uses_training_mean():
    norm_stats = {"volume": [80000.0, 5000.0]}
    obs = _snap_to_obs(_SNAP_MINIMAL, balance=5000.0, norm_stats=norm_stats)
    # Нет ohlcv → используем mean=80000, нормализация: (80000-80000)/5000 = 0.0
    assert obs[4] == pytest.approx(0.0)


def test_portfolio_features_set_correctly():
    obs = _snap_to_obs(_SNAP_MINIMAL, balance=3000.0, norm_stats=None)
    assert obs[11] == pytest.approx(3000.0)  # balance
    assert obs[12] == pytest.approx(0.0)  # position (нет позиции при инференсе)
    assert obs[13] == pytest.approx(3000.0)  # current_value


# ── Тесты DQNSignal ───────────────────────────────────


def test_dqn_signal_not_loaded_without_model_file():
    sig = DQNSignal()
    assert sig.loaded is False


def test_dqn_signal_returns_hold_when_not_loaded():
    sig = DQNSignal()
    result = sig.get_signal(_SNAP_MINIMAL, balance=5000.0)
    assert result["action"] == "hold"
    assert result["confidence"] == 0.0
    assert result["source"] == "sac"


def test_dqn_signal_hold_with_ohlcv_snap_not_loaded():
    sig = DQNSignal()
    result = sig.get_signal(_SNAP_WITH_OHLCV, balance=5000.0)
    assert result["action"] == "hold"
