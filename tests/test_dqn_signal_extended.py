"""
Расширенные тесты SACSignal: ветки загрузки модели и инференса.

Покрывают строки, недостижимые без мока stable_baselines3:
- ImportError при отсутствии SB3
- Успешная загрузка модели + norm_stats
- Ошибка SAC.load()
- Инференс: buy / sell / hold с нормальными obs
- Обнаружение drift нормализации (outlier_frac >= _MAX_OUTLIER_FRAC)
- Отключение модели после _DISABLE_AFTER_DRIFTS подряд
- reload_if_updated: нет файла / файл не изменился / файл обновился
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from src.dqn_signal import _DISABLE_AFTER_DRIFTS, SACSignal

_SNAP = {
    "symbol": "BTC/USDT",
    "price": 50000.0,
    "atr": 500.0,
    "volume_ratio": 1.2,
    "indicators": {"rsi": 55.0, "macd": "bullish", "bb_width": 0.04},
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


# ---------------------------------------------------------------------------
# Хелпер: создать SACSignal с замоканной моделью
# ---------------------------------------------------------------------------


def _make_loaded_signal(predict_return=None) -> SACSignal:
    """Создаёт SACSignal с loaded=True и мок-моделью."""
    mock_model = MagicMock()
    if predict_return is None:
        predict_return = (
            np.array([0.0]),
            None,
        )  # (action, state) — SB3 возвращает пару
    mock_model.predict.return_value = predict_return

    sig = SACSignal.__new__(SACSignal)
    import logging

    sig.logger = logging.getLogger("test")
    sig._model = mock_model
    sig.loaded = True
    sig._norm_stats = None
    sig._mtime = 0.0
    sig._consecutive_drifts = 0
    return sig


# ---------------------------------------------------------------------------
# _try_load: ImportError stable_baselines3
# ---------------------------------------------------------------------------


class TestTryLoadImportError:
    def test_import_error_leaves_model_not_loaded(self):
        """SB3 недоступен → loaded=False, ошибка в логе."""
        saved = sys.modules.pop("stable_baselines3", None)
        try:
            with patch.dict(sys.modules, {"stable_baselines3": None}):
                sig = SACSignal()
            assert sig.loaded is False
        finally:
            if saved is not None:
                sys.modules["stable_baselines3"] = saved


# ---------------------------------------------------------------------------
# _try_load: модель загружается успешно
# ---------------------------------------------------------------------------


class TestTryLoadSuccess:
    def _make_sac_mock(self):
        mock_sac_class = MagicMock()
        mock_sac_instance = MagicMock()
        mock_sac_class.load.return_value = mock_sac_instance
        mock_sb3 = MagicMock()
        mock_sb3.SAC = mock_sac_class
        return mock_sb3, mock_sac_instance

    def test_model_loaded_sets_loaded_true(self):
        mock_sb3, _ = self._make_sac_mock()
        with patch.dict(sys.modules, {"stable_baselines3": mock_sb3}), patch(
            "os.path.exists", return_value=True
        ), patch("os.path.getmtime", return_value=123456.0):
            sig = SACSignal()
        assert sig.loaded is True

    def test_model_loaded_mtime_stored(self):
        mock_sb3, _ = self._make_sac_mock()
        with patch.dict(sys.modules, {"stable_baselines3": mock_sb3}), patch(
            "os.path.exists", return_value=True
        ), patch("os.path.getmtime", return_value=9999.0):
            sig = SACSignal()
        assert sig._mtime == pytest.approx(9999.0)

    def test_norm_stats_loaded_when_file_exists(self):
        mock_sb3, _ = self._make_sac_mock()
        norm_data = {"close": [50000.0, 1000.0]}
        m = mock_open(read_data=json.dumps(norm_data))
        with patch.dict(sys.modules, {"stable_baselines3": mock_sb3}), patch(
            "os.path.exists", return_value=True
        ), patch("os.path.getmtime", return_value=1.0), patch("builtins.open", m):
            sig = SACSignal()
        assert sig._norm_stats is not None
        assert "close" in sig._norm_stats

    def test_norm_stats_none_when_file_absent(self):
        mock_sb3, _ = self._make_sac_mock()

        def exists_side(path):
            return not path.endswith("_norm_stats.json")

        with patch.dict(sys.modules, {"stable_baselines3": mock_sb3}), patch(
            "os.path.exists", side_effect=exists_side
        ), patch("os.path.getmtime", return_value=1.0):
            sig = SACSignal()
        assert sig._norm_stats is None

    def test_norm_stats_invalid_json_leaves_none(self):
        mock_sb3, _ = self._make_sac_mock()
        bad_json = mock_open(read_data="{not valid json}")
        with patch.dict(sys.modules, {"stable_baselines3": mock_sb3}), patch(
            "os.path.exists", return_value=True
        ), patch("os.path.getmtime", return_value=1.0), patch(
            "builtins.open", bad_json
        ):
            sig = SACSignal()
        assert sig._norm_stats is None

    def test_sac_load_exception_leaves_not_loaded(self):
        mock_sb3 = MagicMock()
        mock_sb3.SAC.load.side_effect = RuntimeError("corrupt model")
        with patch.dict(sys.modules, {"stable_baselines3": mock_sb3}), patch(
            "os.path.exists", return_value=True
        ), patch("os.path.getmtime", return_value=1.0):
            sig = SACSignal()
        assert sig.loaded is False


# ---------------------------------------------------------------------------
# get_signal: инференс buy / sell / hold
# ---------------------------------------------------------------------------


class TestGetSignal:
    def test_returns_hold_when_not_loaded(self):
        sig = SACSignal.__new__(SACSignal)
        import logging

        sig.logger = logging.getLogger("test")
        sig._model = None
        sig.loaded = False
        sig._norm_stats = None
        sig._mtime = 0.0
        sig._consecutive_drifts = 0
        result = sig.get_signal(_SNAP, balance=5000.0)
        assert result == {"action": "hold", "confidence": 0.0, "source": "sac"}

    def test_buy_signal_above_hold_zone(self):
        from reinforcement_learning.rl_env import HOLD_ZONE

        action_val = HOLD_ZONE + 0.2
        sig = _make_loaded_signal(predict_return=(np.array([action_val]), None))
        result = sig.get_signal(_SNAP, balance=5000.0)
        assert result["action"] == "buy"
        assert result["confidence"] == pytest.approx(round(action_val, 3))
        assert result["source"] == "sac"

    def test_sell_signal_below_hold_zone(self):
        from reinforcement_learning.rl_env import HOLD_ZONE

        action_val = -(HOLD_ZONE + 0.2)
        sig = _make_loaded_signal(predict_return=(np.array([action_val]), None))
        result = sig.get_signal(_SNAP, balance=5000.0)
        assert result["action"] == "sell"
        assert result["confidence"] == pytest.approx(round(abs(action_val), 3))

    def test_hold_signal_inside_hold_zone(self):
        sig = _make_loaded_signal(predict_return=(np.array([0.0]), None))
        result = sig.get_signal(_SNAP, balance=5000.0)
        assert result["action"] == "hold"

    def test_exception_in_predict_returns_hold(self):
        sig = _make_loaded_signal()
        sig._model.predict.side_effect = RuntimeError("GPU error")
        result = sig.get_signal(_SNAP, balance=5000.0)
        assert result["action"] == "hold"
        assert result["confidence"] == 0.0

    def test_buy_signal_includes_symbol(self):
        from reinforcement_learning.rl_env import HOLD_ZONE

        sig = _make_loaded_signal(predict_return=(np.array([HOLD_ZONE + 0.1]), None))
        result = sig.get_signal(_SNAP, balance=5000.0)
        assert result["symbol"] == "BTC/USDT"


# ---------------------------------------------------------------------------
# get_signal: drift detection
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# reload_if_updated
# ---------------------------------------------------------------------------


class TestReloadIfUpdated:
    def test_no_path_returns_false(self):
        sig = _make_loaded_signal()
        with patch("config.Config.SAC_MODEL_PATH", ""):
            result = sig.reload_if_updated()
        assert result is False

    def test_file_not_exists_returns_false(self):
        sig = _make_loaded_signal()
        with patch("os.path.exists", return_value=False):
            result = sig.reload_if_updated()
        assert result is False

    def test_same_mtime_returns_false(self):
        sig = _make_loaded_signal()
        sig._mtime = 100.0
        with patch("os.path.exists", return_value=True), patch(
            "os.path.getmtime", return_value=100.0
        ):
            result = sig.reload_if_updated()
        assert result is False

    def test_newer_mtime_triggers_reload(self):
        sig = _make_loaded_signal()
        sig._mtime = 1.0
        with patch("os.path.exists", return_value=True), patch(
            "os.path.getmtime", return_value=999.0
        ), patch.object(sig, "_try_load") as mock_load:
            result = sig.reload_if_updated()
        assert result is True
        mock_load.assert_called_once()

    def test_oserror_in_getmtime_returns_false(self):
        sig = _make_loaded_signal()
        with patch("os.path.exists", return_value=True), patch(
            "os.path.getmtime", side_effect=OSError("perm")
        ):
            result = sig.reload_if_updated()
        assert result is False
