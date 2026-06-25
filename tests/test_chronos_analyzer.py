"""Тесты chronos_analyzer — мокаем torch и ChronosPipeline."""

import sys
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_module_state():
    """Удаляет кэшированный модуль и сбрасывает глобальные переменные."""
    for mod_name in list(sys.modules.keys()):
        if "chronos_analyzer" in mod_name or mod_name == "chronos":
            del sys.modules[mod_name]


def _make_torch_mock():
    """Возвращает минимальный мок torch с поддержкой tensor/bfloat16."""
    torch_mock = MagicMock()
    torch_mock.bfloat16 = "bfloat16"
    torch_mock.float32 = "float32"

    # tensor(...).unsqueeze(0) → tensor-like mock
    tensor_instance = MagicMock()
    tensor_instance.unsqueeze = MagicMock(return_value=tensor_instance)
    torch_mock.tensor = MagicMock(return_value=tensor_instance)
    return torch_mock, tensor_instance


def _make_forecast_mock(median_value: float):
    """Возвращает мок forecast tensor с заданным медианным значением."""
    median_mock = MagicMock()
    median_mock.item = MagicMock(return_value=median_value)

    slice_mock = MagicMock()
    slice_mock.median = MagicMock(return_value=median_mock)

    forecast = MagicMock()
    forecast.__getitem__ = MagicMock(return_value=slice_mock)
    return forecast


# ---------------------------------------------------------------------------
# _load_pipeline()
# ---------------------------------------------------------------------------


class TestLoadPipeline:
    def setup_method(self):
        _reset_module_state()

    def test_load_sets_pipeline_on_success(self):
        torch_mock, _ = _make_torch_mock()
        chronos_mock = MagicMock()
        pipeline_instance = MagicMock()
        chronos_mock.ChronosPipeline.from_pretrained = MagicMock(
            return_value=pipeline_instance
        )
        with patch.dict("sys.modules", {"torch": torch_mock, "chronos": chronos_mock}):
            import src.chronos_analyzer as ca

            ca._load_attempted = False
            ca._pipeline = None
            ca._load_pipeline()
            assert ca._pipeline is pipeline_instance
            assert ca._load_attempted is True

    def test_load_sets_pipeline_none_on_import_error(self):
        """Если chronos не установлен — _pipeline остаётся None."""
        with patch.dict("sys.modules", {"torch": None, "chronos": None}):
            _reset_module_state()
            import src.chronos_analyzer as ca

            ca._load_attempted = False
            ca._pipeline = None
            ca._load_pipeline()
            assert ca._pipeline is None
            assert ca._load_attempted is True

    def test_load_skips_if_already_attempted(self):
        """Повторный вызов не пытается загрузить снова."""
        _reset_module_state()
        import src.chronos_analyzer as ca

        ca._load_attempted = True
        ca._pipeline = None  # остаётся None

        # Мок chronos никогда не должен вызываться
        chronos_mock = MagicMock()
        with patch.dict("sys.modules", {"chronos": chronos_mock}):
            ca._load_pipeline()
        chronos_mock.ChronosPipeline.from_pretrained.assert_not_called()
        assert ca._pipeline is None

    def test_load_graceful_on_exception(self):
        """Исключение при загрузке → pipeline=None, нет краша."""
        torch_mock, _ = _make_torch_mock()
        chronos_mock = MagicMock()
        chronos_mock.ChronosPipeline.from_pretrained.side_effect = RuntimeError(
            "CUDA OOM"
        )
        with patch.dict("sys.modules", {"torch": torch_mock, "chronos": chronos_mock}):
            _reset_module_state()
            import src.chronos_analyzer as ca

            ca._load_attempted = False
            ca._pipeline = None
            ca._load_pipeline()
            assert ca._pipeline is None


# ---------------------------------------------------------------------------
# predict_direction()
# ---------------------------------------------------------------------------


class TestPredictDirection:
    def setup_method(self):
        _reset_module_state()

    def _import(self):
        import src.chronos_analyzer as ca

        return ca

    def test_returns_neutral_when_pipeline_none(self):
        ca = self._import()
        ca._load_attempted = True
        ca._pipeline = None
        result = ca.predict_direction([100.0] * 20)
        assert result == "neutral"

    def test_returns_neutral_when_too_few_prices(self):
        ca = self._import()
        ca._load_attempted = True
        ca._pipeline = MagicMock()  # pipeline not None
        result = ca.predict_direction([100.0] * 9)  # < 10
        assert result == "neutral"

    def test_returns_up_when_median_above_threshold(self):
        ca = self._import()
        prices = [100.0] * 20
        # median = 100.5 → diff_pct = 0.005 > 0.002
        forecast_mock = _make_forecast_mock(100.5)
        pipeline_mock = MagicMock()
        pipeline_mock.predict = MagicMock(return_value=forecast_mock)
        ca._load_attempted = True
        ca._pipeline = pipeline_mock

        torch_mock, tensor_instance = _make_torch_mock()
        with patch.dict("sys.modules", {"torch": torch_mock}):
            result = ca.predict_direction(prices)
        assert result == "up"

    def test_returns_down_when_median_below_threshold(self):
        ca = self._import()
        prices = [100.0] * 20
        # median = 99.5 → diff_pct = -0.005 < -0.002
        forecast_mock = _make_forecast_mock(99.5)
        pipeline_mock = MagicMock()
        pipeline_mock.predict = MagicMock(return_value=forecast_mock)
        ca._load_attempted = True
        ca._pipeline = pipeline_mock

        torch_mock, tensor_instance = _make_torch_mock()
        with patch.dict("sys.modules", {"torch": torch_mock}):
            result = ca.predict_direction(prices)
        assert result == "down"

    def test_returns_neutral_when_median_within_threshold(self):
        ca = self._import()
        prices = [100.0] * 20
        # median = 100.1 → diff_pct = 0.001 < 0.002
        forecast_mock = _make_forecast_mock(100.1)
        pipeline_mock = MagicMock()
        pipeline_mock.predict = MagicMock(return_value=forecast_mock)
        ca._load_attempted = True
        ca._pipeline = pipeline_mock

        torch_mock, _ = _make_torch_mock()
        with patch.dict("sys.modules", {"torch": torch_mock}):
            result = ca.predict_direction(prices)
        assert result == "neutral"

    def test_returns_neutral_when_last_price_is_zero(self):
        ca = self._import()
        prices = [0.0] * 20
        forecast_mock = _make_forecast_mock(0.0)
        pipeline_mock = MagicMock()
        pipeline_mock.predict = MagicMock(return_value=forecast_mock)
        ca._load_attempted = True
        ca._pipeline = pipeline_mock

        torch_mock, _ = _make_torch_mock()
        with patch.dict("sys.modules", {"torch": torch_mock}):
            result = ca.predict_direction(prices)
        assert result == "neutral"

    def test_returns_neutral_on_exception_in_predict(self):
        ca = self._import()
        prices = [100.0] * 20
        pipeline_mock = MagicMock()
        pipeline_mock.predict.side_effect = RuntimeError("predict failed")
        ca._load_attempted = True
        ca._pipeline = pipeline_mock

        torch_mock, _ = _make_torch_mock()
        with patch.dict("sys.modules", {"torch": torch_mock}):
            result = ca.predict_direction(prices)
        assert result == "neutral"

    def test_custom_threshold_respected(self):
        ca = self._import()
        prices = [100.0] * 20
        # median = 100.3 → diff_pct = 0.003
        # With default threshold=0.002 → "up"; with threshold=0.005 → "neutral"
        forecast_mock = _make_forecast_mock(100.3)
        pipeline_mock = MagicMock()
        pipeline_mock.predict = MagicMock(return_value=forecast_mock)
        ca._load_attempted = True
        ca._pipeline = pipeline_mock

        torch_mock, _ = _make_torch_mock()
        with patch.dict("sys.modules", {"torch": torch_mock}):
            result_default = ca.predict_direction(prices)
            result_high_threshold = ca.predict_direction(prices, threshold_pct=0.005)
        assert result_default == "up"
        assert result_high_threshold == "neutral"

    def test_context_truncated_to_64_prices(self):
        """Только последние 64 цены используются как контекст."""
        ca = self._import()
        # 100 prices, last 64 have value 200.0
        prices = [100.0] * 36 + [200.0] * 64

        captured_prices = []

        def fake_tensor(p, dtype=None):
            captured_prices.extend(p)
            t = MagicMock()
            t.unsqueeze = MagicMock(return_value=t)
            return t

        forecast_mock = _make_forecast_mock(200.0)
        pipeline_mock = MagicMock()
        pipeline_mock.predict = MagicMock(return_value=forecast_mock)
        ca._load_attempted = True
        ca._pipeline = pipeline_mock

        torch_mock, _ = _make_torch_mock()
        torch_mock.tensor = fake_tensor
        with patch.dict("sys.modules", {"torch": torch_mock}):
            ca.predict_direction(prices)

        assert len(captured_prices) == 64
        assert all(p == 200.0 for p in captured_prices)


# ---------------------------------------------------------------------------
# is_available()
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def setup_method(self):
        _reset_module_state()

    def test_returns_false_when_pipeline_none(self):
        import src.chronos_analyzer as ca

        ca._load_attempted = True
        ca._pipeline = None
        assert ca.is_available() is False

    def test_returns_true_when_pipeline_loaded(self):
        import src.chronos_analyzer as ca

        ca._load_attempted = True
        ca._pipeline = MagicMock()
        assert ca.is_available() is True

    def test_is_available_triggers_load(self):
        """is_available() вызывает _load_pipeline() если ещё не загружали."""
        import src.chronos_analyzer as ca

        ca._load_attempted = False
        ca._pipeline = None
        with patch.object(ca, "_load_pipeline") as mock_load:
            mock_load.side_effect = lambda: None  # no-op
            ca.is_available()
        mock_load.assert_called_once()
