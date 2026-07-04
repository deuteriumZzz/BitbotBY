"""Тесты crypto_sentiment_analyzer — мокаем transformers.pipeline."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_module_state():
    """Удаляет кэшированный модуль и сбрасывает глобальные переменные."""
    for mod_name in list(sys.modules.keys()):
        if "crypto_sentiment_analyzer" in mod_name or mod_name == "transformers":
            del sys.modules[mod_name]


def _make_transformers_mock():
    """Возвращает мок transformers с pipeline(), возвращающим callable-мок."""
    transformers_mock = MagicMock()
    pipeline_callable = MagicMock()
    transformers_mock.pipeline = MagicMock(return_value=pipeline_callable)
    return transformers_mock, pipeline_callable


# ---------------------------------------------------------------------------
# _load_pipeline()
# ---------------------------------------------------------------------------


class TestLoadPipeline:
    def setup_method(self):
        _reset_module_state()

    def test_load_sets_pipeline_on_success(self):
        transformers_mock, pipeline_callable = _make_transformers_mock()
        with patch.dict("sys.modules", {"transformers": transformers_mock}):
            import src.crypto_sentiment_analyzer as csa

            csa._load_attempted = False
            csa._pipeline = None
            csa._load_pipeline()
            assert csa._pipeline is pipeline_callable
            assert csa._load_attempted is True

    def test_load_sets_pipeline_none_on_import_error(self):
        """Если transformers не установлен — _pipeline остаётся None."""
        with patch.dict("sys.modules", {"transformers": None}):
            _reset_module_state()
            import src.crypto_sentiment_analyzer as csa

            csa._load_attempted = False
            csa._pipeline = None
            csa._load_pipeline()
            assert csa._pipeline is None
            assert csa._load_attempted is True

    def test_load_skips_if_already_attempted(self):
        """Повторный вызов не пытается загрузить снова."""
        _reset_module_state()
        import src.crypto_sentiment_analyzer as csa

        csa._load_attempted = True
        csa._pipeline = None  # остаётся None

        transformers_mock, _ = _make_transformers_mock()
        with patch.dict("sys.modules", {"transformers": transformers_mock}):
            csa._load_pipeline()
        transformers_mock.pipeline.assert_not_called()
        assert csa._pipeline is None

    def test_load_graceful_on_exception(self):
        """Исключение при загрузке → pipeline=None, нет краша."""
        transformers_mock = MagicMock()
        transformers_mock.pipeline.side_effect = RuntimeError("model fetch failed")
        with patch.dict("sys.modules", {"transformers": transformers_mock}):
            _reset_module_state()
            import src.crypto_sentiment_analyzer as csa

            csa._load_attempted = False
            csa._pipeline = None
            csa._load_pipeline()
            assert csa._pipeline is None


# ---------------------------------------------------------------------------
# predict_sentiment()
# ---------------------------------------------------------------------------


class TestPredictSentiment:
    def setup_method(self):
        _reset_module_state()

    def _import(self):
        import src.crypto_sentiment_analyzer as csa

        return csa

    def test_returns_neutral_when_pipeline_none(self):
        csa = self._import()
        csa._load_attempted = True
        csa._pipeline = None
        result = csa.predict_sentiment(["to the moon"])
        assert result == (0.0, "neutral")

    def test_returns_neutral_when_no_texts(self):
        csa = self._import()
        csa._load_attempted = True
        csa._pipeline = MagicMock()
        result = csa.predict_sentiment([])
        assert result == (0.0, "neutral")

    def test_returns_neutral_when_texts_are_blank(self):
        csa = self._import()
        csa._load_attempted = True
        csa._pipeline = MagicMock()
        result = csa.predict_sentiment(["", "   "])
        assert result == (0.0, "neutral")

    def test_returns_bullish_above_threshold(self):
        csa = self._import()
        csa._load_attempted = True
        pipeline_mock = MagicMock()
        pipeline_mock.return_value = [
            [
                {"label": "Bullish", "score": 0.9},
                {"label": "Bearish", "score": 0.05},
                {"label": "Neutral", "score": 0.05},
            ]
        ]
        csa._pipeline = pipeline_mock
        net_score, signal = csa.predict_sentiment(["to the moon"])
        assert signal == "reddit_bullish"
        assert net_score == pytest.approx(0.85, abs=0.001)

    def test_returns_bearish_below_threshold(self):
        csa = self._import()
        csa._load_attempted = True
        pipeline_mock = MagicMock()
        pipeline_mock.return_value = [
            [
                {"label": "Bullish", "score": 0.05},
                {"label": "Bearish", "score": 0.9},
                {"label": "Neutral", "score": 0.05},
            ]
        ]
        csa._pipeline = pipeline_mock
        net_score, signal = csa.predict_sentiment(["rugpull incoming"])
        assert signal == "reddit_bearish"
        assert net_score == pytest.approx(-0.85, abs=0.001)

    def test_returns_neutral_within_threshold(self):
        csa = self._import()
        csa._load_attempted = True
        pipeline_mock = MagicMock()
        pipeline_mock.return_value = [
            [
                {"label": "Bullish", "score": 0.4},
                {"label": "Bearish", "score": 0.35},
                {"label": "Neutral", "score": 0.25},
            ]
        ]
        csa._pipeline = pipeline_mock
        net_score, signal = csa.predict_sentiment(["mixed feelings about this"])
        assert signal == "neutral"
        assert net_score == pytest.approx(0.05, abs=0.001)

    def test_averages_multiple_texts(self):
        csa = self._import()
        csa._load_attempted = True
        pipeline_mock = MagicMock()
        pipeline_mock.return_value = [
            [{"label": "Bullish", "score": 1.0}, {"label": "Bearish", "score": 0.0}],
            [{"label": "Bullish", "score": 0.0}, {"label": "Bearish", "score": 0.0}],
        ]
        csa._pipeline = pipeline_mock
        net_score, _ = csa.predict_sentiment(["moon", "meh"])
        assert net_score == pytest.approx(0.5, abs=0.001)  # (1.0 + 0.0) / 2

    def test_truncates_long_text(self):
        csa = self._import()
        csa._load_attempted = True
        pipeline_mock = MagicMock()
        pipeline_mock.return_value = [
            [{"label": "Neutral", "score": 1.0}],
        ]
        csa._pipeline = pipeline_mock
        long_text = "a" * 5000
        csa.predict_sentiment([long_text])
        called_texts = pipeline_mock.call_args[0][0]
        assert len(called_texts[0]) == 200

    def test_returns_neutral_on_exception(self):
        csa = self._import()
        csa._load_attempted = True
        pipeline_mock = MagicMock()
        pipeline_mock.side_effect = RuntimeError("inference failed")
        csa._pipeline = pipeline_mock
        result = csa.predict_sentiment(["to the moon"])
        assert result == (0.0, "neutral")


# ---------------------------------------------------------------------------
# is_available()
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def setup_method(self):
        _reset_module_state()

    def test_returns_false_when_pipeline_none(self):
        import src.crypto_sentiment_analyzer as csa

        csa._load_attempted = True
        csa._pipeline = None
        assert csa.is_available() is False

    def test_returns_true_when_pipeline_loaded(self):
        import src.crypto_sentiment_analyzer as csa

        csa._load_attempted = True
        csa._pipeline = MagicMock()
        assert csa.is_available() is True

    def test_is_available_triggers_load(self):
        """is_available() вызывает _load_pipeline() если ещё не загружали."""
        import src.crypto_sentiment_analyzer as csa

        csa._load_attempted = False
        csa._pipeline = None
        with patch.object(csa, "_load_pipeline") as mock_load:
            mock_load.side_effect = lambda: None  # no-op
            csa.is_available()
        mock_load.assert_called_once()
