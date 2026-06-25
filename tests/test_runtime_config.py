"""Tests for RuntimeConfig — basic get/set methods via mocked Redis."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def rc_bundle():
    redis_mock = MagicMock()
    redis_mock.redis_client.get.return_value = None
    redis_mock.redis_client.smembers.return_value = set()

    cfg = MagicMock()
    cfg.MODE = "ai"
    cfg.SCAN_TOP_N = 20
    cfg.AUTO_EXECUTE = False
    cfg.AI_PROVIDER = "auto"
    cfg.LEVERAGE_MODE = "volatility"
    cfg.LEVERAGE_TARGET_RISK = 0.01
    cfg.MAX_DRAWDOWN_PERCENT = 0.15
    cfg.RISK_PER_TRADE = 0.02
    cfg.MAX_POSITIONS = 3
    cfg.DRAWDOWN_SCALE_ENABLED = True
    cfg.TRAIN_TOP_N = 20
    cfg.PAPER_TRADING = True
    cfg.TRADING_HOURS = ""
    cfg.MIN_VOLUME_USDT = 1_000_000
    cfg.MAX_VOLUME_USDT = 0
    cfg.MARKET_TYPE = "linear"
    cfg.TIMEFRAME = "15m"

    with patch("src.runtime_config.Config", cfg):
        from src.runtime_config import RuntimeConfig
        rc = RuntimeConfig(redis_mock)
        yield rc, redis_mock, cfg


class TestRuntimeConfigMode:
    def test_get_mode_returns_config_default_when_no_redis(self, rc_bundle):
        rc, _, cfg = rc_bundle
        cfg.MODE = "hybrid"
        assert rc.get_mode() == "hybrid"

    def test_get_mode_returns_redis_value_when_valid(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        redis_mock.redis_client.get.return_value = b"local"
        assert rc.get_mode() == "local"

    def test_set_mode_valid(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        result = rc.set_mode("hybrid")
        assert result is True
        redis_mock.redis_client.set.assert_called_once()

    def test_set_mode_invalid_returns_false(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        result = rc.set_mode("invalid_mode")
        assert result is False
        redis_mock.redis_client.set.assert_not_called()


class TestRuntimeConfigPause:
    def test_is_paused_false_by_default(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.is_paused() is False

    def test_is_paused_true_when_redis_returns_1(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        redis_mock.redis_client.get.return_value = b"1"
        assert rc.is_paused() is True

    def test_set_paused_writes_to_redis(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        rc.set_paused(True)
        redis_mock.redis_client.set.assert_called_once()


class TestRuntimeConfigAutoExecute:
    def test_get_auto_execute_returns_config_default(self, rc_bundle):
        rc, _, cfg = rc_bundle
        cfg.AUTO_EXECUTE = True
        assert rc.get_auto_execute() is True

    def test_get_auto_execute_from_redis(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        redis_mock.redis_client.get.return_value = b"1"
        assert rc.get_auto_execute() is True


class TestRuntimeConfigScanTopN:
    def test_get_scan_top_n_default(self, rc_bundle):
        rc, _, cfg = rc_bundle
        cfg.SCAN_TOP_N = 30
        assert rc.get_scan_top_n() == 30

    def test_set_scan_top_n_valid(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        assert rc.set_scan_top_n(50) is True

    def test_set_scan_top_n_too_large(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.set_scan_top_n(999) is False

    def test_set_scan_top_n_zero(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.set_scan_top_n(0) is False


class TestRuntimeConfigLastAIProvider:
    def test_get_last_ai_provider_empty_by_default(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.get_last_ai_provider() == ""

    def test_get_last_ai_provider_returns_stored_value(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        redis_mock.redis_client.get.return_value = b"gemini"
        assert rc.get_last_ai_provider() == "gemini"

    def test_set_last_ai_provider_writes_to_redis(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        rc.set_last_ai_provider("groq")
        redis_mock.redis_client.set.assert_called_once()

    def test_set_then_get_last_ai_provider(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        rc.set_last_ai_provider("deepseek")
        redis_mock.redis_client.get.return_value = b"deepseek"
        assert rc.get_last_ai_provider() == "deepseek"


class TestRuntimeConfigAIProvider:
    def test_get_ai_provider_default(self, rc_bundle):
        rc, _, cfg = rc_bundle
        cfg.AI_PROVIDER = "groq"
        assert rc.get_ai_provider() == "groq"

    def test_set_ai_provider_valid(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        assert rc.set_ai_provider("anthropic") is True

    def test_set_ai_provider_invalid(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.set_ai_provider("unknown_ai") is False


class TestRuntimeConfigGetSet:
    def test_get_returns_none_on_redis_error(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        redis_mock.redis_client.get.side_effect = Exception("connection refused")
        assert rc._get("any_key") is None

    def test_set_logs_warning_on_redis_error(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        redis_mock.redis_client.set.side_effect = Exception("write failed")
        rc._set("any_key", "value")

    def test_get_returns_string_value(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        redis_mock.redis_client.get.return_value = "string_value"
        assert rc._get("any_key") == "string_value"

    def test_get_returns_none_for_unexpected_type(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        redis_mock.redis_client.get.return_value = 12345
        assert rc._get("any_key") is None


class TestRuntimeConfigPauseUnpause:
    def test_set_paused_false_writes_0(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        rc.set_paused(False)
        redis_mock.redis_client.set.assert_called_once()

    def test_set_auto_execute_false(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        rc.set_auto_execute(False)
        redis_mock.redis_client.set.assert_called_once()
