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


class TestRuntimeConfigSymbols:
    def test_get_forced_symbols_empty(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.get_forced_symbols() == set()

    def test_get_forced_symbols_bytes(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        redis_mock.redis_client.smembers.return_value = {b"BTC/USDT", b"ETH/USDT"}
        assert rc.get_forced_symbols() == {"BTC/USDT", "ETH/USDT"}

    def test_add_forced_symbol(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        rc.add_forced_symbol("SOL/USDT")
        redis_mock.redis_client.sadd.assert_called_once()

    def test_remove_forced_symbol(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        rc.remove_forced_symbol("SOL/USDT")
        redis_mock.redis_client.srem.assert_called_once()

    def test_add_excluded_symbol(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        rc.add_excluded_symbol("XRP/USDT")
        redis_mock.redis_client.sadd.assert_called_once()

    def test_remove_excluded_symbol(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        rc.remove_excluded_symbol("XRP/USDT")
        redis_mock.redis_client.srem.assert_called_once()

    def test_smembers_error_returns_empty_set(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        redis_mock.redis_client.smembers.side_effect = Exception("redis error")
        assert rc._smembers("any_key") == set()


class TestRuntimeConfigRisk:
    def test_get_max_positions_default(self, rc_bundle):
        rc, _, cfg = rc_bundle
        cfg.MAX_POSITIONS = 5
        assert rc.get_max_positions() == 5

    def test_set_max_positions_valid(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        assert rc.set_max_positions(3) is True

    def test_set_max_positions_too_large(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.set_max_positions(21) is False

    def test_get_risk_per_trade_default(self, rc_bundle):
        rc, _, cfg = rc_bundle
        cfg.RISK_PER_TRADE = 0.02
        assert rc.get_risk_per_trade() == pytest.approx(0.02)

    def test_set_risk_per_trade_valid(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.set_risk_per_trade(0.02) is True

    def test_set_risk_per_trade_too_large(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.set_risk_per_trade(0.6) is False

    def test_get_drawdown_scale_default(self, rc_bundle):
        rc, _, cfg = rc_bundle
        cfg.DRAWDOWN_SCALE_ENABLED = True
        assert rc.get_drawdown_scale_enabled() is True

    def test_set_drawdown_scale(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        rc.set_drawdown_scale_enabled(False)
        redis_mock.redis_client.set.assert_called_once()

    def test_apply_risk_preset_conservative(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.apply_risk_preset("conservative") is True

    def test_apply_risk_preset_unknown(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.apply_risk_preset("unknown_preset") is False

    def test_get_risk_summary_returns_dict(self, rc_bundle):
        rc, _, _ = rc_bundle
        summary = rc.get_risk_summary()
        assert "max_positions" in summary
        assert "risk_per_trade" in summary
        assert "drawdown_scale_enabled" in summary


class TestRuntimeConfigTradingHours:
    def test_get_trading_hours_default(self, rc_bundle):
        rc, _, cfg = rc_bundle
        cfg.TRADING_HOURS = ""
        assert rc.get_trading_hours() == ""

    def test_set_trading_hours_valid(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.set_trading_hours("9-22") is True

    def test_set_trading_hours_empty_resets(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.set_trading_hours("") is True

    def test_set_trading_hours_invalid_format(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.set_trading_hours("not-valid") is False

    def test_set_trading_hours_same_start_end(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.set_trading_hours("9-9") is False

    def test_is_trading_time_no_hours(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.is_trading_time() is True

    def test_is_trading_time_with_hours(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        redis_mock.redis_client.get.return_value = b"0-23"
        result = rc.is_trading_time()
        assert isinstance(result, bool)


class TestRuntimeConfigStrategies:
    def test_get_disabled_strategies_empty(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.get_disabled_strategies() == set()

    def test_disable_strategy(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        rc.disable_strategy("scalping")
        redis_mock.redis_client.sadd.assert_called_once()

    def test_enable_strategy(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        rc.enable_strategy("scalping")
        redis_mock.redis_client.srem.assert_called_once()

    def test_reset_strategies(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        rc.reset_strategies()
        redis_mock.redis_client.delete.assert_called_once()

    def test_toggle_strategy_disables_when_enabled(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        redis_mock.redis_client.smembers.return_value = set()
        result = rc.toggle_strategy("scalping")
        assert result is False

    def test_toggle_strategy_enables_when_disabled(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        redis_mock.redis_client.smembers.return_value = {b"scalping"}
        result = rc.toggle_strategy("scalping")
        assert result is True


class TestRuntimeConfigChronos:
    def test_get_chronos_disabled_by_default(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.get_chronos_enabled() is False

    def test_get_chronos_enabled_from_redis(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        redis_mock.redis_client.get.return_value = b"1"
        assert rc.get_chronos_enabled() is True

    def test_set_chronos_enabled(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        rc.set_chronos_enabled(True)
        redis_mock.redis_client.set.assert_called_once()


class TestRuntimeConfigMarketProfile:
    def test_get_market_profile_empty_default(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.get_market_profile() == ""

    def test_get_market_profile_from_redis(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        redis_mock.redis_client.get.return_value = b"bluechip"
        assert rc.get_market_profile() == "bluechip"

    def test_apply_market_profile_bluechip(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.apply_market_profile("bluechip") is True

    def test_apply_market_profile_altcoin(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.apply_market_profile("altcoin") is True

    def test_apply_market_profile_unknown(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.apply_market_profile("unknown") is False

    def test_get_market_profiles_info_returns_dict(self, rc_bundle):
        rc, _, _ = rc_bundle
        info = rc.get_market_profiles_info()
        assert "bluechip" in info
        assert "altcoin" in info

    def test_get_market_profile_config(self, rc_bundle):
        rc, _, _ = rc_bundle
        cfg = rc.get_market_profile_config("bluechip")
        assert isinstance(cfg, dict)
        assert len(cfg) > 0


class TestRuntimeConfigPaperTrading:
    def test_get_paper_trading_override_none_by_default(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.get_paper_trading_override() is None

    def test_get_paper_trading_override_true(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        redis_mock.redis_client.get.return_value = b"1"
        assert rc.get_paper_trading_override() is True

    def test_get_paper_trading_override_false(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        redis_mock.redis_client.get.return_value = b"0"
        assert rc.get_paper_trading_override() is False

    def test_set_paper_trading_override(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        rc.set_paper_trading_override(True)
        redis_mock.redis_client.set.assert_called_once()


class TestRuntimeConfigSeasonMode:
    def test_get_season_switch_mode_default(self, rc_bundle):
        rc, _, _ = rc_bundle
        result = rc.get_season_switch_mode()
        assert result in ("alert", "auto")

    def test_set_season_switch_mode_valid(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.set_season_switch_mode("auto") is True
        assert rc.set_season_switch_mode("alert") is True

    def test_set_season_switch_mode_invalid(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.set_season_switch_mode("invalid") is False


class TestRuntimeConfigConfirmTimeout:
    def test_get_confirm_timeout_default(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.get_confirm_timeout() == 60

    def test_set_confirm_timeout_valid(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.set_confirm_timeout(30) is True

    def test_set_confirm_timeout_zero(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.set_confirm_timeout(0) is True

    def test_set_confirm_timeout_too_small(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.set_confirm_timeout(5) is False


class TestRuntimeConfigSAC:
    def test_is_sac_prompted_false_by_default(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.is_sac_prompted() is False

    def test_is_sac_prompted_with_profile(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        redis_mock.redis_client.get.return_value = b"1"
        assert rc.is_sac_prompted("bluechip") is True

    def test_set_sac_prompted(self, rc_bundle):
        rc, redis_mock, _ = rc_bundle
        rc.set_sac_prompted("altcoin")
        redis_mock.redis_client.set.assert_called_once()


class TestRuntimeConfigLeverage:
    def test_get_leverage_mode_default(self, rc_bundle):
        rc, _, cfg = rc_bundle
        cfg.LEVERAGE_MODE = "volatility"
        assert rc.get_leverage_mode() == "volatility"

    def test_set_leverage_mode_valid(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.set_leverage_mode("fixed") is True
        assert rc.set_leverage_mode("volatility") is True
        assert rc.set_leverage_mode("full") is True

    def test_set_leverage_mode_invalid(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.set_leverage_mode("turbo") is False

    def test_get_leverage_target_risk_default(self, rc_bundle):
        rc, _, cfg = rc_bundle
        cfg.LEVERAGE_TARGET_RISK = 0.01
        assert rc.get_leverage_target_risk() == pytest.approx(0.01)

    def test_set_leverage_target_risk_valid(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.set_leverage_target_risk(0.02) is True

    def test_set_leverage_target_risk_invalid(self, rc_bundle):
        rc, _, _ = rc_bundle
        assert rc.set_leverage_target_risk(2.0) is False
