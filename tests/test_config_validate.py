import pytest

from config import Config


def _paper(**kwargs) -> Config:
    """Минимально валидная конфигурация paper trading."""
    defaults = dict(
        PAPER_TRADING=True,
        INITIAL_BALANCE=1000.0,
        RISK_PER_TRADE=0.02,
        MAX_POSITION_SIZE=0.1,
        COMMISSION_RATE=0.001,
        TRADING_INTERVAL=30,
        SYMBOL="BTC/USDT",
        MODE="local",
        AI_PROVIDER="auto",
        ANTHROPIC_API_KEY="",
        DEEPSEEK_API_KEY="",
        OPENAI_API_KEY="",
    )
    defaults.update(kwargs)
    return Config(**defaults)


class TestConfigValidate:
    def test_valid_paper_trading_passes(self):
        _paper().validate()

    def test_stablecoin_symbol_rejected(self):
        with pytest.raises(ValueError, match="stablecoin"):
            _paper(SYMBOL="USDC/USDT").validate()

    def test_busd_symbol_rejected(self):
        with pytest.raises(ValueError, match="stablecoin"):
            _paper(SYMBOL="BUSD/USDT").validate()

    def test_zero_initial_balance_rejected(self):
        with pytest.raises(ValueError, match="INITIAL_BALANCE"):
            _paper(INITIAL_BALANCE=0.0).validate()

    def test_negative_initial_balance_rejected(self):
        with pytest.raises(ValueError, match="INITIAL_BALANCE"):
            _paper(INITIAL_BALANCE=-500.0).validate()

    def test_risk_above_1_rejected(self):
        with pytest.raises(ValueError, match="RISK_PER_TRADE"):
            _paper(RISK_PER_TRADE=1.5).validate()

    def test_risk_zero_rejected(self):
        with pytest.raises(ValueError, match="RISK_PER_TRADE"):
            _paper(RISK_PER_TRADE=0.0).validate()

    def test_invalid_ai_provider_rejected(self):
        with pytest.raises(ValueError, match="AI_PROVIDER"):
            _paper(MODE="ai", AI_PROVIDER="gemini", ANTHROPIC_API_KEY="key").validate()

    def test_missing_bybit_keys_in_live_mode(self):
        with pytest.raises(ValueError, match="BYBIT_API_KEY"):
            _paper(
                PAPER_TRADING=False,
                BYBIT_API_KEY="",
                BYBIT_API_SECRET="",
            ).validate()

    def test_ai_mode_no_keys_rejected(self):
        with pytest.raises(ValueError, match="нужен хотя бы один ключ"):
            _paper(
                MODE="ai",
                AI_PROVIDER="auto",
                ANTHROPIC_API_KEY="",
                DEEPSEEK_API_KEY="",
                OPENAI_API_KEY="",
            ).validate()

    def test_anthropic_provider_without_key_rejected(self):
        with pytest.raises(ValueError, match="ключ не задан"):
            _paper(
                MODE="ai",
                AI_PROVIDER="anthropic",
                ANTHROPIC_API_KEY="",
            ).validate()

    def test_deepseek_provider_without_key_rejected(self):
        with pytest.raises(ValueError, match="ключ не задан"):
            _paper(
                MODE="ai",
                AI_PROVIDER="deepseek",
                DEEPSEEK_API_KEY="",
            ).validate()

    def test_ai_mode_with_any_key_passes(self):
        _paper(
            MODE="ai",
            AI_PROVIDER="auto",
            DEEPSEEK_API_KEY="sk-test",
        ).validate()

    def test_local_mode_no_keys_needed(self):
        _paper(MODE="local").validate()
