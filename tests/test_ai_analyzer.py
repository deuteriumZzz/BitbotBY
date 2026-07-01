"""Тесты AIAnalyzer и вспомогательных функций."""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs):
    """Возвращает MagicMock, имитирующий Config с заданными атрибутами."""
    cfg = MagicMock()
    cfg.AI_PROVIDER = kwargs.get("AI_PROVIDER", "auto")
    cfg.ANTHROPIC_API_KEY = kwargs.get("ANTHROPIC_API_KEY", "")
    cfg.DEEPSEEK_API_KEY = kwargs.get("DEEPSEEK_API_KEY", "")
    cfg.GROQ_API_KEY = kwargs.get("GROQ_API_KEY", "")
    cfg.OPENAI_API_KEY = kwargs.get("OPENAI_API_KEY", "")
    cfg.GEMINI_API_KEY = kwargs.get("GEMINI_API_KEY", "")
    cfg.MIN_SIGNAL_CONFIDENCE = kwargs.get("MIN_SIGNAL_CONFIDENCE", 0.65)
    cfg.MIN_SIGNAL_CONFIDENCE_PAPER = kwargs.get("MIN_SIGNAL_CONFIDENCE_PAPER", 0.60)
    cfg.PAPER_TRADING = kwargs.get("PAPER_TRADING", False)
    cfg.AI_MODEL = kwargs.get("AI_MODEL", "claude-sonnet-4-6")
    cfg.DEEPSEEK_MODEL = kwargs.get("DEEPSEEK_MODEL", "deepseek-chat")
    cfg.GROQ_MODEL = kwargs.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    cfg.OPENAI_MODEL = kwargs.get("OPENAI_MODEL", "gpt-4o-mini")
    cfg.GEMINI_MODEL = kwargs.get("GEMINI_MODEL", "gemini-1.5-flash")
    return cfg


def _make_strategy(name: str) -> dict:
    return {
        "name": name,
        "description": (
            f"Strategy {name} description that is longer than sixty chars for testing"
        ),
        "risk_level": "medium",
        "market_type": "trending",
    }


# ---------------------------------------------------------------------------
# Фикстура: минимальный анализатор без ключей (паттерн из test_ai_parse_response)
# ---------------------------------------------------------------------------


@pytest.fixture
def no_key_analyzer():
    """Экземпляр AIAnalyzer без API-ключей — enabled=False."""
    cfg = _make_config(AI_PROVIDER="auto")
    with patch("src.ai_analyzer.Config", cfg), patch(
        "src.ai_analyzer.get_all_strategies", return_value=[]
    ):
        from src.ai_analyzer import AIAnalyzer

        return AIAnalyzer()


@pytest.fixture
def anthropic_analyzer():
    """Экземпляр AIAnalyzer только с ключом Anthropic."""
    cfg = _make_config(AI_PROVIDER="anthropic", ANTHROPIC_API_KEY="test_key")
    with patch("src.ai_analyzer.Config", cfg), patch(
        "src.ai_analyzer.get_all_strategies",
        return_value=[_make_strategy("ema_crossover")],
    ), patch.dict("sys.modules", {"anthropic": MagicMock()}):
        from src.ai_analyzer import AIAnalyzer

        # Патчим импорт модуля anthropic внутри __init__
        with patch("anthropic.AsyncAnthropic", MagicMock(return_value=MagicMock())):
            return AIAnalyzer()


# ---------------------------------------------------------------------------
# _resolve_provider()  — определение провайдера
# ---------------------------------------------------------------------------


class TestResolveProvider:
    def _call(self, **kwargs):
        cfg = _make_config(**kwargs)
        with patch("src.ai_analyzer.Config", cfg):
            from src.ai_analyzer import _resolve_provider

            return _resolve_provider()

    def test_resolve_anthropic_with_key(self):
        result = self._call(AI_PROVIDER="anthropic", ANTHROPIC_API_KEY="key123")
        assert result == "anthropic"

    def test_resolve_anthropic_no_key(self):
        result = self._call(AI_PROVIDER="anthropic", ANTHROPIC_API_KEY="")
        assert result == "none"

    def test_resolve_deepseek_with_key(self):
        result = self._call(AI_PROVIDER="deepseek", DEEPSEEK_API_KEY="key123")
        assert result == "deepseek"

    def test_resolve_deepseek_no_key(self):
        result = self._call(AI_PROVIDER="deepseek", DEEPSEEK_API_KEY="")
        assert result == "none"

    def test_resolve_openai_with_key(self):
        result = self._call(AI_PROVIDER="openai", OPENAI_API_KEY="key123")
        assert result == "openai"

    def test_resolve_openai_no_key(self):
        result = self._call(AI_PROVIDER="openai", OPENAI_API_KEY="")
        assert result == "none"

    def test_resolve_auto_prefers_anthropic(self):
        result = self._call(
            AI_PROVIDER="auto",
            ANTHROPIC_API_KEY="anthropic_key",
            DEEPSEEK_API_KEY="deepseek_key",
            OPENAI_API_KEY="openai_key",
        )
        assert result == "anthropic"

    def test_resolve_auto_falls_to_openai(self):
        # Anthropic → OpenAI → DeepSeek → Groq: без Anthropic берём OpenAI
        result = self._call(
            AI_PROVIDER="auto",
            ANTHROPIC_API_KEY="",
            OPENAI_API_KEY="openai_key",
            DEEPSEEK_API_KEY="deepseek_key",
        )
        assert result == "openai"

    def test_resolve_auto_falls_to_deepseek(self):
        # Без Anthropic и OpenAI берём DeepSeek
        result = self._call(
            AI_PROVIDER="auto",
            ANTHROPIC_API_KEY="",
            OPENAI_API_KEY="",
            DEEPSEEK_API_KEY="deepseek_key",
        )
        assert result == "deepseek"

    def test_resolve_auto_falls_to_groq(self):
        # Без всех платных берём Groq (бесплатный)
        result = self._call(
            AI_PROVIDER="auto",
            ANTHROPIC_API_KEY="",
            OPENAI_API_KEY="",
            DEEPSEEK_API_KEY="",
            GROQ_API_KEY="groq_key",
        )
        assert result == "groq"

    def test_resolve_auto_no_keys(self):
        result = self._call(
            AI_PROVIDER="auto",
            ANTHROPIC_API_KEY="",
            DEEPSEEK_API_KEY="",
            OPENAI_API_KEY="",
        )
        assert result == "none"


# ---------------------------------------------------------------------------
# AIAnalyzer.__init__()  — инициализация
# ---------------------------------------------------------------------------


class TestAIAnalyzerInit:
    def _make(self, **kwargs):
        cfg = _make_config(**kwargs)
        mock_openai_module = MagicMock()
        mock_anthropic_module = MagicMock()
        with patch("src.ai_analyzer.Config", cfg), patch(
            "src.ai_analyzer.get_all_strategies", return_value=[]
        ), patch.dict(
            "sys.modules",
            {
                "anthropic": mock_anthropic_module,
                "openai": mock_openai_module,
            },
        ):
            from src.ai_analyzer import AIAnalyzer

            return AIAnalyzer()

    def test_init_no_keys_disabled(self):
        ai = self._make(AI_PROVIDER="auto")
        assert ai.enabled is False
        assert ai._provider_order == []

    def test_init_anthropic_key_only(self):
        ai = self._make(AI_PROVIDER="anthropic", ANTHROPIC_API_KEY="key123")
        assert ai.enabled is True
        assert "anthropic" in ai._provider_order
        assert ai._provider_order[0] == "anthropic"

    def test_init_deepseek_key_only(self):
        ai = self._make(AI_PROVIDER="deepseek", DEEPSEEK_API_KEY="key123")
        assert ai.enabled is True
        assert ai._provider_order[0] == "deepseek"

    def test_init_openai_key_only(self):
        ai = self._make(AI_PROVIDER="openai", OPENAI_API_KEY="key123")
        assert ai.enabled is True
        assert ai._provider_order[0] == "openai"

    def test_init_all_keys_primary_first(self):
        ai = self._make(
            AI_PROVIDER="deepseek",
            ANTHROPIC_API_KEY="a_key",
            DEEPSEEK_API_KEY="d_key",
            OPENAI_API_KEY="o_key",
            GEMINI_API_KEY="g_key",
        )
        assert ai._provider_order[0] == "deepseek"
        assert len(ai._provider_order) == 4


# ---------------------------------------------------------------------------
# AIAnalyzer._build_strategy_list()  — формирование списка стратегий
# ---------------------------------------------------------------------------


class TestBuildStrategyList:
    def test_build_strategy_list_returns_string(self, no_key_analyzer):
        # Override strategies with real data
        no_key_analyzer.strategies = [
            _make_strategy("ema_crossover"),
            _make_strategy("rsi_momentum"),
        ]
        result = no_key_analyzer._build_strategy_list()
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ema_crossover" in result
        assert "rsi_momentum" in result

    def test_build_strategy_list_empty_strategies(self, no_key_analyzer):
        no_key_analyzer.strategies = []
        result = no_key_analyzer._build_strategy_list()
        assert result == ""

    def test_build_strategy_list_includes_risk_and_market(self, no_key_analyzer):
        no_key_analyzer.strategies = [_make_strategy("breakout")]
        result = no_key_analyzer._build_strategy_list()
        assert "risk=" in result
        assert "market=" in result


# ---------------------------------------------------------------------------
# AIAnalyzer._build_prompt()  — построение промпта
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_build_prompt_contains_balance(self, no_key_analyzer):
        no_key_analyzer.strategies = []
        prompt = no_key_analyzer._build_prompt([{"symbol": "BTC/USDT"}], balance=5000.0)
        assert "$5000.00" in prompt

    def test_build_prompt_contains_snapshot_data(self, no_key_analyzer):
        no_key_analyzer.strategies = []
        snap = {"symbol": "ETH/USDT", "price": 3000.0}
        prompt = no_key_analyzer._build_prompt([snap], balance=1000.0)
        assert "ETH/USDT" in prompt

    def test_build_prompt_contains_min_confidence(self, no_key_analyzer):
        no_key_analyzer.strategies = []
        cfg = _make_config(MIN_SIGNAL_CONFIDENCE=0.75)
        with patch("src.ai_analyzer.Config", cfg):
            from src.ai_analyzer import AIAnalyzer

            ai = AIAnalyzer.__new__(AIAnalyzer)
            ai.strategies = []
            ai.logger = logging.getLogger("test")
            prompt = ai._build_prompt([], balance=1000.0)
        assert "0.75" in prompt

    def test_build_prompt_is_string(self, no_key_analyzer):
        no_key_analyzer.strategies = []
        result = no_key_analyzer._build_prompt([], balance=100.0)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# AIAnalyzer._parse_response() — дополнительные случаи, не в test_ai_parse_response
# ---------------------------------------------------------------------------


class TestParseResponseAdditional:
    def test_parse_filters_below_min_confidence(self, no_key_analyzer):
        rec = {
            "symbol": "BTC/USDT",
            "action": "buy",
            "strategy": "ema_crossover",
            "confidence": 0.1,
        }
        result = no_key_analyzer._parse_response(json.dumps([rec]))
        assert result == []

    def test_parse_removes_invalid_action(self, no_key_analyzer):
        rec = {
            "symbol": "BTC/USDT",
            "action": "ignore",
            "strategy": "ema_crossover",
            "confidence": 0.8,
        }
        result = no_key_analyzer._parse_response(json.dumps([rec]))
        assert result == []

    def test_parse_hold_action_kept(self, no_key_analyzer):
        rec = {
            "symbol": "BTC/USDT",
            "action": "hold",
            "strategy": "ema_crossover",
            "confidence": 0.8,
        }
        result = no_key_analyzer._parse_response(json.dumps([rec]))
        assert len(result) == 1
        assert result[0]["action"] == "hold"

    def test_parse_handles_non_list_response(self, no_key_analyzer):
        rec = {
            "symbol": "BTC/USDT",
            "action": "sell",
            "strategy": "rsi_momentum",
            "confidence": 0.9,
        }
        # Одиночный dict вместо списка
        result = no_key_analyzer._parse_response(json.dumps(rec))
        assert len(result) == 1
        assert result[0]["action"] == "sell"


# ---------------------------------------------------------------------------
# AIAnalyzer.analyze()  — основной метод анализа
# ---------------------------------------------------------------------------


class TestAnalyze:
    def _make_disabled_ai(self):
        ai = object.__new__(_get_analyzer_class())
        ai.enabled = False
        ai.logger = logging.getLogger("test")
        ai._provider_order = []
        return ai

    def _make_ai(self, provider_order=None):
        ai = object.__new__(_get_analyzer_class())
        ai.enabled = True
        ai.logger = logging.getLogger("test")
        ai._provider_order = provider_order or []
        ai.strategies = []
        ai._provider_retry_after = {}
        return ai

    @pytest.mark.asyncio
    async def test_analyze_returns_empty_when_disabled(self):
        ai = self._make_disabled_ai()
        result = await ai.analyze([{"symbol": "BTC/USDT"}], 10000)
        assert result == []

    @pytest.mark.asyncio
    async def test_analyze_returns_empty_for_empty_snapshots(self):
        ai = self._make_ai(provider_order=["anthropic"])
        result = await ai.analyze([], 10000)
        assert result == []

    @pytest.mark.asyncio
    async def test_analyze_calls_anthropic_and_returns_recs(self):
        cfg = _make_config(MIN_SIGNAL_CONFIDENCE=0.65)
        with patch("src.ai_analyzer.Config", cfg):
            ai = self._make_ai(provider_order=["anthropic"])
            valid_response = json.dumps(
                [
                    {
                        "symbol": "BTC/USDT",
                        "action": "buy",
                        "strategy": "ema_crossover",
                        "confidence": 0.9,
                        "entry": 50000,
                        "stop_loss": 48000,
                        "take_profit": 55000,
                    }
                ]
            )
            ai._call_anthropic = AsyncMock(return_value=valid_response)
            ai._build_prompt = MagicMock(return_value="prompt")
            result = await ai.analyze([{"symbol": "BTC/USDT"}], 10000)
        assert len(result) == 1
        assert result[0]["action"] == "buy"

    @pytest.mark.asyncio
    async def test_analyze_returns_empty_on_json_error(self):
        ai = self._make_ai(provider_order=["anthropic"])
        ai._call_anthropic = AsyncMock(side_effect=json.JSONDecodeError("err", "", 0))
        ai._build_prompt = MagicMock(return_value="prompt")
        result = await ai.analyze([{"symbol": "BTC/USDT"}], 10000)
        assert result == []

    @pytest.mark.asyncio
    async def test_analyze_falls_back_on_billing_error(self):
        cfg = _make_config(MIN_SIGNAL_CONFIDENCE=0.65)
        with patch("src.ai_analyzer.Config", cfg):
            ai = self._make_ai(provider_order=["anthropic", "deepseek"])
            # First provider raises 429
            billing_error = Exception("Too Many Requests")
            billing_error.status_code = 429
            ai._call_anthropic = AsyncMock(side_effect=billing_error)
            valid_response = json.dumps(
                [
                    {
                        "symbol": "ETH/USDT",
                        "action": "sell",
                        "strategy": "rsi_momentum",
                        "confidence": 0.85,
                    }
                ]
            )
            ai._call_deepseek = AsyncMock(return_value=valid_response)
            ai._build_prompt = MagicMock(return_value="prompt")
            result = await ai.analyze([{"symbol": "ETH/USDT"}], 5000)
        assert len(result) == 1
        assert result[0]["action"] == "sell"

    @pytest.mark.asyncio
    async def test_analyze_returns_empty_when_all_providers_fail(self):
        ai = self._make_ai(provider_order=["anthropic", "deepseek"])
        billing_error_1 = Exception("Ошибка биллинга")
        billing_error_1.status_code = 429
        billing_error_2 = Exception("Квота исчерпана")
        billing_error_2.status_code = 429
        ai._call_anthropic = AsyncMock(side_effect=billing_error_1)
        ai._call_deepseek = AsyncMock(side_effect=billing_error_2)
        ai._build_prompt = MagicMock(return_value="prompt")
        result = await ai.analyze([{"symbol": "BTC/USDT"}], 10000)
        assert result == []

    @pytest.mark.asyncio
    async def test_analyze_returns_empty_on_unexpected_error(self):
        ai = self._make_ai(provider_order=["anthropic"])
        unexpected_error = Exception("Таймаут сети")
        # Нет status_code — обрабатывается как общая ошибка
        ai._call_anthropic = AsyncMock(side_effect=unexpected_error)
        ai._build_prompt = MagicMock(return_value="prompt")
        result = await ai.analyze([{"symbol": "BTC/USDT"}], 10000)
        assert result == []

    @pytest.mark.asyncio
    async def test_analyze_sets_last_ai_provider_on_success(self):
        cfg = _make_config(MIN_SIGNAL_CONFIDENCE=0.65)
        with patch("src.ai_analyzer.Config", cfg):
            ai = self._make_ai(provider_order=["groq"])
            valid_response = json.dumps(
                [
                    {
                        "symbol": "BTC/USDT",
                        "action": "buy",
                        "strategy": "ema_crossover",
                        "confidence": 0.9,
                    }
                ]
            )
            ai._call_groq = AsyncMock(return_value=valid_response)
            ai._build_prompt = MagicMock(return_value="prompt")
            rc = MagicMock()
            rc.get_signal_confidence.return_value = 0.65
            ai._runtime_config = rc
            await ai.analyze([{"symbol": "BTC/USDT"}], 10000)
        rc.set_last_ai_provider.assert_called_once_with("groq")

    @pytest.mark.asyncio
    async def test_analyze_no_runtime_config_does_not_raise(self):
        cfg = _make_config(MIN_SIGNAL_CONFIDENCE=0.65)
        with patch("src.ai_analyzer.Config", cfg):
            ai = self._make_ai(provider_order=["groq"])
            valid_response = json.dumps(
                [
                    {
                        "symbol": "SOL/USDT",
                        "action": "sell",
                        "strategy": "rsi_momentum",
                        "confidence": 0.8,
                    }
                ]
            )
            ai._call_groq = AsyncMock(return_value=valid_response)
            ai._build_prompt = MagicMock(return_value="prompt")
            result = await ai.analyze([{"symbol": "SOL/USDT"}], 10000)
        assert len(result) == 1


def _get_analyzer_class():
    """Импортирует класс AIAnalyzer — кэшируется после первого вызова."""
    with patch("src.ai_analyzer.Config", _make_config()), patch(
        "src.ai_analyzer.get_all_strategies", return_value=[]
    ):
        from src.ai_analyzer import AIAnalyzer

        return AIAnalyzer


# ---------------------------------------------------------------------------
# AIAnalyzer.recommend_strategy_local()  — локальная рекомендация стратегии
# ---------------------------------------------------------------------------


class TestRecommendStrategyLocal:
    @pytest.fixture(autouse=True)
    def analyzer(self, no_key_analyzer):
        self.ai = no_key_analyzer

    def _snap(self, **kwargs):
        """Строит минимальный снапшот с заданными переопределениями индикаторов."""
        indicators = {
            "rsi": kwargs.pop("rsi", 50),
            "bb_width": kwargs.pop("bb_width", 0.04),
            "trend": kwargs.pop("trend", "sideways"),
            "macd": kwargs.pop("macd", "neutral"),
            "bb_position": kwargs.pop("bb_position", "middle"),
        }
        return {
            "indicators": indicators,
            "volume_ratio": kwargs.pop("volume_ratio", 1.0),
            **kwargs,
        }

    def test_local_breakout(self):
        snap = self._snap(bb_width=0.09, volume_ratio=2.5)
        strategy, conf = self.ai.recommend_strategy_local(snap)
        assert strategy == "breakout"
        assert conf == 0.72

    def test_local_bb_near_lower(self):
        snap = self._snap(bb_position="near_lower", rsi=30)
        strategy, conf = self.ai.recommend_strategy_local(snap)
        assert strategy == "bollinger_bands"
        assert conf == 0.75

    def test_local_bb_near_upper(self):
        snap = self._snap(bb_position="near_upper", rsi=70)
        strategy, conf = self.ai.recommend_strategy_local(snap)
        assert strategy == "bollinger_bands"
        assert conf == 0.75

    def test_local_uptrend_bullish_low_rsi(self):
        snap = self._snap(trend="uptrend", macd="bullish", rsi=55)
        strategy, conf = self.ai.recommend_strategy_local(snap)
        assert strategy == "swing_trading"
        assert conf == 0.70

    def test_local_uptrend_bullish_high_rsi(self):
        snap = self._snap(trend="uptrend", macd="bullish", rsi=65)
        strategy, conf = self.ai.recommend_strategy_local(snap)
        assert strategy == "trend_following"
        assert conf == 0.70

    def test_local_downtrend_bearish_high_rsi(self):
        snap = self._snap(trend="downtrend", macd="bearish", rsi=45)
        strategy, conf = self.ai.recommend_strategy_local(snap)
        assert strategy == "swing_trading"
        assert conf == 0.70

    def test_local_downtrend_bearish_low_rsi(self):
        snap = self._snap(trend="downtrend", macd="bearish", rsi=35)
        strategy, conf = self.ai.recommend_strategy_local(snap)
        assert strategy == "trend_following"
        assert conf == 0.70

    def test_local_rsi_oversold(self):
        snap = self._snap(rsi=25)
        strategy, conf = self.ai.recommend_strategy_local(snap)
        assert strategy == "rsi_momentum"
        assert conf == 0.78

    def test_local_rsi_overbought(self):
        snap = self._snap(rsi=75)
        strategy, conf = self.ai.recommend_strategy_local(snap)
        assert strategy == "rsi_momentum"
        assert conf == 0.78

    def test_local_default(self):
        # sideways + rsi=50 → mean_reversion (боковик с нейтральным RSI)
        snap = self._snap()
        strategy, conf = self.ai.recommend_strategy_local(snap)
        assert strategy == "mean_reversion"
        assert conf == 0.65

    def test_local_missing_indicators(self):
        # Empty snapshot → defaults: trend=sideways, rsi=50 → mean_reversion
        strategy, conf = self.ai.recommend_strategy_local({})
        assert strategy == "mean_reversion"
        assert conf == 0.65

    def test_local_ema_crossover_pure_sideways(self):
        # sideways + RSI вне диапазона 38-62 → ema_crossover
        snap = self._snap(trend="sideways", rsi=35)
        strategy, conf = self.ai.recommend_strategy_local(snap)
        assert strategy == "ema_crossover"
        assert conf == 0.60
