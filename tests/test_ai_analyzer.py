"""Тесты AIAnalyzer и вспомогательных функций."""

import json
import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.strategies import STRATEGY_REGISTRY

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
    cfg.OPENROUTER_API_KEY = kwargs.get("OPENROUTER_API_KEY", "")
    cfg.NVIDIA_API_KEY = kwargs.get("NVIDIA_API_KEY", "")
    cfg.MIN_SIGNAL_CONFIDENCE = kwargs.get("MIN_SIGNAL_CONFIDENCE", 0.65)
    cfg.MIN_SIGNAL_CONFIDENCE_PAPER = kwargs.get("MIN_SIGNAL_CONFIDENCE_PAPER", 0.60)
    cfg.PAPER_TRADING = kwargs.get("PAPER_TRADING", False)
    cfg.AI_MODEL = kwargs.get("AI_MODEL", "claude-sonnet-4-6")
    cfg.DEEPSEEK_MODEL = kwargs.get("DEEPSEEK_MODEL", "deepseek-chat")
    cfg.GROQ_MODEL = kwargs.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    cfg.OPENAI_MODEL = kwargs.get("OPENAI_MODEL", "gpt-4o-mini")
    cfg.GEMINI_MODEL = kwargs.get("GEMINI_MODEL", "gemini-3.6-flash")
    cfg.OPENROUTER_MODEL = kwargs.get(
        "OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"
    )
    cfg.NVIDIA_MODEL = kwargs.get("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
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


@pytest.fixture(autouse=True)
def _no_ai_keys_in_env(monkeypatch):
    """Ключи из окружения разработчика не должны попадать в цепочку тестов."""
    from src import ai_providers

    for p in ai_providers.PROVIDERS:
        for env in (p.key_env, p.model_env, p.url_env):
            if env:
                monkeypatch.delenv(env, raising=False)


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
# Цепочка провайдеров из ключей и настроек (каталог src/ai_providers.py)
# ---------------------------------------------------------------------------


class TestChainFromKeys:
    def _make(self, rc=None, **kwargs):
        cfg = _make_config(**kwargs)
        with patch("src.ai_analyzer.Config", cfg), patch(
            "src.ai_analyzer.get_all_strategies", return_value=[]
        ):
            from src.ai_analyzer import AIAnalyzer

            return AIAnalyzer(runtime_config=rc)

    def _rc(self, chain=None, pinned="auto"):
        rc = MagicMock()
        rc.get_ai_chain.return_value = chain or {}
        rc.get_ai_provider.return_value = pinned
        return rc

    def test_no_keys_disabled(self):
        ai = self._make()
        assert ai.enabled is False
        assert ai._provider_order == []

    def test_free_by_default_in_catalog_order(self):
        ai = self._make(GEMINI_API_KEY="g", GROQ_API_KEY="q", OPENROUTER_API_KEY="o")
        assert ai._provider_order == ["groq", "gemini", "openrouter"]

    def test_paid_key_alone_is_not_used_in_free_mode(self):
        ai = self._make(DEEPSEEK_API_KEY="d", ANTHROPIC_API_KEY="a")
        assert ai.enabled is False

    def test_pinned_paid_provider_goes_first_even_in_free_mode(self):
        ai = self._make(
            rc=self._rc(pinned="deepseek"), DEEPSEEK_API_KEY="d", GROQ_API_KEY="q"
        )
        assert ai._provider_order == ["deepseek", "groq"]

    def test_auto_mode_puts_paid_after_free(self):
        ai = self._make(
            rc=self._rc({"mode": "auto"}),
            ANTHROPIC_API_KEY="a",
            DEEPSEEK_API_KEY="d",
            GEMINI_API_KEY="g",
        )
        assert ai._provider_order == ["gemini", "deepseek", "anthropic"]

    def test_user_order_and_disabled(self):
        rc = self._rc({"order": ["gemini", "groq"], "disabled": ["openrouter"]})
        ai = self._make(
            rc=rc, GEMINI_API_KEY="g", GROQ_API_KEY="q", OPENROUTER_API_KEY="o"
        )
        assert ai._provider_order == ["gemini", "groq"]

    def test_keyless_provider_only_when_switched_on(self):
        assert self._make().enabled is False
        ai = self._make(rc=self._rc({"keyless": ["llm7"]}))
        assert ai._provider_order == ["llm7"]

    def test_model_from_env_then_app_choice(self):
        cfg = _make_config(GEMINI_API_KEY="g", GEMINI_MODEL="gemini-2.5-flash")
        with patch("src.ai_analyzer.Config", cfg), patch(
            "src.ai_analyzer.get_all_strategies", return_value=[]
        ):
            from src.ai_analyzer import AIAnalyzer

            ai = AIAnalyzer()
            assert ai._chain() == [("gemini", "gemini-2.5-flash")]
            ai._runtime_config = self._rc({"models": {"gemini": "gemini-3.6-flash"}})
            assert ai._chain() == [("gemini", "gemini-3.6-flash")]

    def test_cloudflare_needs_account_id(self, monkeypatch):
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "t" * 20)
        assert self._make().enabled is False
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a" * 32)
        assert self._make()._provider_order == ["cloudflare"]


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

    def test_build_strategy_list_excludes_disabled(self, no_key_analyzer):
        no_key_analyzer.strategies = [
            _make_strategy("ema_crossover"),
            _make_strategy("rsi_momentum"),
        ]
        no_key_analyzer._runtime_config = MagicMock()
        no_key_analyzer._runtime_config.get_disabled_strategies.return_value = {
            "rsi_momentum"
        }
        result = no_key_analyzer._build_strategy_list()
        assert "ema_crossover" in result
        assert "rsi_momentum" not in result


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


_REC = json.dumps(
    [
        {
            "symbol": "BTC/USDT",
            "action": "buy",
            "strategy": "ema_crossover",
            "confidence": 0.9,
        }
    ]
)


def _http_error(code, text="error"):
    e = Exception(text)
    e.status_code = code
    return e


class TestAnalyze:
    def _make_ai(self, chain):
        ai = object.__new__(_get_analyzer_class())
        ai.logger = logging.getLogger("test")
        ai.strategies = []
        ai._provider_retry_after = {}
        ai._notified_at = {}
        ai._NOTIFY_COOLDOWN = 3600.0
        ai._telegram = None
        ai._emit = MagicMock()
        ai._runtime_config = MagicMock()
        ai._runtime_config.get_signal_confidence.return_value = 0.65
        ai._chain = MagicMock(return_value=chain)
        ai._build_prompt = MagicMock(return_value="prompt")
        return ai

    async def _run(self, ai, responses):
        """responses: провайдер → текст ответа или исключение."""

        async def call(provider, model, prompt):
            r = responses[provider]
            if isinstance(r, Exception):
                raise r
            return r

        ai._call = AsyncMock(side_effect=call)
        cfg = _make_config(MIN_SIGNAL_CONFIDENCE=0.65)
        with patch("src.ai_analyzer.Config", cfg):
            return await ai.analyze([{"symbol": "BTC/USDT"}], 10000)

    @pytest.mark.asyncio
    async def test_empty_without_chain_or_snapshots(self):
        ai = self._make_ai([])
        assert await ai.analyze([{"symbol": "BTC/USDT"}], 10000) == []
        ai = self._make_ai([("groq", "m")])
        assert await ai.analyze([], 10000) == []

    @pytest.mark.asyncio
    async def test_success_marks_provider_and_emits_status(self):
        ai = self._make_ai([("groq", "openai/gpt-oss-120b")])
        result = await self._run(ai, {"groq": _REC})
        assert result[0]["action"] == "buy"
        ai._runtime_config.set_last_ai_provider.assert_called_once_with("groq")
        ai._runtime_config.record_ai_call.assert_called_once_with("groq")
        type_, data = ai._emit.call_args.args
        assert type_ == "ai.status" and data["status"] == "ok"
        assert data["model"] == "openai/gpt-oss-120b"

    @pytest.mark.asyncio
    async def test_json_error_goes_to_next_provider(self):
        ai = self._make_ai([("groq", "m"), ("gemini", "m")])
        result = await self._run(
            ai, {"groq": json.JSONDecodeError("bad", "", 0), "gemini": _REC}
        )
        assert len(result) == 1
        ai._runtime_config.set_provider_status.assert_any_call("groq", "bad_format")

    @pytest.mark.asyncio
    async def test_rate_limit_pauses_provider_for_seven_minutes(self):
        ai = self._make_ai(
            [("gemini", "m"), ("openrouter", "m"), ("llm7", "m"), ("nvidia", "m")]
        )
        result = await self._run(
            ai,
            {
                "gemini": _http_error(429, "Too Many Requests"),
                "openrouter": _http_error(429, "Rate limit, retry in 30s"),
                "llm7": _http_error(429, "Too many requests. Retry after 10 seconds."),
                "nvidia": _REC,
            },
        )
        assert len(result) == 1
        now = time.time()
        assert 400 < ai._provider_retry_after["gemini"] - now <= 420
        assert 30 < ai._provider_retry_after["openrouter"] - now <= 35
        assert 10 < ai._provider_retry_after["llm7"] - now <= 15

    @pytest.mark.asyncio
    async def test_no_balance_bad_key_and_missing_model(self):
        ai = self._make_ai([("deepseek", "m"), ("groq", "m"), ("gemini", "x")])
        result = await self._run(
            ai,
            {
                "deepseek": _http_error(402),
                "groq": _http_error(401),
                "gemini": _http_error(404),
            },
        )
        assert result == []
        rc = ai._runtime_config
        rc.set_provider_status.assert_any_call("deepseek", "no_balance")
        rc.set_provider_status.assert_any_call("groq", "bad_key")
        rc.set_provider_status.assert_any_call("gemini", "bad_model")
        assert ai._provider_retry_after["deepseek"] - time.time() > 5 * 3600
        rc.set_last_ai_provider.assert_called_with("local")

    @pytest.mark.asyncio
    async def test_missing_custom_model_falls_back_to_recommended(self):
        ai = self._make_ai([("gemini", "gemini-2.0-flash"), ("groq", "m")])
        seen = []

        async def call(provider, model, prompt):
            seen.append((provider, model))
            if model == "gemini-2.0-flash":
                raise _http_error(404, "model not found")
            return _REC

        ai._call = AsyncMock(side_effect=call)
        with patch("src.ai_analyzer.Config", _make_config()):
            result = await ai.analyze([{"symbol": "BTC/USDT"}], 10000)
        assert len(result) == 1
        assert seen == [
            ("gemini", "gemini-2.0-flash"),
            ("gemini", "gemini-flash-latest"),
        ]
        assert "gemini" not in ai._provider_retry_after

    @pytest.mark.asyncio
    async def test_bad_request_does_not_stop_the_chain(self):
        # Раньше 400 обрывал цикл; с десятком провайдеров один капризный
        # (не принял параметры запроса) не должен оставлять бота без AI.
        ai = self._make_ai([("cohere", "m"), ("groq", "m")])
        result = await self._run(ai, {"cohere": _http_error(400), "groq": _REC})
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_provider_on_pause_is_skipped(self):
        ai = self._make_ai([("groq", "m"), ("gemini", "m")])
        ai._provider_retry_after["groq"] = time.time() + 60
        result = await self._run(ai, {"groq": _REC, "gemini": _REC})
        assert len(result) == 1
        assert [c.args[0] for c in ai._call.await_args_list] == ["gemini"]

    @pytest.mark.asyncio
    async def test_network_error_has_no_pause(self):
        ai = self._make_ai([("groq", "m")])
        assert await self._run(ai, {"groq": Exception("timeout")}) == []
        assert "groq" not in ai._provider_retry_after


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

    @staticmethod
    def _fake_registry(signals: dict) -> dict:
        """Строит {name: class} реестр с захардкоженным generate_signal()."""
        registry = {}
        for name, sig in signals.items():
            registry[name] = type(
                "Fake",
                (),
                {"generate_signal": lambda self, data, _sig=sig: dict(_sig)},
            )
        return registry

    def _patch_registry(self, registry: dict):
        return (
            patch("src.ai_analyzer.STRATEGY_REGISTRY", registry),
            patch(
                "src.ai_analyzer.create_strategy",
                side_effect=lambda name: registry[name](),
            ),
        )

    def test_picks_highest_confidence_actionable_signal(self):
        # "c" holds highest confidence overall, but action=hold — must be
        # ignored in favour of the strongest actual buy/sell.
        registry = self._fake_registry(
            {
                "a": {"action": "buy", "confidence": 0.60},
                "b": {"action": "sell", "confidence": 0.85},
                "c": {"action": "hold", "confidence": 0.99},
            }
        )
        p1, p2 = self._patch_registry(registry)
        with p1, p2:
            strat, conf, sig = self.ai.recommend_strategy_local(pd.DataFrame())
        assert strat == "b"
        assert conf == 0.85
        assert sig["action"] == "sell"

    def test_all_hold_returns_hold(self):
        registry = self._fake_registry({"a": {"action": "hold", "confidence": 0.99}})
        p1, p2 = self._patch_registry(registry)
        with p1, p2:
            strat, conf, sig = self.ai.recommend_strategy_local(pd.DataFrame())
        assert (strat, conf, sig) == ("hold", 0.0, {})

    def test_disabled_strategy_via_telegram_is_skipped(self):
        registry = self._fake_registry(
            {
                "a": {"action": "buy", "confidence": 0.90},
                "b": {"action": "buy", "confidence": 0.50},
            }
        )
        self.ai._runtime_config = MagicMock()
        self.ai._runtime_config.get_disabled_strategies.return_value = {"a"}
        p1, p2 = self._patch_registry(registry)
        with p1, p2:
            strat, conf, _ = self.ai.recommend_strategy_local(pd.DataFrame())
        assert strat == "b"
        assert conf == 0.50

    def test_exclude_param_removes_candidate(self):
        registry = self._fake_registry(
            {
                "a": {"action": "buy", "confidence": 0.90},
                "b": {"action": "buy", "confidence": 0.50},
            }
        )
        p1, p2 = self._patch_registry(registry)
        with p1, p2:
            strat, conf, _ = self.ai.recommend_strategy_local(
                pd.DataFrame(), exclude={"a"}
            )
        assert strat == "b"
        assert conf == 0.50

    def test_no_runtime_config_still_works(self):
        # _runtime_config=None (дефолт) — не должно падать при отсутствии RC.
        registry = self._fake_registry({"a": {"action": "buy", "confidence": 0.70}})
        p1, p2 = self._patch_registry(registry)
        with p1, p2:
            strat, conf, _ = self.ai.recommend_strategy_local(pd.DataFrame())
        assert strat == "a"
        assert conf == 0.70

    def test_real_registry_returns_structurally_valid_result(self):
        # Интеграционный смоук-тест на настоящем STRATEGY_REGISTRY — все 16+
        # стратегии (включая ранее недостижимые macd_crossover/scalping/
        # мем-профиль) должны быть реально вызываемы без исключений.
        from src.indicators import calculate_technical_indicators
        from tests.conftest import make_ohlcv

        df = calculate_technical_indicators(make_ohlcv(n=100, trend=0.002))
        strat, conf, sig = self.ai.recommend_strategy_local(df)
        assert strat == "hold" or strat in STRATEGY_REGISTRY
        assert 0.0 <= conf <= 1.0
        if strat != "hold":
            assert sig["action"] in ("buy", "sell")


# ---------------------------------------------------------------------------
# _record_usage() / refresh_deepseek_balance()
# ---------------------------------------------------------------------------


class TestRecordUsage:
    def test_noop_when_no_tokens(self, no_key_analyzer):
        no_key_analyzer._runtime_config = MagicMock()
        no_key_analyzer._record_usage("groq", None)
        no_key_analyzer._runtime_config.record_token_usage.assert_not_called()

    def test_noop_when_zero_tokens(self, no_key_analyzer):
        no_key_analyzer._runtime_config = MagicMock()
        no_key_analyzer._record_usage("groq", 0)
        no_key_analyzer._runtime_config.record_token_usage.assert_not_called()

    def test_noop_when_no_runtime_config(self, no_key_analyzer):
        no_key_analyzer._runtime_config = None
        no_key_analyzer._record_usage("groq", 500)  # must not raise

    def test_records_when_tokens_present(self, no_key_analyzer):
        no_key_analyzer._runtime_config = MagicMock()
        no_key_analyzer._record_usage("groq", 500)
        no_key_analyzer._runtime_config.record_token_usage.assert_called_once_with(
            "groq", 500
        )

    def test_swallows_exception_from_runtime_config(self, no_key_analyzer):
        no_key_analyzer._runtime_config = MagicMock()
        no_key_analyzer._runtime_config.record_token_usage.side_effect = RuntimeError(
            "boom"
        )
        no_key_analyzer._record_usage("groq", 500)  # must not raise


def _deepseek_key(value="dk"):
    return patch(
        "src.ai_analyzer._env",
        side_effect=lambda n: value if n == "DEEPSEEK_API_KEY" else "",
    )


def _aiohttp_session(status, payload=None):
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(return_value=payload or {})
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.get = MagicMock(return_value=resp)
    return session


class TestRefreshDeepseekBalance:
    @pytest.mark.asyncio
    async def test_noop_without_deepseek_key(self, no_key_analyzer):
        no_key_analyzer._runtime_config = MagicMock()
        with _deepseek_key(""):
            await no_key_analyzer.refresh_deepseek_balance()
        no_key_analyzer._runtime_config.set_deepseek_balance.assert_not_called()

    @pytest.mark.asyncio
    async def test_noop_without_runtime_config(self, no_key_analyzer):
        no_key_analyzer._runtime_config = None
        with _deepseek_key():
            await no_key_analyzer.refresh_deepseek_balance()  # must not raise

    @pytest.mark.asyncio
    async def test_caches_usd_balance_on_success(self, no_key_analyzer):
        no_key_analyzer._runtime_config = MagicMock()
        payload = {
            "is_available": True,
            "balance_infos": [{"currency": "USD", "total_balance": "12.34"}],
        }
        with _deepseek_key(), patch(
            "aiohttp.ClientSession", return_value=_aiohttp_session(200, payload)
        ):
            await no_key_analyzer.refresh_deepseek_balance()
        no_key_analyzer._runtime_config.set_deepseek_balance.assert_called_once_with(
            12.34
        )

    @pytest.mark.asyncio
    async def test_skips_caching_on_non_200(self, no_key_analyzer):
        no_key_analyzer._runtime_config = MagicMock()
        with _deepseek_key(), patch(
            "aiohttp.ClientSession", return_value=_aiohttp_session(401)
        ):
            await no_key_analyzer.refresh_deepseek_balance()
        no_key_analyzer._runtime_config.set_deepseek_balance.assert_not_called()

    @pytest.mark.asyncio
    async def test_swallows_network_exception(self, no_key_analyzer):
        no_key_analyzer._runtime_config = MagicMock()
        with _deepseek_key(), patch(
            "aiohttp.ClientSession", side_effect=ConnectionError("boom")
        ):
            await no_key_analyzer.refresh_deepseek_balance()  # must not raise
        no_key_analyzer._runtime_config.set_deepseek_balance.assert_not_called()
