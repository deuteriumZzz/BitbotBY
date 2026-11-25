"""Запрос v2: вероятности вместо сделок, без баланса и порога; прогнозы — в журнал."""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bitbot.ai.analyzer import FORECAST_HORIZON, PROMPT_VERSION_V2, AIAnalyzer
from tests.ai.test_ai_analyzer import _make_config


def _ai(**cfg):
    ai = AIAnalyzer.__new__(AIAnalyzer)
    ai.strategies = []
    ai.logger = logging.getLogger("test")
    ai._runtime_config = None
    return ai, _make_config(
        AI_FORECAST_PROMPT=True, TIMEFRAME="15m", MARKET_TYPE="linear", **cfg
    )


def test_prompt_v2_hides_balance_and_threshold_and_asks_case_against():
    ai, cfg = _ai(MIN_SIGNAL_CONFIDENCE=0.77)
    with patch("bitbot.ai.analyzer.Config", cfg):
        prompt = ai._build_prompt(
            [{"symbol": "SOL/USDT", "price": 150.0, "atr": 3.0}], 13373.0
        )
    assert (
        "13373" not in prompt and "BALANCE" not in prompt
    )  # баланс провайдеру не уходит
    assert "0.77" not in prompt  # порог модель не знает — не подгоняет под него
    assert "boosts buy" not in prompt  # без «правил учебника»
    assert "case_against" in prompt and f"next {FORECAST_HORIZON} candles" in prompt
    assert '"trades":[]' in prompt  # пустой список сделок — нормальный ответ


def test_parse_v2_keeps_forecasts_and_turns_p_win_into_confidence():
    ai, cfg = _ai()
    raw = json.dumps(
        {
            "forecasts": [
                {"symbol": "SOL/USDT", "case_against": "x", "p_up": 0.55},
                {"symbol": "BAD", "p_up": 2},  # вне 0..1 — выбрасывается
                {"symbol": "ETH/USDT", "p_up": True},  # не число
            ],
            "trades": [
                {
                    "symbol": "SOL/USDT",
                    "action": "buy",
                    "strategy": "ema_crossover",
                    "case_against": "y",
                    "p_win": 0.7,
                    "entry": 150,
                    "stop_loss": 145,
                    "take_profit": 160,
                },
                {
                    "symbol": "ETH/USDT",
                    "action": "buy",
                    "strategy": "x",
                    "p_win": 0.3,
                },  # ниже порога
            ],
        }
    )
    with patch("bitbot.ai.analyzer.Config", cfg):
        recs = ai._parse_response(raw)
    assert ai._last_forecasts == [{"symbol": "SOL/USDT", "p_up": 0.55}]
    assert [(r["symbol"], r["confidence"]) for r in recs] == [("SOL/USDT", 0.7)]


@pytest.mark.asyncio
async def test_temperature_low_and_retried_without_when_rejected():
    ai, cfg = _ai()
    create = AsyncMock(
        side_effect=[
            Exception("Unsupported parameter: 'temperature'"),
            MagicMock(usage=None, choices=[MagicMock(message=MagicMock(content="[]"))]),
        ]
    )
    client = MagicMock()
    client.chat.completions.create = create
    ai._client = lambda p: client
    ai._record_usage = lambda *a: None
    with (
        patch("bitbot.ai.analyzer.Config", cfg),
        patch("bitbot.ai.analyzer.catalog.BY_ID", {"groq": MagicMock(kind="openai")}),
    ):
        assert await ai._call("groq", "m", "q") == "[]"
    assert create.await_args_list[0].kwargs["temperature"] == 0.1
    assert "temperature" not in create.await_args_list[1].kwargs


def test_forecast_written_to_ledger():
    ai, cfg = _ai()
    written = []
    ai._ledger = MagicMock(write=lambda kind, **f: written.append((kind, f)))
    ai._last_forecasts = [
        {"symbol": "SOL/USDT", "p_up": 0.6},
        {"symbol": "ZZZ", "p_up": 0.5},
    ]
    snaps = [{"symbol": "SOL/USDT", "price": 150.0, "atr": 3.0}]
    trades = [
        {
            "symbol": "SOL/USDT",
            "action": "buy",
            "entry": 150,
            "stop_loss": 145,
            "take_profit": 160,
            "confidence": 0.7,
        }
    ]
    with patch("bitbot.ai.analyzer.Config", cfg):
        ai._write_forecast("groq", "m", snaps, trades)
    kind, f = written[0]
    assert kind == "ai_forecast" and f["prompt_version"] == PROMPT_VERSION_V2
    assert f["items"] == [
        {"symbol": "SOL/USDT", "price": 150.0, "atr": 3.0, "p_up": 0.6}
    ]
    assert f["trades"][0]["p_win"] == 0.7 and f["horizon"] == FORECAST_HORIZON


@pytest.mark.asyncio
async def test_untrusted_provider_cannot_open_trade_but_forecasts_still_logged(
    tmp_path,
):
    from bitbot.storage.decision_ledger import DecisionLedger
    from tests.core.test_decision_ledger import TestAnalyze, _lines

    reply = json.dumps(
        {
            "forecasts": [{"symbol": "BTC/USDT", "p_up": 0.6}],
            "trades": [
                {"symbol": "BTC/USDT", "action": "buy", "strategy": "s", "p_win": 0.9}
            ],
        }
    )
    ai = TestAnalyze()._make_ai([("groq", "m1")])
    ai._ledger = DecisionLedger(path=str(tmp_path / "l.jsonl"), enabled=True)
    ai._call = AsyncMock(return_value=reply)
    cfg = _make_config(
        AI_FORECAST_PROMPT=True, AI_TRUST_SCOREBOARD=True, TIMEFRAME="15m"
    )
    cfg.AI_STRICT_VALIDATION = False
    snaps = [{"symbol": "BTC/USDT", "price": 100.0, "atr": 2.0}]
    with patch("bitbot.ai.analyzer.Config", cfg):
        ai._untrusted = lambda: []
        assert len(await ai.analyze(snaps, 1000)) == 1  # доверенный — сделка проходит
        ai._untrusted = lambda: ["groq"]
        assert await ai.analyze(snaps, 1000) == []  # не лучше монетки — сделки нет
    kinds = [r["kind"] for r in _lines(tmp_path / "l.jsonl")]
    assert kinds.count("ai_forecast") == 2 and kinds[-1] == "ai_untrusted"
