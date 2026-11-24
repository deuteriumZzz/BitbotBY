"""Токены в записи ai_call: откуда берутся, как попадают в журнал, как считаются в сводке."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.ai import ai_providers
from src.ai.ai_analyzer import AIAnalyzer
from src.ai_infra.health import estimate_cost, provider_health, tokens_by_model
from src.ops.daily_digest import build_digest, format_digest

OPENAI_KIND = next(p.id for p in ai_providers.PROVIDERS if p.kind != "anthropic")
ANTHROPIC = next(p.id for p in ai_providers.PROVIDERS if p.kind == "anthropic")
REC = json.dumps(
    [
        {
            "symbol": "BTC/USDT",
            "action": "buy",
            "strategy": "rsi",
            "confidence": 0.95,
            "entry": 100.0,
            "stop_loss": 95.0,
            "take_profit": 110.0,
        }
    ]
)
SNAP = {
    "symbol": "BTC/USDT",
    "price": 100.0,
    "atr": 2.0,
    "changes": {},
    "volume_ratio": 1.0,
    "indicators": {},
    "levels": {},
    "news_sentiment": 0.0,
    "social_sentiment": None,
    "top_headlines": [],
}


@pytest.fixture(autouse=True)
def _no_ai_keys(monkeypatch):
    for p in ai_providers.PROVIDERS:
        for env in (p.key_env, p.model_env, p.url_env):
            if env:
                monkeypatch.delenv(env, raising=False)


def analyzer_with(client):
    a = AIAnalyzer()
    a._client = lambda provider: client
    return a


async def test_openai_style_usage_is_captured():
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=resp))
        )
    )
    a = analyzer_with(client)
    assert await a._call(OPENAI_KIND, "m", "p") == "[]"
    assert a._last_usage == {"input": 10, "output": 5, "total": 15}


async def test_anthropic_style_usage_is_captured():
    msg = SimpleNamespace(
        content=[SimpleNamespace(text="[]")],
        usage=SimpleNamespace(input_tokens=7, output_tokens=3),
    )
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=msg))
    )
    a = analyzer_with(client)
    await a._call(ANTHROPIC, "m", "p")
    assert a._last_usage == {"input": 7, "output": 3, "total": 10}


async def test_missing_usage_leaves_none_not_zero():
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))], usage=None
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=resp))
        )
    )
    a = analyzer_with(client)
    a._last_usage = None
    await a._call(OPENAI_KIND, "m", "p")
    assert a._last_usage is None


async def test_analyze_writes_tokens_per_call_and_resets_between_calls():
    a = AIAnalyzer()
    a._chain = lambda: [(OPENAI_KIND, "m")]
    a._ledger_write = MagicMock()
    usages = [{"input": 100, "output": 20, "total": 120}, None]

    async def fake_call(provider, model, prompt):
        a._last_usage = usages.pop(0)
        return REC

    a._call = fake_call
    await a.analyze([SNAP], 1000.0)
    await a.analyze(
        [SNAP], 1000.0
    )  # второй ответ без usage — токены предыдущего вызова не переносятся
    calls = [
        c
        for c in a._ledger_write.call_args_list
        if c.args[0] == "ai_call" and c.kwargs.get("status") == "ok"
    ]
    assert calls[0].kwargs["tokens"] == {"input": 100, "output": 20, "total": 120}
    assert calls[1].kwargs["tokens"] is None


def ok_call(provider, model, total, ts="t"):
    rec = {
        "kind": "ai_call",
        "status": "ok",
        "provider": provider,
        "model": model,
        "latency_ms": 50,
        "ts": ts,
        "attempts": [],
    }
    if total is not None:
        rec["tokens"] = {"input": total - 10, "output": 10, "total": total}
    return rec


def test_health_sums_tokens_by_provider_and_model_and_skips_unknown():
    recs = [
        ok_call("p", "m1", 100),
        ok_call("p", "m1", 50),
        ok_call("p", "m2", None),
        ok_call("q", "m3", 10),
        {
            "kind": "ai_call",
            "status": "exhausted",
            "provider": None,
            "tokens": {"total": 999},
            "attempts": [],
        },
    ]
    health = provider_health(recs)
    assert health["p"]["tokens"] == 150 and health["q"]["tokens"] == 10
    by_model = tokens_by_model(recs)
    assert by_model == {
        "m1": 150,
        "m3": 10,
    }  # без токенов и неуспешные вызовы не учитываются
    cost = estimate_cost(by_model, {"m1": 0.5})
    assert cost == {"m1": 0.000075, "m3": None}  # модель без цены — None, не 0


def test_digest_line_shows_tokens():
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    recs = [{**ok_call("p", "m", 1234, ts=now.isoformat()), "trace_id": "a"}]
    assert "токенов 1234" in format_digest(build_digest(recs, 24, now))
