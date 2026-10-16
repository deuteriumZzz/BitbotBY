"""Каталог AI-провайдеров и правила цепочки (src/ai_providers.py)."""

import pytest

from src import ai_providers as ap


def test_catalog_is_consistent():
    ids = [p.id for p in ap.PROVIDERS]
    assert len(ids) == len(set(ids))
    for p in ap.PROVIDERS:
        rec = p.model(p.default_model)
        # Рекомендованная модель решает за бота — только крупная.
        assert rec is not None and rec.strong, p.id
        assert p.kind == "anthropic" or p.base_url, p.id
        # Платный провайдер не может предлагать бесплатную модель по умолчанию.
        assert p.free_tier or not rec.free, p.id
    assert [p.id for p in ap.PROVIDERS if not p.free_tier] == [
        "deepseek",
        "openai",
        "anthropic",
    ]


def test_free_flag_for_custom_models():
    orouter = ap.BY_ID["openrouter"]
    assert orouter.is_free("some/new-model:free")
    assert not orouter.is_free("some/new-model")
    assert ap.BY_ID["groq"].is_free("any-new-groq-model")
    assert not ap.BY_ID["openai"].is_free("gpt-9")


def test_url_with_account_and_override():
    cf = ap.BY_ID["cloudflare"]
    assert cf.url(lambda env: "") == ""
    assert cf.url(lambda env: "abc") == (
        "https://api.cloudflare.com/client/v4/accounts/abc/ai/v1"
    )
    ollama = ap.BY_ID["ollama"]
    assert ollama.url(lambda env: "") == "http://localhost:11434/v1"
    assert ollama.url(lambda env: "http://gpu-box:11434/v1/") == (
        "http://gpu-box:11434/v1"
    )


def test_clean_chain_validates():
    assert ap.clean_chain({"mode": "auto", "order": ["groq", "groq", "gemini"]}) == {
        "mode": "auto",
        "order": ["groq", "gemini"],
    }
    for bad in (
        {"mode": "cheap"},
        {"order": ["nope"]},
        {"keyless": ["groq"]},  # без ключа работают только LLM7 и Ollama
        {"models": {"groq": ""}},
        [],
    ):
        with pytest.raises(ValueError):
            ap.clean_chain(bad)


def test_chain_modes():
    keys = ["groq", "deepseek", "gemini", "anthropic"]
    assert [p for p, _ in ap.chain({}, keys)] == ["groq", "gemini"]
    assert [p for p, _ in ap.chain({"mode": "paid"}, keys)] == [
        "deepseek",
        "anthropic",
    ]
    auto = ap.chain({"mode": "auto", "order": ["deepseek", "gemini"]}, keys)
    assert [p for p, _ in auto] == ["gemini", "groq", "deepseek", "anthropic"]
    # Платная модель у бесплатного провайдера в режиме «только бесплатные» — мимо.
    paid_model = {"models": {"gemini": "gemini-pro-latest"}}
    assert [p for p, _ in ap.chain(paid_model, keys)] == ["groq"]
