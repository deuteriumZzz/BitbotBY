"""Каталог AI-провайдеров и правила цепочки (backend/bitbot/config/ai_providers.py)."""

import pytest

from bitbot.config import ai_providers as ap


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
    assert ap.BY_ID["groq"].is_free("any-new-groq-model")
    assert not ap.BY_ID["openai"].is_free("gpt-9")


def test_clean_chain_keeps_only_models():
    assert ap.clean_chain({"models": {"groq": " openai/gpt-oss-20b "}}) == {
        "models": {"groq": "openai/gpt-oss-20b"}
    }
    # прежние настройки цепочки больше ничего не значат
    assert ap.clean_chain({"mode": "paid", "order": ["gemini"]}) == {"models": {}}
    with pytest.raises(ValueError):
        ap.clean_chain({"models": {"nope": "m"}})
    with pytest.raises(ValueError):
        ap.clean_chain({"models": {"groq": ""}})


def test_chain_is_free_first_then_paid_and_ignores_old_settings():
    avail = ["anthropic", "groq", "deepseek", "gemini"]
    old = {"mode": "paid", "order": ["anthropic"], "disabled": ["groq"], "single": True}
    assert [p for p, _ in ap.chain(old, avail)] == [
        "groq",
        "gemini",
        "deepseek",
        "anthropic",
    ]
    # выбранная в приложении модель — главнее рекомендованной
    assert ("groq", "openai/gpt-oss-20b") in ap.chain(
        {"models": {"groq": "openai/gpt-oss-20b"}}, avail
    )
    assert ap.chain({}, []) == []
