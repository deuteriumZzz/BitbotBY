"""Стратегия одна на все профили: профиль меняет её числа, а не плодит копии."""

import re
from pathlib import Path

import pytest

from bitbot.config import runtime
from bitbot.config.runtime import STRATEGY_PARAMS
from bitbot.domain.strategies import LEGACY_ALIASES, STRATEGY_REGISTRY, create_strategy

PROFILES = ("bluechip", "altcoin", "meme")
STRATEGIES_DIR = (
    Path(__file__).resolve().parents[2] / "bitbot" / "domain" / "strategies"
)


def test_no_strategy_copy_per_profile():
    copies = [
        n for n in STRATEGY_REGISTRY if n.endswith(tuple(f"_{p}" for p in PROFILES))
    ]
    assert (
        not copies
    ), f"копии стратегии под профиль — числа профиля в STRATEGY_PARAMS: {copies}"


def test_no_profile_numbers_inside_strategy_code():
    pattern = re.compile(r"[\"'](bluechip|altcoin|meme)[\"']\s*:")
    found = [
        f"{p.relative_to(STRATEGIES_DIR)}:{i}"
        for p in STRATEGIES_DIR.rglob("*.py")
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line) and p.name != "__init__.py"
    ]
    assert (
        not found
    ), f"числа профиля внутри стратегии — перенесите в STRATEGY_PARAMS: {found}"


def test_params_table_names_real_strategies_and_profiles():
    for name, params in STRATEGY_PARAMS.items():
        assert name in STRATEGY_REGISTRY, name
        for by_profile in params.values():
            assert set(by_profile) <= set(PROFILES), (name, by_profile)


@pytest.mark.parametrize(
    "profile, spike", [("bluechip", 2.0), ("altcoin", 3.0), ("meme", 5.0)]
)
def test_one_strategy_takes_numbers_of_active_profile(monkeypatch, profile, spike):
    monkeypatch.setattr(runtime.active, "_active_profile", profile)
    assert create_strategy("volume_spike").threshold() == spike
    assert create_strategy("scalping").param("high_volume", 0) == 1.5
    # старые имена копий в истории сделок ведут на ту же одну стратегию
    assert {LEGACY_ALIASES[f"volume_spike_{profile}"]} == {"volume_spike"}
