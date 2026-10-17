"""Реестр настроек и /api/config (src/settings_registry.py)."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from config import Config
from src import settings_registry as reg
from src.dashboard import data
from tests.test_bot_bus import FakeRedis


def test_every_setting_has_label_and_known_type():
    for s in reg.SETTINGS:
        assert s.label and s.group
        assert s.type in ("int", "float", "pct", "bool", "choice", "str"), s.key
        if s.type == "choice":
            assert s.options, s.key


def test_coerce_checks_type_and_range():
    s = reg.BY_KEY["MAX_POSITIONS"]
    assert reg.coerce(s, "5") == 5
    with pytest.raises(ValueError, match="Не больше 20"):
        reg.coerce(s, 21)
    with pytest.raises(ValueError, match="целое"):
        reg.coerce(s, 2.5)
    with pytest.raises(ValueError):
        reg.coerce(reg.BY_KEY["AUTO_EXECUTE"], "maybe")
    with pytest.raises(ValueError, match="Нет такого"):
        reg.coerce(reg.BY_KEY["MODE"], "yolo")
    with pytest.raises(ValueError, match="Не меньше 10"):
        reg.coerce(reg.BY_KEY["TELEGRAM_CONFIRM_TIMEOUT"], 5)
    # Дольше 2 минут на решение — сигнал устаревает.
    with pytest.raises(ValueError, match="Не больше 120"):
        reg.coerce(reg.BY_KEY["TELEGRAM_CONFIRM_TIMEOUT"], 300)


def _current():
    return {
        k: {"value": getattr(Config, s.config_attr, s.default)}
        for k, s in reg.BY_KEY.items()
    }


def test_validate_patch_rejects_unknown_and_readonly():
    _, errors = reg.validate_patch(Config, {"NOPE": 1, "REDIS_URL": "x"}, _current())
    assert set(errors) == {"NOPE", "REDIS_URL"}


def test_validate_patch_runs_config_validate(monkeypatch):
    monkeypatch.setattr(Config, "PAPER_TRADING", True)
    # Дневной лимит больше предельной просадки — Config.validate() это запрещает.
    patch_ = {"DAILY_LOSS_LIMIT": 0.3, "MAX_DRAWDOWN_PERCENT": 0.1}
    _, errors = reg.validate_patch(Config, patch_, _current())
    assert "_config" in errors and "DAILY_LOSS_LIMIT" in errors["_config"]


def test_validate_patch_leverage_min_max():
    _, errors = reg.validate_patch(
        Config, {"LEVERAGE_MIN": 10, "LEVERAGE_MAX": 5}, _current()
    )
    assert "LEVERAGE_MIN" in errors


def test_apply_patch_hot_and_restart():
    r, rc = FakeRedis(), MagicMock()
    rc.set_max_positions.return_value = True
    result = reg.apply_patch(rc, r, {"MAX_POSITIONS": 4, "ATR_SPIKE_MULT": 4.0})
    rc.set_max_positions.assert_called_once_with(4)
    assert result["applied_hot"] == ["MAX_POSITIONS"]
    assert result["pending_restart"] == ["ATR_SPIKE_MULT"]
    assert json.loads(r.hgetall(reg.OVERRIDES_KEY)["ATR_SPIKE_MULT"]) == 4.0


def test_apply_patch_none_removes_override():
    r = FakeRedis()
    r.hset(reg.OVERRIDES_KEY, "ATR_SPIKE_MULT", "4.0")
    reg.apply_patch(MagicMock(), r, {"ATR_SPIKE_MULT": None})
    assert "ATR_SPIKE_MULT" not in r.hgetall(reg.OVERRIDES_KEY)


def test_load_overrides_skips_invalid_values():
    r = FakeRedis()
    r.hset(reg.OVERRIDES_KEY, "ATR_SPIKE_MULT", "999")  # больше max
    r.hset(reg.OVERRIDES_KEY, "LOG_LEVEL", '"DEBUG"')
    assert reg.load_overrides(r) == {"LOG_LEVEL": "DEBUG"}


class _Cfg:
    ATR_SPIKE_MULT = 3.0
    STOP_LOSS_PERCENT = 0.05

    def validate(self):
        if self.STOP_LOSS_PERCENT > 0.2:
            raise ValueError("STOP_LOSS_PERCENT too big")


def test_apply_overrides_sets_config_and_env(monkeypatch):
    monkeypatch.delenv("ATR_SPIKE_MULT", raising=False)
    r = FakeRedis()
    r.hset(reg.OVERRIDES_KEY, "ATR_SPIKE_MULT", "4.5")
    cfg = type("Cfg", (_Cfg,), {})
    assert reg.apply_overrides(cfg, r) == ["ATR_SPIKE_MULT"]
    assert cfg.ATR_SPIKE_MULT == 4.5
    assert os.environ["ATR_SPIKE_MULT"] == "4.5"
    monkeypatch.delenv("ATR_SPIKE_MULT", raising=False)


def test_apply_overrides_rolls_back_when_they_break_validate(monkeypatch):
    monkeypatch.delenv("STOP_LOSS_PERCENT", raising=False)
    r = FakeRedis()
    r.hset(reg.OVERRIDES_KEY, "STOP_LOSS_PERCENT", "0.25")
    cfg = type("Cfg", (_Cfg,), {})
    assert reg.apply_overrides(cfg, r) == []
    assert cfg.STOP_LOSS_PERCENT == 0.05
    assert "STOP_LOSS_PERCENT" not in os.environ


# ── /api/config ───────────────────────────────────────────────────────────────


@pytest.fixture()
def keyed(monkeypatch):
    monkeypatch.setenv("DASHBOARD_API_KEY", "secret")
    monkeypatch.setattr(Config, "PAPER_TRADING", True)
    from src.dashboard import app

    return TestClient(app, raise_server_exceptions=False)


H = {"X-API-Key": "secret"}


def test_schema_lists_public_fields(keyed):
    resp = keyed.get("/api/config/schema", headers=H)
    assert resp.status_code == 200
    first = resp.json()[0]
    assert {"key", "label", "type", "hot", "options"} <= first.keys()
    assert "hot_set" not in first


def test_patch_config_saves_and_reports(keyed):
    r = FakeRedis()
    with patch.object(data, "get_redis", return_value=r):
        resp = keyed.patch("/api/config", json={"ATR_SPIKE_MULT": 4.0}, headers=H)
    assert resp.status_code == 200, resp.text
    assert resp.json()["pending_restart"] == ["ATR_SPIKE_MULT"]
    assert "config.changed" in [m["type"] for _, m in r.published]


def test_patch_config_rejects_invalid(keyed):
    with patch.object(data, "get_redis", return_value=FakeRedis()):
        resp = keyed.patch("/api/config", json={"MAX_POSITIONS": 99}, headers=H)
    assert resp.status_code == 422
    assert "MAX_POSITIONS" in resp.json()["error"]["fields"]
