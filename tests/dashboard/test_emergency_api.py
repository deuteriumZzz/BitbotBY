"""Аварийное управление из приложения: /api/emergency и команда шины `emergency`."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from src.ops import emergency

KEY = "test-panel-key"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("DASHBOARD_API_KEY", KEY)
    monkeypatch.setattr(emergency, "DEFAULT_PATH", str(tmp_path / "em.json"))
    from src.dashboard import app

    return TestClient(app, raise_server_exceptions=False)


def _post(client, body):
    return client.post("/api/emergency", json=body, headers={"X-API-Key": KEY})


def test_state_and_level_descriptions(client):
    resp = client.get("/api/emergency", headers={"X-API-Key": KEY})
    assert resp.status_code == 200
    data = resp.json()
    assert data["level"] == 0 and set(data["levels"]) == {"1", "2", "3", "4"}


def test_preview_without_confirm_sends_nothing_to_the_bot(client):
    with patch("src.dashboard.routers.control._command", new=AsyncMock()) as cmd:
        resp = _post(client, {"action": "3"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["preview"] is True and len(body["steps"]) == 3
    cmd.assert_not_awaited()


def test_bad_level_is_rejected(client):
    assert _post(client, {"action": "9", "confirm": True}).status_code == 400


def test_confirmed_action_goes_to_the_bot(client):
    done = AsyncMock(return_value=JSONResponse({"status": "done"}))
    with patch("src.dashboard.routers.control._command", new=done):
        resp = _post(
            client, {"action": "2", "confirm": True, "request_id": "r-12345678"}
        )
    assert resp.status_code == 200
    _, kind, args = done.await_args.args
    assert kind == "emergency" and args == {"action": "2", "confirm": True}


def test_control_needs_a_key(client):
    assert client.post("/api/emergency", json={"action": "1"}).status_code == 401


async def test_bus_handler_emits_event_only_when_something_changed():
    from src.core.bot_commands import BotCommands

    cmds = object.__new__(BotCommands)
    cmds._bot = SimpleNamespace(redis=SimpleNamespace(redis_client=object()))
    with patch("src.core.bot_commands.emit") as emit, patch(
        "src.core.ops_tasks.emergency_apply",
        new=AsyncMock(return_value={"ok": True, "preview": True, "steps": ["1. x"]}),
    ):
        await cmds.emergency({"action": "1"})
        emit.assert_not_called()  # предпросмотр ничего не меняет
    with patch("src.core.bot_commands.emit") as emit, patch(
        "src.core.ops_tasks.emergency_apply",
        new=AsyncMock(return_value={"level": 1, "executed": ["block_entries"]}),
    ):
        await cmds.emergency({"action": "1", "confirm": True})
        emit.assert_called_once()
