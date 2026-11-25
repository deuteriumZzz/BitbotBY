"""Допуск SAC и счётчик мета-модели в API приложения."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from bitbot.research import meta_dataset

KEY = "test-panel-key"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DASHBOARD_API_KEY", KEY)
    from bitbot.api import app

    return TestClient(app, raise_server_exceptions=False)


def test_approve_goes_to_the_bot_with_profile(client):
    ok = JSONResponse({"status": "done"})
    with patch(
        "bitbot.api.routers.control._command", new=AsyncMock(return_value=ok)
    ) as cmd:
        resp = client.post(
            "/api/model/approve",
            json={"profile": "meme", "request_id": "abcdefgh12"},
            headers={"X-API-Key": KEY},
        )
    assert resp.status_code == 200
    assert cmd.await_args.args[1:] == ("approve_model", {"profile": "meme"})


def test_meta_status_reads_ledger(client, monkeypatch, tmp_path):
    monkeypatch.setenv("DECISION_LEDGER_PATH", str(tmp_path / "none.jsonl"))
    monkeypatch.setattr(meta_dataset, "_cache", {})
    resp = client.get("/api/meta/status", headers={"X-API-Key": KEY})
    assert resp.status_code == 200
    assert resp.json()["have"] == 0 and resp.json()["need"] == meta_dataset.NEED
