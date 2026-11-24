"""Отчёты для приложения: сводка, разбор решения, проверка БД (только чтение)."""

import json
import sqlite3
from datetime import datetime, timezone

import pytest

from src.ops import reports as tv
from src.ops.decision_audit import audit_decision

NOW = datetime.now(timezone.utc).isoformat()
RECS = [
    {"ts": NOW, "kind": "data_quality", "trace_id": "T1", "counts": {"VALID": 3}},
    {
        "ts": NOW,
        "kind": "ai_call",
        "trace_id": "T1",
        "status": "ok",
        "provider": "p",
        "latency_ms": 90,
        "attempts": [],
    },
    {
        "ts": NOW,
        "kind": "signals",
        "trace_id": "T1",
        "decisions": [{"decision_id": "d1", "symbol": "BTC/USDT"}],
    },
    {
        "ts": NOW,
        "kind": "execution",
        "trace_id": "T1",
        "decision_id": "d1",
        "opened": True,
    },
    {
        "ts": NOW,
        "kind": "outcome",
        "trace_id": "T9",
        "decision_id": "d1",
        "reason": "tp",
        "pnl": 4.2,
    },
]


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    p = tmp_path / "ledger.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in RECS) + "\n")
    monkeypatch.setenv("DECISION_LEDGER_PATH", str(p))
    return str(p)


def test_parse_hours_clamps_and_ignores_garbage():
    assert tv.parse_hours(None) == 24.0 and tv.parse_hours("6") == 6.0
    assert (
        tv.parse_hours("0") == 1.0
        and tv.parse_hours("9999") == 168.0
        and tv.parse_hours("abc") == 24.0
    )


def test_audit_finds_outcome_in_another_cycle_by_decision_id(ledger):
    a = audit_decision(RECS, "d1")
    assert a["chain"]["outcome"]["pnl"] == 4.2 and a["complete"]
    assert "Исход" in tv.audit_text("d1", ledger)
    assert "не найдено" in tv.audit_text(
        "zzz", ledger
    ) and "Использование" in tv.audit_text("  ")


def test_digest_text_and_empty_ledger(ledger, tmp_path):
    text = tv.digest_text(24, ledger)
    assert "Сводка за 24 ч" in text and "Тревоги" in text
    assert "пуст" in tv.digest_text(24, str(tmp_path / "missing.jsonl"))


def test_long_text_is_cut():
    cut = tv._cut("x" * 10_000)
    assert len(cut) <= tv.LIMIT and cut.endswith("(обрезано)")


def test_dbcheck_text_ok_and_missing(tmp_path):
    p = tmp_path / "t.db"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE trades (id INTEGER)")
    c.executemany("INSERT INTO trades VALUES (?)", [(i,) for i in range(7)])
    c.commit()
    c.close()
    out = tv.dbcheck_text([str(p)])
    assert "✅" in out and "trades=7" in out and "боевой файл не затронут" in out
    assert (
        tv.dbcheck_text([str(tmp_path / "none.db")]) == "Файлов БД сделок не найдено."
    )


# ── эндпоинты приложения ──────────────────────────────────────────────────────
@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from src.dashboard import app

    return TestClient(app, raise_server_exceptions=False)


def test_report_endpoints_return_text(ledger, client):
    digest = client.get("/api/reports/digest?hours=6")
    assert digest.status_code == 200 and "Сводка за 6 ч" in digest.json()["text"]
    audit = client.get("/api/reports/audit/d1")
    assert audit.status_code == 200 and "Аудит решения d1" in audit.json()["text"]
    assert client.get("/api/reports/audit/bad id!").status_code in (400, 404)


def test_report_failure_hides_details(monkeypatch, client):
    def boom(*_):
        raise OSError("/secret/path/trades.db")

    monkeypatch.setattr(tv, "digest_text", boom)
    resp = client.get("/api/reports/digest")
    assert resp.status_code == 500
    assert "secret" not in resp.text and "OSError" not in resp.text
