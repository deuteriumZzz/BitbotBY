"""Telegram-команды /digest /audit /dbcheck: тексты и обработчики (только чтение)."""

import json
import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.ops import telegram_views as tv
from src.ops.decision_audit import audit_decision
from src.telegram_bot.commander import TelegramCommander

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


# ── обработчики командера ─────────────────────────────────────────────────────
def commander():
    c = TelegramCommander.__new__(TelegramCommander)
    c._notifier = SimpleNamespace(_chat_id="42")
    return c


def update(chat_id="42"):
    return SimpleNamespace(
        effective_chat=SimpleNamespace(id=chat_id),
        message=SimpleNamespace(reply_text=AsyncMock()),
    )


async def test_handlers_reply_plain_text_when_authorized(ledger):
    c = commander()
    u = update()
    await c._cmd_digest(u, SimpleNamespace(args=["6"]))
    call = u.message.reply_text.await_args
    assert "Сводка за 6 ч" in call.args[0] and "parse_mode" not in call.kwargs
    u2 = update()
    await c._cmd_audit(u2, SimpleNamespace(args=["d1"]))
    assert "Аудит решения d1" in u2.message.reply_text.await_args.args[0]
    u3 = update()
    await c._cmd_audit(u3, SimpleNamespace(args=[]))
    assert "Использование" in u3.message.reply_text.await_args.args[0]


async def test_handlers_ignore_foreign_chats(ledger):
    c = commander()
    for handler in (c._cmd_digest, c._cmd_audit, c._cmd_dbcheck):
        u = update(chat_id="999")
        await handler(u, SimpleNamespace(args=["d1"]))
        u.message.reply_text.assert_not_awaited()


async def test_handler_failure_becomes_a_message_not_an_exception(monkeypatch):
    c = commander()

    def boom(*_):
        raise OSError("disk")

    monkeypatch.setattr(tv, "digest_text", boom)
    u = update()
    await c._cmd_digest(u, SimpleNamespace(args=[]))
    assert "Не удалось построить сводку" in u.message.reply_text.await_args.args[0]
