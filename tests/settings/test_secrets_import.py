"""Перенос ключей из .env в зашифрованное хранилище (src/settings/secrets_import.py)."""

import json

import pytest
from cryptography.fernet import Fernet

from src.settings import secrets_import as si
from src.settings import secrets_store as ss

GROQ = "gsk_" + "a" * 30


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("BITBOT_SECRETS_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("SECRETS_PATH", str(tmp_path / "secrets.enc.json"))
    return tmp_path / "secrets.enc.json"


def test_parse_env_handles_comments_quotes_and_inline_notes():
    text = (
        "# комментарий\n\nGROQ_API_KEY=abc\nA='q # not comment'\nB=\"x\"\n"
        "C=plain   # заметка\nexport D=1\nEMPTY=\n"
    )
    assert si.parse_env(text) == {
        "GROQ_API_KEY": "abc",
        "A": "q # not comment",
        "B": "x",
        "C": "plain",
        "D": "1",
    }


def test_import_saves_encrypted_and_never_returns_values(store, capsys):
    done, skipped = si.import_values(
        {"GROQ_API_KEY": GROQ, "UNKNOWN": "x", "MODE": "ai"}
    )
    assert done == {"groq": ["GROQ_API_KEY"]} and skipped == []
    assert ss.load_values()["GROQ_API_KEY"] == GROQ
    assert GROQ not in store.read_text()  # на диске только шифртекст
    meta = json.loads(store.read_text())["meta"]["groq"]
    assert meta["status"] == "warn" and "не проверено" in meta["message"]


def test_existing_values_are_not_overwritten_unless_asked(store):
    si.import_values({"GROQ_API_KEY": GROQ})
    newer = "gsk_" + "b" * 30
    assert si.import_values({"GROQ_API_KEY": newer})[0] == {}
    assert ss.load_values()["GROQ_API_KEY"] == GROQ
    assert si.import_values({"GROQ_API_KEY": newer}, overwrite=True)[0] == {
        "groq": ["GROQ_API_KEY"]
    }
    assert ss.load_values()["GROQ_API_KEY"] == newer


def test_bad_format_is_skipped_not_saved(store):
    done, skipped = si.import_values(
        {"TELEGRAM_BOT_TOKEN": "x" * 40, "TELEGRAM_CHAT_ID": "не число"}
    )
    assert "TELEGRAM_CHAT_ID" in skipped
    assert "TELEGRAM_CHAT_ID" not in ss.load_values()
    assert done == {"telegram": ["TELEGRAM_BOT_TOKEN"]}


def test_main_refuses_without_master_key(monkeypatch, capsys):
    monkeypatch.delenv("BITBOT_SECRETS_KEY", raising=False)
    assert si.main() == 3
    assert "BITBOT_SECRETS_KEY" in capsys.readouterr().out
