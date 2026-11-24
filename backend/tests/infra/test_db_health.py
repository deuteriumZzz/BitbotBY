"""Проверка целостности и резервные копии SQLite: исходник не меняется, плохая копия не остаётся."""

import hashlib
import os
import sqlite3
from datetime import datetime, timezone

from bitbot.storage import db_health as dh


def sha(path):
    return (
        hashlib.sha256(open(path, "rb").read()).hexdigest()
        if os.path.exists(path)
        else None
    )


def make_db(path, rows=3000, wal=False):
    conn = sqlite3.connect(path)
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE trades (id INTEGER PRIMARY KEY, symbol TEXT, pnl REAL, note TEXT)"
    )
    conn.executemany(
        "INSERT INTO trades(symbol, pnl, note) VALUES (?,?,?)",
        [("BTC/USDT", i * 0.1, "x" * 40) for i in range(rows)],
    )
    conn.commit()
    return conn


def corrupt(path):
    with open(path, "r+b") as f:
        f.seek(4096 * 4)
        f.write(b"\xff" * 600)


def test_wal_database_is_checked_without_touching_source(tmp_path):
    p = str(tmp_path / "live.db")
    writer = make_db(p, wal=True)
    writer.executemany(
        "INSERT INTO trades(symbol, pnl, note) VALUES (?,?,?)",
        [("ETH/USDT", 1.0, "y")] * 50,
    )
    writer.commit()  # эти строки ещё в WAL, не в основном файле
    before = (sha(p), sha(p + "-wal"), os.path.getmtime(p))
    rep = dh.integrity_report(p)
    assert rep["ok"] and rep["tables"] == {
        "trades": 3050
    }  # снимок включает зафиксированный WAL
    assert (
        sha(p),
        sha(p + "-wal"),
        os.path.getmtime(p),
    ) == before  # боевые файлы не изменились
    writer.close()


def test_missing_and_corrupted_database_reported(tmp_path):
    assert dh.integrity_report(str(tmp_path / "none.db"))["error"] == "файл не найден"
    p = str(tmp_path / "bad.db")
    make_db(p).close()
    corrupt(p)
    rep = dh.integrity_report(p)
    assert rep["ok"] is False and (rep["messages"] or rep["error"])


def test_backup_is_verified_rotated_and_leaves_source_untouched(tmp_path):
    p = str(tmp_path / "trades_paper.db")
    make_db(p, rows=200).close()
    dest = str(tmp_path / "backups")
    os.makedirs(dest)
    foreign = os.path.join(dest, "notes.txt")
    open(foreign, "w").write("keep me")
    src_hash = sha(p)
    made = []
    for day in range(1, 5):
        r = dh.backup(
            p, dest, keep=2, now=datetime(2026, 11, day, 12, tzinfo=timezone.utc)
        )
        assert r["ok"]
        made.append(os.path.basename(r["backup"]))
    left = sorted(f for f in os.listdir(dest) if f.endswith(".db"))
    assert left == made[-2:]  # остались две последние
    assert os.path.exists(foreign) and sha(p) == src_hash  # чужой файл и исходник целы
    assert dh.integrity_report(os.path.join(dest, left[-1]))["tables"] == {
        "trades": 200
    }


def test_backup_of_corrupted_database_leaves_no_bad_copy(tmp_path):
    p = str(tmp_path / "bad.db")
    make_db(p).close()
    corrupt(p)
    dest = str(tmp_path / "b")
    r = dh.backup(p, dest, now=datetime(2026, 11, 1, tzinfo=timezone.utc))
    assert r["ok"] is False and r["error"]
    assert not [f for f in os.listdir(dest) if f.endswith(".db")]
    assert dh.backup(str(tmp_path / "none.db"), dest)["error"] == "файл не найден"


def test_cli_exit_codes_and_backup_flag(tmp_path, capsys):
    good, bad = str(tmp_path / "g.db"), str(tmp_path / "b.db")
    make_db(good, rows=50).close()
    make_db(bad).close()
    corrupt(bad)
    dest = str(tmp_path / "bk")
    assert dh.main([good, "--backup", "--dest", dest, "--keep", "3"]) == 0
    assert "OK" in capsys.readouterr().out and len(os.listdir(dest)) == 1
    assert dh.main([bad]) == 1
    assert "ПРОБЛЕМА" in capsys.readouterr().out


def test_check_all_skips_missing_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert dh.check_all() == []
