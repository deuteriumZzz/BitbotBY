"""Проверка целостности и резервные копии SQLite (боевые БД сделок) — без записи в исходник.

Урок инцидента 2026-11-23 (AUDIT_REPORT §10d): обычное `sqlite3.connect(...).execute(...)`
к БД в режиме WAL вливает журнал в файл и может испортить базу. Поэтому здесь:
  * исходник открывается ТОЛЬКО `mode=ro` (никаких checkpoint/записи);
  * снимок делается backup API в отдельный файл, а проверка (`integrity_check`) идёт по
    снимку — с боевым файлом ничего не происходит;
  * резервная копия проверяется сразу после создания; негодная удаляется и не считается копией;
  * ротация трогает только файлы своей же маски `<имя>.<метка>.db`.
Если снять копию не удалось (например, БД повреждена настолько, что backup падает), это
отдельная находка: `ok=False`, `error` с причиной.
CLI:  python -m bitbot.storage.db_health [путь ...] [--backup] [--dest data/backups] [--keep 7]
"""

from __future__ import annotations

import os
import re
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote

DEFAULT_DBS = ("data/trades_paper.db", "data/trades_live.db")
MAX_MESSAGES = 5


def _ro(path: str) -> sqlite3.Connection:
    uri = "file:" + quote(os.path.abspath(path)) + "?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=5)


def snapshot(path: str, dest: str) -> None:
    """Согласованная копия через backup API с read-only соединения. Исходник не меняется."""
    src = _ro(path)
    try:
        dst = sqlite3.connect(dest)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


def _check(path: str, quick: bool) -> Dict[str, Any]:
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            "PRAGMA quick_check" if quick else "PRAGMA integrity_check"
        ).fetchall()
        msgs = [str(r[0]) for r in rows]
        ok = msgs == ["ok"]
        tables: Dict[str, int] = {}
        if ok:
            names = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            ]
            for name in names[:20]:
                quoted = name.replace(
                    '"', '""'
                )  # имя таблицы — идентификатор, не параметр
                tables[name] = conn.execute(
                    f'SELECT COUNT(*) FROM "{quoted}"'
                ).fetchone()[0]
        return {"ok": ok, "messages": msgs[:MAX_MESSAGES], "tables": tables}
    finally:
        conn.close()


def integrity_report(path: str, quick: bool = False) -> Dict[str, Any]:
    """Целостность по снимку (боевой файл не пишется). Не бросает исключений."""
    out: Dict[str, Any] = {
        "path": path,
        "ok": False,
        "messages": [],
        "tables": {},
        "error": None,
    }
    if not os.path.exists(path):
        out["error"] = "файл не найден"
        return out
    out["size_mb"] = round(os.path.getsize(path) / 1e6, 2)
    wal = path + "-wal"
    out["wal_mb"] = round(os.path.getsize(wal) / 1e6, 2) if os.path.exists(wal) else 0.0
    try:
        with tempfile.TemporaryDirectory() as tmp:
            copy = os.path.join(tmp, "snapshot.db")
            snapshot(path, copy)
            out.update(_check(copy, quick))
    except sqlite3.Error as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _rm(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def backup(
    path: str, dest_dir: str, keep: int = 7, now: Optional[datetime] = None
) -> Dict[str, Any]:
    """Проверенная копия `<имя>.<UTC-метка>.db` + ротация (оставляет `keep` последних)."""
    res: Dict[str, Any] = {
        "source": path,
        "ok": False,
        "backup": None,
        "error": None,
        "removed": [],
    }
    if not os.path.exists(path):
        res["error"] = "файл не найден"
        return res
    stem = os.path.splitext(os.path.basename(path))[0]
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(dest_dir, exist_ok=True)
    target = os.path.join(dest_dir, f"{stem}.{stamp}.db")
    try:
        snapshot(path, target)
        verdict = _check(target, quick=True)
    except sqlite3.Error as exc:
        _rm(target)
        res["error"] = f"{type(exc).__name__}: {exc}"
        return res
    if not verdict["ok"]:
        _rm(
            target
        )  # негодная копия хуже отсутствия: создаёт ложное чувство безопасности
        res["error"] = "копия не прошла проверку: " + "; ".join(verdict["messages"])
        return res
    res.update(ok=True, backup=target)
    pattern = re.compile(rf"^{re.escape(stem)}\.\d{{8}}T\d{{6}}Z\.db$")
    mine = sorted(f for f in os.listdir(dest_dir) if pattern.match(f))
    for old in mine[: max(0, len(mine) - max(1, keep))]:
        _rm(os.path.join(dest_dir, old))
        res["removed"].append(old)
    return res


def check_all(
    paths: Optional[List[str]] = None, quick: bool = False
) -> List[Dict[str, Any]]:
    return [
        integrity_report(p, quick)
        for p in (paths or list(DEFAULT_DBS))
        if os.path.exists(p)
    ]


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    do_backup = "--backup" in args
    dest, keep, paths = "data/backups", 7, []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--dest" and i + 1 < len(args):
            dest = args[i + 1]
            i += 1
        elif a == "--keep" and i + 1 < len(args):
            keep = int(args[i + 1])
            i += 1
        elif not a.startswith("--"):
            paths.append(a)
        i += 1
    bad = 0
    for rep in check_all(paths or None):
        status = "OK" if rep["ok"] else "ПРОБЛЕМА"
        detail = rep["tables"] or rep["messages"] or rep["error"]
        print(
            f"{status}  {rep['path']}  {rep.get('size_mb')} МБ (WAL {rep.get('wal_mb')} МБ)  {detail}"
        )
        bad += 0 if rep["ok"] else 1
        if do_backup and rep["ok"]:
            b = backup(rep["path"], dest, keep)
            print("  копия:", b["backup"] if b["ok"] else f"НЕ СОЗДАНА ({b['error']})")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
