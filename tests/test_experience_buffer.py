"""Тесты модуля experience_buffer."""
import json
import os

import pytest

from src import experience_buffer as eb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SNAP = {
    "symbol": "BTC/USDT",
    "price": 64000.0,
    "volume_ratio": 1.4,
    "indicators": {"rsi": 55.2, "macd": 12.3},
}


def _write_lines(path: str, lines: list) -> None:
    """Write raw text lines to a file, creating parent dirs."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------


class TestSave:
    def test_save_buy_records_pnl(self, tmp_path):
        path = str(tmp_path / "exp.jsonl")
        eb.save(_SNAP, "buy", entry_price=100.0, exit_price=110.0, path=path)
        assert os.path.exists(path)
        record = json.loads(open(path).readline())
        expected_pnl = round((110.0 - 100.0) / 100.0, 6)
        assert record["pnl_pct"] == expected_pnl

    def test_save_sell_records_pnl(self, tmp_path):
        path = str(tmp_path / "exp.jsonl")
        eb.save(_SNAP, "sell", entry_price=100.0, exit_price=90.0, path=path)
        record = json.loads(open(path).readline())
        expected_pnl = round((100.0 - 90.0) / 100.0, 6)
        assert record["pnl_pct"] == expected_pnl

    def test_save_empty_snap_is_noop(self, tmp_path):
        path = str(tmp_path / "exp.jsonl")
        eb.save({}, "buy", entry_price=100.0, exit_price=110.0, path=path)
        assert not os.path.exists(path)

    def test_save_creates_directory(self, tmp_path):
        subdir = tmp_path / "nested" / "dir"
        path = str(subdir / "exp.jsonl")
        eb.save(_SNAP, "buy", entry_price=100.0, exit_price=110.0, path=path)
        assert os.path.exists(path)

    def test_save_appends_multiple(self, tmp_path):
        path = str(tmp_path / "exp.jsonl")
        eb.save(_SNAP, "buy", entry_price=100.0, exit_price=110.0, path=path)
        eb.save(_SNAP, "sell", entry_price=110.0, exit_price=105.0, path=path)
        lines = [l for l in open(path).readlines() if l.strip()]
        assert len(lines) == 2

    def test_save_record_has_required_keys(self, tmp_path):
        path = str(tmp_path / "exp.jsonl")
        eb.save(_SNAP, "buy", entry_price=100.0, exit_price=110.0, path=path)
        record = json.loads(open(path).readline())
        for key in ("ts", "symbol", "action", "entry_price", "exit_price",
                    "pnl_pct", "indicators", "volume_ratio", "price"):
            assert key in record, f"Missing key: {key}"

    def test_save_record_values(self, tmp_path):
        path = str(tmp_path / "exp.jsonl")
        eb.save(_SNAP, "buy", entry_price=64000.0, exit_price=65000.0, path=path)
        record = json.loads(open(path).readline())
        assert record["symbol"] == "BTC/USDT"
        assert record["action"] == "buy"
        assert record["entry_price"] == 64000.0
        assert record["exit_price"] == 65000.0
        assert record["volume_ratio"] == 1.4
        assert record["price"] == 64000.0


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_missing_file_returns_empty(self, tmp_path):
        path = str(tmp_path / "nonexistent.jsonl")
        assert eb.load(path=path) == []

    def test_load_valid_jsonl(self, tmp_path):
        path = str(tmp_path / "exp.jsonl")
        records = [
            json.dumps({"a": 1}),
            json.dumps({"b": 2}),
            json.dumps({"c": 3}),
        ]
        _write_lines(path, records)
        result = eb.load(path=path)
        assert len(result) == 3
        assert result[0] == {"a": 1}

    def test_load_skips_invalid_json_lines(self, tmp_path):
        path = str(tmp_path / "exp.jsonl")
        lines = [
            json.dumps({"ok": True}),
            "not json at all {{",
            json.dumps({"also": "valid"}),
        ]
        _write_lines(path, lines)
        result = eb.load(path=path)
        assert len(result) == 2
        assert result[0] == {"ok": True}
        assert result[1] == {"also": "valid"}

    def test_load_skips_empty_lines(self, tmp_path):
        path = str(tmp_path / "exp.jsonl")
        # Write file manually with blank lines
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"x": 1}) + "\n")
            f.write("\n")
            f.write("   \n")
            f.write(json.dumps({"y": 2}) + "\n")
        result = eb.load(path=path)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# count()
# ---------------------------------------------------------------------------


class TestCount:
    def test_count_missing_file(self, tmp_path):
        path = str(tmp_path / "nonexistent.jsonl")
        assert eb.count(path=path) == 0

    def test_count_empty_file(self, tmp_path):
        path = str(tmp_path / "exp.jsonl")
        open(path, "w").close()
        assert eb.count(path=path) == 0

    def test_count_with_records(self, tmp_path):
        path = str(tmp_path / "exp.jsonl")
        n = 5
        lines = [json.dumps({"i": i}) for i in range(n)]
        _write_lines(path, lines)
        assert eb.count(path=path) == n

    def test_count_skips_blank_lines(self, tmp_path):
        path = str(tmp_path / "exp.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"a": 1}) + "\n")
            f.write("\n")
            f.write("   \n")
            f.write(json.dumps({"b": 2}) + "\n")
        assert eb.count(path=path) == 2


# ---------------------------------------------------------------------------
# _trim()
# ---------------------------------------------------------------------------


class TestTrim:
    def test_trim_keeps_last_max_lines(self, tmp_path):
        path = str(tmp_path / "exp.jsonl")
        max_lines = eb._MAX_LINES
        total = max_lines + 10
        lines = [json.dumps({"i": i}) for i in range(total)]
        _write_lines(path, lines)
        eb._trim(path)
        with open(path, encoding="utf-8") as f:
            remaining = f.readlines()
        assert len(remaining) == max_lines
        # Last line should be the last record written
        last = json.loads(remaining[-1])
        assert last["i"] == total - 1

    def test_trim_noop_when_under_limit(self, tmp_path):
        path = str(tmp_path / "exp.jsonl")
        max_lines = eb._MAX_LINES
        total = max_lines - 1
        lines = [json.dumps({"i": i}) for i in range(total)]
        _write_lines(path, lines)
        eb._trim(path)
        with open(path, encoding="utf-8") as f:
            remaining = f.readlines()
        assert len(remaining) == total
