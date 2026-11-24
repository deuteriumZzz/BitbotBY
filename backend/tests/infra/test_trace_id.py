"""trace_id цикла попадает в каждую строку лога — тот же, что в журнале решений."""

import json
import logging

from bitbot.infra.logger import TRACE_ID, JSONFormatter, _TraceFilter
from bitbot.storage.decision_ledger import DecisionLedger


def _line(fmt: logging.Formatter) -> str:
    record = logging.LogRecord("bot", logging.INFO, __file__, 1, "цикл", (), None)
    _TraceFilter().filter(record)
    return fmt.format(record)


def test_cycle_trace_id_in_json_and_text_logs(tmp_path):
    text = logging.Formatter("%(name)s%(trace_tag)s — %(message)s")
    token = TRACE_ID.set("")
    try:
        assert _line(text) == "bot — цикл"
        assert "trace_id" not in json.loads(_line(JSONFormatter()))

        tid = DecisionLedger(path=str(tmp_path / "l.jsonl")).begin_cycle(7)
        assert _line(text) == f"bot [{tid}] — цикл"
        assert json.loads(_line(JSONFormatter()))["trace_id"] == tid
    finally:
        TRACE_ID.reset(token)
