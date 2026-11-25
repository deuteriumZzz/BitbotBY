"""Decision Ledger v0: постоянный журнал решений (append-only JSONL).

Ничего не решает и не меняет поведение бота. Пишет то, что раньше не
сохранялось (AUDIT_REPORT H-2, H-5): какой AI-провайдер фактически ответил,
сырой ответ модели, цепочку отказов, сигналы по стадиям уверенности и итог
исполнения. Нужен, чтобы позже оценивать AI по провайдерам и воспроизводить
решения.

Формат строки: {"ts": ISO-UTC, "kind": str, "trace_id": str, ...поля}.
Запись никогда не бросает исключений — журнал не должен ронять торговый цикл.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, List

from bitbot.infra.logger import TRACE_ID

logger = logging.getLogger(__name__)

_RAW_LIMIT = 8_000  # символов сырого ответа модели на запись


class DecisionLedger:
    """Потокобезопасный append-only журнал с одной ротацией по размеру."""

    def __init__(
        self,
        path: str | None = None,
        max_bytes: int | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.path = path or os.getenv(
            "DECISION_LEDGER_PATH", "data/decision_ledger.jsonl"
        )
        self.max_bytes = (
            max_bytes
            if max_bytes is not None
            else int(os.getenv("DECISION_LEDGER_MAX_BYTES", str(50 * 1024 * 1024)))
        )
        self.enabled = (
            enabled
            if enabled is not None
            else os.getenv("DECISION_LEDGER_ENABLED", "1") != "0"
        )
        self.trace_id = ""
        self._lock = threading.Lock()
        # Необязательный приёмник копий записей (шина событий): компактная отметка в Redis Stream.
        self.sink: Any = None

    def begin_cycle(self, cycle: int) -> str:
        """Новый trace_id на торговый цикл; его получают все записи цикла и
        все строки лога цикла (`infra.logger.TRACE_ID`)."""
        self.trace_id = f"{int(datetime.now(timezone.utc).timestamp())}-{cycle}"
        TRACE_ID.set(self.trace_id)
        return self.trace_id

    def write(self, kind: str, **fields: Any) -> None:
        """Дописывает запись. Не бросает исключений."""
        if not self.enabled:
            return
        try:
            raw = fields.get("raw")
            if isinstance(raw, str) and len(raw) > _RAW_LIMIT:
                fields["raw"] = raw[:_RAW_LIMIT]
                fields["raw_truncated"] = True
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "kind": kind,
                "trace_id": fields.pop("trace_id", self.trace_id),
                **fields,
            }
            line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
            with self._lock:
                os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
                self._rotate_if_needed()
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line)
            if self.sink is not None:
                try:
                    self.sink(kind, record)
                except Exception as exc:  # приёмник не должен ломать запись журнала
                    logger.debug("ledger sink failed: %s", exc)
        except Exception as exc:  # журнал не должен ломать торговлю
            logger.warning("DecisionLedger write failed (%s): %s", kind, exc)

    def _rotate_if_needed(self) -> None:
        try:
            if os.path.getsize(self.path) <= self.max_bytes:
                return
        except OSError:
            return  # файла ещё нет
        try:
            # Прежний .1 не затираем: из истории учится мета-модель, ей нужны все сделки.
            # ponytail: архив без сжатия (переименование мгновенно, цикл не ждёт); gzip — если диск станет тесным
            if os.path.exists(self.path + ".1"):
                arch = archive_dir(self.path)
                os.makedirs(arch, exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
                os.replace(self.path + ".1", os.path.join(arch, f"{stamp}.jsonl"))
            os.replace(self.path, self.path + ".1")
        except OSError as exc:
            logger.warning("DecisionLedger rotation failed: %s", exc)


def archive_dir(path: str) -> str:
    """Куда уходят старые части журнала: `data/decision_ledger_archive/`."""
    root, _ = os.path.splitext(os.path.abspath(path))
    return root + "_archive"


def ledger_files(path: str) -> List[str]:
    """Все части журнала от старых к новым: архив, `.1`, текущий файл."""
    arch = archive_dir(path)
    try:
        names = sorted(n for n in os.listdir(arch) if n.endswith(".jsonl"))
    except OSError:
        names = []
    old = [os.path.join(arch, n) for n in names]
    return old + [p for p in (path + ".1", path) if os.path.exists(p)]
