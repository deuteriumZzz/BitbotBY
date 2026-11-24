"""Шина событий на Redis Streams (реестр идей №2; решение — ADR-005).

Дополняет, а не заменяет: pub/sub-канал `bot:events` остаётся для живых уведомлений приложению
(доставка «сейчас или никогда»), а Stream — для потребителей, которым нужна ДОСТАВКА С ПОДТВЕРЖДЕНИЕМ
и повторное чтение (группы потребителей, `XACK`, догоняющее чтение после падения). Kafka не нужна:
один хост, один Redis, поток — события решений и исполнения, а не тики.

* `publish` не бросает: события не должны ломать торговлю; длина потока ограничена (`maxlen`, ~).
* К Ledger подключается как необязательный `sink` (`EVENT_STREAM_ENABLED`, по умолчанию выключено):
  в поток идёт компактная отметка (тип, trace_id, decision_id), полные данные остаются в журнале.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from src.infra import fastjson

STREAM_KEY = "bot:stream:events"


def _t(x: Any) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


def _parse(entries: Any) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    for eid, fields in entries or []:
        f = {_t(k): v for k, v in fields.items()}
        try:
            data = fastjson.loads(f.get("data", "{}"))
        except ValueError:
            data = {}
        out.append(
            (
                _t(eid),
                {
                    "type": _t(f.get("type", "")),
                    "ts": float(_t(f.get("ts", "0"))),
                    "data": data,
                },
            )
        )
    return out


class EventStream:
    def __init__(self, r: Any, key: str = STREAM_KEY, maxlen: int = 10_000) -> None:
        self.r, self.key, self.maxlen = r, key, maxlen

    def publish(
        self, type_: str, data: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        if self.r is None:
            return None
        try:
            eid = self.r.xadd(
                self.key,
                {
                    "type": type_,
                    "ts": f"{time.time():.6f}",
                    "data": fastjson.dumps(data or {}),
                },
                maxlen=self.maxlen,
                approximate=True,
            )
            return _t(eid)
        except Exception:  # noqa: BLE001 — шина не должна ломать торговый цикл
            return None

    def read(
        self, last_id: str = "0-0", count: int = 100, block_ms: Optional[int] = None
    ) -> List[Tuple[str, Dict[str, Any]]]:
        res = self.r.xread({self.key: last_id}, count=count, block=block_ms)
        return _parse(res[0][1]) if res else []

    def ensure_group(self, group: str) -> None:
        try:
            self.r.xgroup_create(self.key, group, id="0", mkstream=True)
        except Exception as exc:  # noqa: BLE001
            if "BUSYGROUP" not in str(exc):  # группа уже есть — штатно
                raise

    def consume(
        self, group: str, consumer: str, count: int = 10, block_ms: Optional[int] = None
    ) -> List[Tuple[str, Dict[str, Any]]]:
        res = self.r.xreadgroup(
            group, consumer, {self.key: ">"}, count=count, block=block_ms
        )
        return _parse(res[0][1]) if res else []

    def ack(self, group: str, ids: List[str]) -> int:
        return int(self.r.xack(self.key, group, *ids)) if ids else 0

    def ledger_sink(self, kind: str, record: Dict[str, Any]) -> None:
        """Подключается как `DecisionLedger.sink`: компактная отметка о записи журнала."""
        self.publish(
            f"ledger.{kind}",
            {
                "trace_id": record.get("trace_id"),
                "decision_id": record.get("decision_id"),
                "symbol": record.get("symbol"),
            },
        )
