"""Emergency Control (Phase 11, MASTER_SPEC §9.6): исполнитель четырёх уровней аварии.

Уровни совпадают с circuit breaker Risk Gate: 1 запрет входов · 2 отмена
отложенных ордеров · 3 сокращение/закрытие по правилам · 4 полная остановка.
Модуль НИЧЕГО не делает сам: действия передаются снаружи вызываемыми
объектами (в боте — `RuntimeConfig.set_paused`, отмена ордеров, закрытие в
`PositionMonitor`). Уровень только растёт; понижение — явная операция `reset`
с указанием, кто и почему. Состояние хранится в файле, чтобы пережить рестарт.
Подключение к боту/Telegram не выполнено.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

LEVELS = {1: "block_entries", 2: "cancel_orders", 3: "reduce_positions", 4: "full_stop"}


class EmergencyControl:
    def __init__(self, path: str, actions: Dict[str, Callable[[], Any]]) -> None:
        missing = [name for name in LEVELS.values() if name not in actions]
        if missing:
            raise ValueError("не заданы действия: " + ", ".join(missing))
        self.path, self.actions = path, actions

    # ── состояние ─────────────────────────────────────────────────────────────
    def state(self) -> Dict[str, Any]:
        try:
            with open(self.path, encoding="utf-8") as f:
                s = json.load(f)
            return s if isinstance(s, dict) and "level" in s else {"level": 0}
        except (OSError, ValueError):
            return {"level": 0}

    @property
    def level(self) -> int:
        return int(self.state().get("level", 0))

    def _save(self, level: int, by: str, reason: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {"level": level, "by": by, "reason": reason, "ts": datetime.now(timezone.utc).isoformat()},
                f,
                ensure_ascii=False,
            )
        os.replace(tmp, self.path)

    # ── управление ────────────────────────────────────────────────────────────
    def escalate(self, level: int, by: str, reason: str) -> Dict[str, Any]:
        """Поднимает уровень и выполняет действия НОВЫХ уровней по порядку.

        Ошибка действия не прерывает остальные (авария важнее аккуратности) и
        возвращается в отчёте; уровень фиксируется до выполнения, чтобы сбой
        на полпути не оставил систему «как будто без аварии».
        """
        if level not in LEVELS:
            raise ValueError("уровень 1–4")
        if not by or not reason:
            raise ValueError("нужны автор и причина")
        current = self.level
        if level <= current:
            return {"level": current, "executed": [], "errors": {}, "noop": True}
        self._save(level, by, reason)
        executed: List[str] = []
        errors: Dict[str, str] = {}
        for lv in range(current + 1, level + 1):
            name = LEVELS[lv]
            try:
                self.actions[name]()
                executed.append(name)
            except Exception as exc:  # noqa: BLE001
                errors[name] = f"{type(exc).__name__}: {exc}"
        return {"level": level, "executed": executed, "errors": errors, "noop": False}

    def reset(self, by: str, reason: str) -> None:
        """Снятие аварии — только явно; сами действия (например, закрытие) не откатываются."""
        if not by or not reason:
            raise ValueError("нужны автор и причина")
        self._save(0, by, reason)

    def entries_allowed(self) -> bool:
        return self.level < 1
