"""Emergency Control: исполнитель четырёх уровней аварии.

Уровни совпадают с circuit breaker Risk Gate: 1 запрет входов · 2 отмена
отложенных ордеров · 3 сокращение/закрытие по правилам · 4 полная остановка.
Модуль НИЧЕГО не делает сам: действия передаются снаружи вызываемыми
объектами (в боте — `RuntimeConfig.set_paused`, отмена ордеров, закрытие в
`PositionMonitor`). Уровень только растёт; понижение — явная операция `reset`
с указанием, кто и почему. Состояние хранится в файле, чтобы пережить рестарт.
Действия подключает бот (backend/bitbot/workers/ops.py); управление — из приложения.
"""

from __future__ import annotations

import inspect
import json
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

DEFAULT_PATH = os.path.join("data", "emergency.json")

LEVELS = {1: "block_entries", 2: "cancel_orders", 3: "reduce_positions", 4: "full_stop"}
# Что делает каждый уровень — для подтверждения в приложении.
LEVEL_TEXT = {
    1: "запрет новых входов (пауза)",
    2: "отмена отложенных ордеров (входные ордера рыночные, биржевые SL/TP не снимаются — ничего не сделает)",
    3: "закрыть убыточные позиции, у прибыльных SL на безубыток",
    4: "закрыть ВСЕ позиции и оставить паузу",
}


def read_state(path: str) -> Dict[str, Any]:
    """Состояние из файла без процесса бота: {level, by, reason, ts}; нет файла — уровень 0."""
    try:
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        return s if isinstance(s, dict) and "level" in s else {"level": 0}
    except (OSError, ValueError):
        return {"level": 0}


def plan(current: int, action: str) -> Dict[str, Any]:
    """Что произойдёт по действию `action` ("1"…"4" или "reset") при уровне `current`.

    Ответ в форме шины команд: {"ok": False, "error": …} — нельзя; иначе
    {"ok": True, "reset": True} или {"ok": True, "level": N, "steps": [...]}.
    Уровень только растёт: понизить можно лишь сбросом.
    """
    act = str(action).strip().lower()
    if act == "reset":
        return {"ok": True, "reset": True}
    if not act.isdigit() or int(act) not in LEVELS:
        return {
            "ok": False,
            "error": {"code": "bad_level", "message": "Уровень — 1…4 или сброс."},
        }
    level = int(act)
    if level <= current:
        return {
            "ok": False,
            "error": {
                "code": "level_active",
                "message": f"Уровень {current} уже активен — понизить можно только сбросом.",
            },
        }
    steps = [f"{lv}. {LEVEL_TEXT[lv]}" for lv in range(current + 1, level + 1)]
    return {"ok": True, "level": level, "steps": steps}


class EmergencyControl:
    def __init__(self, path: str, actions: Dict[str, Callable[[], Any]]) -> None:
        missing = [name for name in LEVELS.values() if name not in actions]
        if missing:
            raise ValueError("не заданы действия: " + ", ".join(missing))
        self.path, self.actions = path, actions

    # ── состояние ─────────────────────────────────────────────────────────────
    def state(self) -> Dict[str, Any]:
        return read_state(self.path)

    @property
    def level(self) -> int:
        return int(self.state().get("level", 0))

    def _save(self, level: int, by: str, reason: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "level": level,
                    "by": by,
                    "reason": reason,
                    "ts": datetime.now(timezone.utc).isoformat(),
                },
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
        current = self._prepare(level, by, reason)
        if current is None:
            return {"level": self.level, "executed": [], "errors": {}, "noop": True}
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

    async def aescalate(self, level: int, by: str, reason: str) -> Dict[str, Any]:
        """То же, что `escalate`, но действия могут быть корутинами (действия бота асинхронны)."""
        current = self._prepare(level, by, reason)
        if current is None:
            return {"level": self.level, "executed": [], "errors": {}, "noop": True}
        executed: List[str] = []
        errors: Dict[str, str] = {}
        for lv in range(current + 1, level + 1):
            name = LEVELS[lv]
            try:
                res = self.actions[name]()
                if inspect.isawaitable(res):
                    await res
                executed.append(name)
            except Exception as exc:  # noqa: BLE001
                errors[name] = f"{type(exc).__name__}: {exc}"
        return {"level": level, "executed": executed, "errors": errors, "noop": False}

    def _prepare(self, level: int, by: str, reason: str) -> Optional[int]:
        """Проверка аргументов и фиксация уровня ДО действий; None — повышать нечего."""
        if level not in LEVELS:
            raise ValueError("уровень 1–4")
        if not by or not reason:
            raise ValueError("нужны автор и причина")
        current = self.level
        if level <= current:
            return None
        self._save(level, by, reason)
        return current

    def reset(self, by: str, reason: str) -> None:
        """Снятие аварии — только явно; сами действия (например, закрытие) не откатываются."""
        if not by or not reason:
            raise ValueError("нужны автор и причина")
        self._save(0, by, reason)

    def entries_allowed(self) -> bool:
        return self.level < 1
