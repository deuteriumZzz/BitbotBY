"""Стадия выкатки системы (Phase 11): Shadow → Paper → Canary → Production, переход явный.

Относится ко всей системе (в отличие от `model_registry`, где стадии у каждой
модели). Повышение — только на следующую стадию и только с автором и
доказательствами; понижение (в т.ч. аварийное) — на любую более раннюю стадию,
с причиной. `capabilities` говорит, что стадия разрешает; `canary` разрешает
реальные ордера только с ограничением доли капитала (`CANARY_CAPITAL_FRACTION`,
предложение 5%) — само ограничение должен применять Risk Gate/исполнитель, здесь
только значение. Подключение к исполнителю не выполнено.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

STAGES = ("shadow", "paper", "canary", "production")
CANARY_CAPITAL_FRACTION = 0.05  # предложение владельцу

_CAPS = {
    "shadow": {"real_orders": False, "paper_orders": False, "capital_fraction": 0.0},
    "paper": {"real_orders": False, "paper_orders": True, "capital_fraction": 0.0},
    "canary": {"real_orders": True, "paper_orders": True, "capital_fraction": CANARY_CAPITAL_FRACTION},
    "production": {"real_orders": True, "paper_orders": True, "capital_fraction": 1.0},
}


class RolloutError(ValueError):
    pass


def capabilities(stage: str) -> Dict[str, Any]:
    if stage not in _CAPS:
        raise RolloutError(f"неизвестная стадия {stage}")
    return dict(_CAPS[stage])


class Rollout:
    """Стадия в файле; без файла — самая безопасная (`shadow`)."""

    def __init__(self, path: str) -> None:
        self.path = path

    def _read(self) -> Dict[str, Any]:
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("stage") in STAGES:
                return d
        except (OSError, ValueError):
            pass
        return {"stage": "shadow", "history": []}

    @property
    def stage(self) -> str:
        return self._read()["stage"]

    def history(self) -> List[Dict[str, Any]]:
        return self._read()["history"]

    def _move(self, dst: str, by: str, note: str, evidence: Dict[str, Any]) -> str:
        d = self._read()
        d["history"].append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "from": d["stage"],
                "to": dst,
                "by": by,
                "note": note,
                "evidence": evidence,
            }
        )
        d["stage"] = dst
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, self.path)
        return dst

    def promote(self, by: str, evidence: Dict[str, Any]) -> str:
        """Шаг вверх; требует автора и непустые доказательства."""
        if not by:
            raise RolloutError("нужен автор перехода")
        if not evidence:
            raise RolloutError("нужны доказательства перехода")
        i = STAGES.index(self.stage)
        if i == len(STAGES) - 1:
            raise RolloutError("выше production стадий нет")
        return self._move(STAGES[i + 1], by, "promote", evidence)

    def demote(self, to: str, by: str, reason: str) -> str:
        if not by or not reason:
            raise RolloutError("нужны автор и причина")
        if to not in STAGES or STAGES.index(to) >= STAGES.index(self.stage):
            raise RolloutError("понижать можно только на более раннюю стадию")
        return self._move(to, by, reason, {})
