"""Реестр моделей SAC и жизненный цикл допуска (Phase 7, MASTER_SPEC §3.3, §9.5).

Стадии: candidate → validated → paper → shadow → canary → production.
Пропускать стадии нельзя; каждый переход требует доказательств (evidence), а
часть переходов — явного одобрения человека (`approved_by`). Файл модели, не
прошедший реестр (стадия `unregistered`), торговать не может: исполнять заявки
разрешено только на стадиях из `EXECUTABLE` — в Paper это paper/canary/production,
в Live только production.

Что значит «canary» пока: ограничение риска для канарейки (доля капитала) в боте
**не реализовано**, поэтому в Live канарейка не исполняется; стадия нужна как
отметка о пройденной проверке и шаг перед production.

Пороги `MIN_DAYS` — предложение (из обсуждения: 1–2 недели на стадию), их
подтверждает владелец. Реестр — один JSON-файл с атомарной записью.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

STAGES = ("candidate", "validated", "paper", "shadow", "canary", "production")
ENDED = ("rejected", "retired")
UNREGISTERED = "unregistered"

EXECUTABLE = {
    "paper": frozenset({"paper", "canary", "production"}),
    "live": frozenset({"production"}),
}

MIN_DAYS = {"paper": 14, "shadow": 14, "canary": 14}  # предложение, не константа
MIN_WF_FOLDS = 3


class PromotionError(ValueError):
    """Переход не разрешён: не хватает доказательств или нарушен порядок стадий."""


def default_registry_path(model_path: str) -> str:
    """Реестр лежит рядом с файлами моделей."""
    return os.path.join(os.path.dirname(os.path.abspath(model_path)), "registry.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _requirements(src: str, dst: str, ev: Dict[str, Any]) -> List[str]:
    """Что не выполнено для перехода src → dst (пустой список = можно)."""
    problems: List[str] = []

    def need(cond: bool, text: str) -> None:
        if not cond:
            problems.append(text)

    if (src, dst) == ("candidate", "validated"):
        wf = ev.get("walk_forward") or {}
        need(bool(wf.get("passed")), "walk_forward.passed")
        need(int(wf.get("n_folds", 0)) >= MIN_WF_FOLDS, f"walk_forward.n_folds>={MIN_WF_FOLDS}")
        need(bool(ev.get("dataset_version")), "dataset_version")
    else:
        need(bool(ev.get("approved_by")), "approved_by")
        key = {"paper": "paper_days", "shadow": "shadow_days", "canary": "canary_days"}.get(src)
        if key:  # выходя из стадии, нужно отработать на ней минимум дней
            need(float(ev.get(key, 0)) >= MIN_DAYS[src], f"{key}>={MIN_DAYS[src]}")
    return problems


class ModelRegistry:
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()

    # ── хранение ──────────────────────────────────────────────────────────────

    def _load(self) -> Dict[str, Any]:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) and "models" in data else {"models": {}}
        except (OSError, ValueError):
            return {"models": {}}

    def _save(self, data: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, self.path)  # атомарно: читатель не увидит половину файла

    def mtime(self) -> float:
        try:
            return os.path.getmtime(self.path)
        except OSError:
            return 0.0

    # ── операции ──────────────────────────────────────────────────────────────

    def register(
        self,
        model_id: str,
        model_path: str,
        *,
        dataset_version: str = "",
        feature_version: str = "",
        params: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        parent: str = "",
    ) -> Dict[str, Any]:
        with self._lock:
            data = self._load()
            if model_id in data["models"]:
                raise PromotionError(f"модель {model_id} уже зарегистрирована")
            entry = {
                "model_id": model_id,
                "path": os.path.abspath(model_path),
                "stage": "candidate",
                "created": _now(),
                "dataset_version": dataset_version,
                "feature_version": feature_version,
                "params": params or {},
                "metrics": metrics or {},
                "parent": parent,
                "history": [],
            }
            data["models"][model_id] = entry
            self._save(data)
            return entry

    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self._load()["models"].get(model_id)

    def find_by_path(self, model_path: str) -> Optional[Dict[str, Any]]:
        target = os.path.abspath(model_path)
        found = [m for m in self._load()["models"].values() if m["path"] == target]
        return found[-1] if found else None

    def stage_of_path(self, model_path: str) -> str:
        entry = self.find_by_path(model_path)
        return entry["stage"] if entry else UNREGISTERED

    def is_executable(self, model_path: str, *, paper: bool) -> bool:
        return self.stage_of_path(model_path) in EXECUTABLE["paper" if paper else "live"]

    def promote(self, model_id: str, evidence: Dict[str, Any], by: str) -> Dict[str, Any]:
        """Шаг на следующую стадию при выполненных требованиях."""
        with self._lock:
            data = self._load()
            entry = data["models"].get(model_id)
            if entry is None:
                raise PromotionError(f"нет модели {model_id}")
            src = entry["stage"]
            if src not in STAGES or src == STAGES[-1]:
                raise PromotionError(f"из стадии {src} повышать некуда")
            dst = STAGES[STAGES.index(src) + 1]
            problems = _requirements(src, dst, evidence)
            if problems:
                raise PromotionError(f"{src}→{dst}: не выполнено: {', '.join(problems)}")
            if dst == "production":  # прежняя production-версия уходит в резерв для отката
                for other in data["models"].values():
                    if other["stage"] == "production":
                        self._move(other, "retired", {"reason": f"заменена {model_id}"}, by)
                        other["retired_from"] = "production"
            self._move(entry, dst, evidence, by)
            self._save(data)
            return entry

    def demote(self, model_id: str, to_stage: str, reason: str, by: str) -> Dict[str, Any]:
        """Понижение или отклонение: всегда разрешено, причина обязательна."""
        if not reason:
            raise PromotionError("причина обязательна")
        if to_stage not in STAGES + ENDED:
            raise PromotionError(f"неизвестная стадия {to_stage}")
        with self._lock:
            data = self._load()
            entry = data["models"].get(model_id)
            if entry is None:
                raise PromotionError(f"нет модели {model_id}")
            self._move(entry, to_stage, {"reason": reason}, by)
            self._save(data)
            return entry

    def rollback(self, reason: str, by: str) -> Optional[Dict[str, Any]]:
        """Текущая production → retired, последняя ушедшая из production → production."""
        with self._lock:
            data = self._load()
            models = list(data["models"].values())
            current = next((m for m in models if m["stage"] == "production"), None)
            previous = [m for m in models if m.get("retired_from") == "production"]
            if current is None or not previous:
                return None
            target = max(previous, key=lambda m: m["history"][-1]["ts"])
            self._move(current, "retired", {"reason": f"откат: {reason}"}, by)
            current["retired_from"] = "production"
            target.pop("retired_from", None)
            self._move(target, "production", {"reason": f"откат: {reason}"}, by)
            self._save(data)
            return target

    @staticmethod
    def _move(entry: Dict[str, Any], dst: str, evidence: Dict[str, Any], by: str) -> None:
        entry["history"].append(
            {"ts": _now(), "from": entry["stage"], "to": dst, "evidence": evidence, "by": by}
        )
        entry["stage"] = dst
