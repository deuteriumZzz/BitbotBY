"""Каталог фоновых задач и их состояние — то, что видит приложение.

Запускает задачи `bitbot.workers.jobs.JobRunner`; API читает отсюда список и
статус, не завися от воркеров.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class JobSpec:
    id: str
    group: str  # задачи одной группы не идут одновременно
    title: str
    ttl: int  # страховка замка на случай падения бота, секунды


JOBS: Dict[str, JobSpec] = {
    s.id: s
    for s in (
        JobSpec("backtest", "backtest", "Бэктест стратегий", 3 * 3600),
        JobSpec("train", "sac", "Обучение SAC", 4 * 3600),
        JobSpec("tune", "sac", "Тюнинг и обучение SAC", 8 * 3600),
        JobSpec("alpha_report", "alpha_report", "Значимость альфы", 1800),
        JobSpec("live_divergence", "live_divergence", "Live vs бэктест", 1800),
        JobSpec("ai_isolation", "ai_isolation", "AI без индикаторов", 3600),
    )
}
GROUPS = sorted({j.group for j in JOBS.values()})


def status(rc: Any) -> Dict[str, Any]:
    """Что идёт сейчас, прогресс и можно ли откатить модель (читает Redis)."""
    backup = rc.get_sac_backup_path()
    return {
        "profiles": rc.get_training_profiles(),
        "backup_profile": rc.get_sac_backup_profile() or None,
        "running": rc.running_jobs(GROUPS),
        "progress": {
            "backtest": rc.get_backtest_progress(),
            "train": rc.get_train_progress(),
            "tune": rc.get_tune_progress(),
        },
        "rollback_available": bool(backup and os.path.exists(backup)),
    }
