"""Точки входа процессов, которые бот и лаунчер запускают отдельно (`python -m <модуль>`).

Отдельные процессы — фоновые задачи (бэктест, обучение и тюнинг SAC, отчёты),
воркер TimesFM в своём окружении, API приложения, импорт ключей. Запуск — только
по имени модуля из этого реестра: путь к файлу не переживает переезд кода и не
работает в Docker, где пакет установлен в site-packages. Тест
tests/test_entrypoints.py проверяет, что каждый модуль находится и запускаем.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

API = "bitbot.api"
SUPERVISOR = "bitbot.workers.supervisor"
BACKTEST = "bitbot.research.backtest"
ALPHA_REPORT = "bitbot.research.paper_alpha_report"
LIVE_DIVERGENCE = "bitbot.research.backtest_live_divergence"
AI_ISOLATION = "bitbot.research.ai_signal_isolation_backtest"
TRAIN_SAC = "bitbot.ml.train_sac"
TRAIN_SAC_V2 = "bitbot.ml.train_sac_v2"
TUNE_SAC = "bitbot.ml.tune_sac"
TIMESFM_WORKER = "bitbot.ai.forecast.timesfm_worker"
SECRETS_IMPORT = "bitbot.config.secrets_import"

ALL = (
    API,
    SUPERVISOR,
    BACKTEST,
    ALPHA_REPORT,
    LIVE_DIVERGENCE,
    AI_ISOLATION,
    TRAIN_SAC,
    TRAIN_SAC_V2,
    TUNE_SAC,
    TIMESFM_WORKER,
    SECRETS_IMPORT,
)


def package_root() -> str:
    """Папка, в которой лежит пакет bitbot, — для PYTHONPATH подпроцессов
    (воркер TimesFM живёт в своём окружении, где bitbot не установлен)."""
    return str(Path(__file__).resolve().parents[2])  # backend/


def python_m(module: str, python: str = sys.executable) -> List[str]:
    """Команда запуска модуля тем же (или указанным) интерпретатором."""
    return [python, "-m", module]
