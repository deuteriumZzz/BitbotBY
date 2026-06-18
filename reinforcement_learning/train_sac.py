"""
Обучение SAC-агента (Soft Actor-Critic) на исторических данных торговли.

Сохраняет модель, нормализационную статистику и резервные копии.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Путь к сохраняемой модели
MODEL_PATH = "models/sac_model.zip"
# Количество шагов обучения
TOTAL_TIMESTEPS = 100_000


def _backup_existing_model(path: str) -> None:
    """
    Создаёт резервную копию модели перед перезаписью.

    Если файл существует — копирует его с timestamp-суффиксом
    вида sac_model_20240101_120000.zip.

    :param path: Путь к файлу модели (с .zip или без).
    """
    zip_path = path if path.endswith(".zip") else path + ".zip"
    if not os.path.exists(zip_path):
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = zip_path.replace(".zip", f"_{ts}.zip")
    shutil.copy2(zip_path, backup)
    logger.info(f"Резервная копия модели → {backup}")


def _save_norm_stats(
    model_path: str, stats: Dict[str, Any]
) -> None:
    """
    Сохраняет статистику нормализации среды в JSON-файл.

    Файл сохраняется рядом с моделью: sac_model_norm_stats.json.

    :param model_path: Путь к файлу модели (.zip).
    :param stats: Словарь {column: [mean, std]}.
    """
    norm_path = model_path.replace(".zip", "_norm_stats.json")
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Норм-статистика сохранена → {norm_path}")


def _compute_norm_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Вычисляет среднее и стандартное отклонение для числовых колонок.

    :param df: DataFrame с данными обучающей выборки.
    :return: Словарь {column: [mean, std]} для всех числовых колонок.
    """
    stats: Dict[str, Any] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        mu = float(df[col].mean())
        sd = float(df[col].std())
        if sd > 0:
            stats[col] = [mu, sd]
    return stats


def train(
    df: pd.DataFrame,
    model_path: str = MODEL_PATH,
    total_timesteps: int = TOTAL_TIMESTEPS,
    save_backup: bool = True,
) -> Optional[str]:
    """
    Обучает SAC-агента на переданных данных и сохраняет модель.

    Алгоритм:
    1. Создаёт TradingEnv и оборачивает в Monitor.
    2. Инициализирует SAC из stable-baselines3.
    3. Запускает model.learn(total_timesteps).
    4. Делает резервную копию старой модели (если save_backup=True).
    5. Сохраняет модель и norm_stats рядом.

    :param df: DataFrame с OHLCV и индикаторами.
    :param model_path: Путь для сохранения модели.
    :param total_timesteps: Количество шагов обучения.
    :param save_backup: Создавать ли резервную копию старой модели.
    :return: Путь к сохранённой модели или None при ошибке.
    :raises ImportError: Если stable-baselines3 не установлен.
    """
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.monitor import Monitor
    except ImportError as e:
        logger.error(
            f"stable-baselines3 не установлен: {e}"
        )
        raise

    from reinforcement_learning.rl_env import TradingEnv

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    logger.info(
        f"Обучение SAC: {total_timesteps} шагов, "
        f"данных {len(df)} свечей"
    )

    try:
        env = Monitor(TradingEnv(df))
        model = SAC("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=total_timesteps)
    except Exception as e:
        logger.error(
            f"Ошибка обучения SAC: {e}", exc_info=True
        )
        return None

    # Резервная копия перед перезаписью
    if save_backup:
        _backup_existing_model(model_path)

    # Сохранение модели
    save_path = model_path.replace(".zip", "")
    model.save(save_path)
    logger.info(f"Модель SAC сохранена → {save_path}.zip")

    # Сохранение нормализационной статистики
    norm_stats = _compute_norm_stats(df)
    _save_norm_stats(model_path, norm_stats)

    return model_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    from src.data_loader import DataLoader

    async def _run() -> None:
        import asyncio
        loader = DataLoader()
        await loader.initialize(
            api_key=os.getenv("BYBIT_API_KEY", ""),
            api_secret=os.getenv("BYBIT_API_SECRET", ""),
        )
        df = await loader.get_paginated_history(
            "BTC/USDT", "15m", months=6
        )
        await loader.close()
        train(df)

    import asyncio
    asyncio.run(_run())
