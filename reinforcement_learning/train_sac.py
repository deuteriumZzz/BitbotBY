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
# Количество шагов обучения (переопределяется env-переменной TOTAL_TIMESTEPS)
TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", "500000"))
# Доля данных для обучения (остаток — тестовая выборка)
TRAIN_SPLIT = 0.8


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


def _save_norm_stats(model_path: str, stats: Dict[str, Any]) -> None:
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


def _evaluate_model(
    model: Any,
    test_df: pd.DataFrame,
    initial_balance: float = 10000.0,
) -> float:
    """
    Прогоняет обученную модель на тестовой выборке (детерминированный инференс).

    :param model: Обученная SAC-модель.
    :param test_df: Тестовый DataFrame (out-of-sample).
    :param initial_balance: Начальный баланс.
    :return: Итоговая стоимость портфеля.
    """
    from reinforcement_learning.rl_env import TradingEnv

    env = TradingEnv(test_df, initial_balance=initial_balance)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)

    pnl_pct = (env.current_value / initial_balance - 1.0) * 100.0
    bh_final = initial_balance * (
        float(test_df.iloc[-1]["close"]) / float(test_df.iloc[0]["close"])
    )
    bh_pct = (bh_final / initial_balance - 1.0) * 100.0
    logger.info(
        "TEST SET (%d candles) — "
        "SAC: $%.2f (%+.1f%%) | "
        "Buy&Hold: $%.2f (%+.1f%%) | "
        "Commissions: $%.2f",
        len(test_df),
        env.current_value,
        pnl_pct,
        bh_final,
        bh_pct,
        env.total_commission,
    )
    return env.current_value


def train(
    df: pd.DataFrame,
    model_path: str = MODEL_PATH,
    total_timesteps: int = TOTAL_TIMESTEPS,
    train_split: float = TRAIN_SPLIT,
    save_backup: bool = True,
) -> Optional[str]:
    """
    Обучает SAC-агента на исторических данных и сохраняет модель.

    Алгоритм:
    1. Разделяет df на train (80%) и test (20%) по времени.
    2. Создаёт TradingEnv на train-части, оборачивает в Monitor.
    3. Инициализирует SAC из stable-baselines3.
    4. Запускает model.learn(total_timesteps).
    5. Оценивает модель на test-части (out-of-sample).
    6. Сохраняет модель и norm_stats (на основе train-данных).

    :param df: DataFrame с OHLCV и индикаторами (полный датасет).
    :param model_path: Путь для сохранения модели.
    :param total_timesteps: Количество шагов обучения.
    :param train_split: Доля данных для обучения (0.8 = 80%).
    :param save_backup: Создавать ли резервную копию старой модели.
    :return: Путь к сохранённой модели или None при ошибке.
    :raises ImportError: Если stable-baselines3 не установлен.
    """
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.monitor import Monitor
    except ImportError as e:
        logger.error(f"stable-baselines3 не установлен: {e}")
        raise

    from reinforcement_learning.rl_env import TradingEnv

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Train / test split — строго по времени, без перемешивания
    split_idx = int(len(df) * train_split)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    logger.info(
        "Данных всего: %d свечей | Train: %d (%.0f%%) | Test: %d (%.0f%%)",
        len(df),
        len(train_df),
        train_split * 100,
        len(test_df),
        (1 - train_split) * 100,
    )
    logger.info("Обучение SAC: %d шагов", total_timesteps)

    try:
        env = Monitor(TradingEnv(train_df))
        model = SAC("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=total_timesteps)
    except Exception as e:
        logger.error(f"Ошибка обучения SAC: {e}", exc_info=True)
        return None

    # Оценка на тестовой выборке (out-of-sample)
    _evaluate_model(model, test_df)

    # Резервная копия перед перезаписью
    if save_backup:
        _backup_existing_model(model_path)

    # Сохранение модели (norm_stats считаем на train-данных)
    save_path = model_path.replace(".zip", "")
    model.save(save_path)
    logger.info(f"Модель SAC сохранена → {save_path}.zip")

    norm_stats = _compute_norm_stats(train_df)
    _save_norm_stats(model_path, norm_stats)

    return model_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    from src.data_loader import DataLoader

    async def _run() -> None:
        loader = DataLoader()
        await loader.initialize(
            api_key=os.getenv("BYBIT_API_KEY", ""),
            api_secret=os.getenv("BYBIT_API_SECRET", ""),
        )
        df = await loader.get_paginated_history("BTC/USDT", "15m", months=6)
        await loader.close()
        train(df)

    import asyncio

    asyncio.run(_run())
