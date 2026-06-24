"""
Обучение SAC-агента (Soft Actor-Critic) на исторических данных торговли.

Сохраняет модель, нормализационную статистику и резервные копии.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Путь к сохраняемой модели (переопределяется env SAC_MODEL_PATH)
MODEL_PATH = os.getenv("SAC_MODEL_PATH", "models/sac_model.zip")
# Количество шагов обучения (переопределяется env-переменной TOTAL_TIMESTEPS)
TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", "500000"))
# Доля данных для обучения (остаток — тестовая выборка)
TRAIN_SPLIT = 0.8
# Walk-forward: обучаем на N месяцах, тестируем на M, сдвигаем на M
_WF_TRAIN_MONTHS = 4
_WF_STEP_MONTHS = 1
_CANDLES_PER_MONTH = 2880  # 15m * 24h * 30d
_PROGRESS_EVERY = 5000  # шагов между TRAIN_PROGRESS строками


class _ProgressCallback:
    """Minimal stable-baselines3 BaseCallback that emits TRAIN_PROGRESS lines."""

    def __init__(self, total_timesteps: int) -> None:
        self._total = total_timesteps
        self._last_report = 0
        self._start_time: float | None = None
        self.n_calls = 0
        self.num_timesteps = 0
        self.model = None
        self.training_env = None
        self.parent = None
        self.verbose = 0
        self.locals: dict = {}
        self.globals: dict = {}
        self.logger = None

    def init_callback(self, model) -> None:  # noqa: ANN001
        import time

        self.model = model
        self._start_time = time.time()

    def on_training_start(self, locals_: dict, globals_: dict) -> None:
        import time

        self._start_time = time.time()

    def on_step(self) -> bool:
        import time

        step = self.model.num_timesteps if self.model else self.num_timesteps
        if step - self._last_report >= _PROGRESS_EVERY:
            self._last_report = step
            pct = round(step / self._total * 100, 1)
            elapsed = time.time() - (self._start_time or time.time())
            eta_min = (
                int((elapsed / step) * (self._total - step) / 60) if step > 0 else None
            )
            data = {
                "pct": pct,
                "step": step,
                "total": self._total,
                "eta_min": eta_min,
            }
            print(f"TRAIN_PROGRESS:{json.dumps(data)}", file=sys.stdout, flush=True)
        return True

    def on_training_end(self) -> None:
        data = {
            "pct": 100.0,
            "step": self._total,
            "total": self._total,
            "eta_min": 0,
        }
        print(f"TRAIN_PROGRESS:{json.dumps(data)}", file=sys.stdout, flush=True)

    def on_rollout_start(self) -> None:
        pass

    def on_rollout_end(self) -> None:
        pass


def _backup_existing_model(path: str) -> str:
    """
    Создаёт резервную копию модели перед перезаписью.

    :param path: Путь к файлу модели (с .zip или без).
    :return: Путь к бэкапу или пустая строка если модели не было.
    """
    zip_path = path if path.endswith(".zip") else path + ".zip"
    if not os.path.exists(zip_path):
        return ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = zip_path.replace(".zip", f"_{ts}.zip")
    shutil.copy2(zip_path, backup)
    logger.info(f"Резервная копия модели → {backup}")
    return backup


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


def _load_norm_stats(model_path: str) -> Dict[str, Any]:
    """
    Загружает статистику нормализации из JSON-файла рядом с моделью.

    :param model_path: Путь к файлу модели (.zip).
    :return: Словарь {column: [mean, std]} или пустой словарь если файл не найден.
    """
    norm_path = model_path.replace(".zip", "_norm_stats.json")
    if not os.path.exists(norm_path):
        logger.warning("norm_stats не найден: %s — нормализация пропущена", norm_path)
        return {}
    with open(norm_path, encoding="utf-8") as f:
        return json.load(f)


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
) -> Dict[str, float]:
    """
    Прогоняет обученную модель на тестовой выборке (детерминированный инференс).

    :param model: Обученная SAC-модель.
    :param test_df: Тестовый DataFrame (out-of-sample).
    :param initial_balance: Начальный баланс.
    :return: {"sac_pct": float, "bh_pct": float, "commission": float}
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
        "ТЕСТОВАЯ ВЫБОРКА (%d свечей) — "
        "SAC: $%.2f (%+.1f%%) | "
        "Buy&Hold: $%.2f (%+.1f%%) | "
        "Комиссии: $%.2f",
        len(test_df),
        env.current_value,
        pnl_pct,
        bh_final,
        bh_pct,
        env.total_commission,
    )
    return {
        "sac_pct": round(pnl_pct, 2),
        "bh_pct": round(bh_pct, 2),
        "commission": round(float(env.total_commission), 2),
    }


def _finetune_on_experiences(model: Any, norm_stats: Dict[str, Any]) -> None:
    """
    Дообучает модель на реальных сделках из data/experiences.jsonl.

    Загружает накопленные опыты, конвертирует в obs-векторы,
    добавляет в replay buffer модели и запускает gradient steps.
    Пропускается если файл пуст или содержит < 50 записей.
    """
    from src.experience_buffer import load as _load_exp

    records = _load_exp()
    if len(records) < 50:
        logger.info(
            "Experiences: %d записей — пропускаем fine-tune (нужно ≥ 50)",
            len(records),
        )
        return

    logger.info("Fine-tune на %d реальных сделках из experiences.jsonl", len(records))

    # Преобразуем каждую запись в obs-вектор (21 признак, порядок как в TradingEnv)
    # Колонки для нормализации — только первые 11 (OHLCV + индикаторы)
    _market_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "rsi",
        "macd",
        "macd_signal",
        "bb_upper",
        "bb_middle",
        "bb_lower",
    ]

    def _to_obs(rec: Dict[str, Any], use_exit: bool = False) -> np.ndarray:
        ind = rec.get("indicators", {})
        entry_price = rec.get("price", rec.get("entry_price", 0.0))
        exit_price = rec.get("exit_price", entry_price)
        price = exit_price if use_exit else entry_price
        pnl = rec.get("pnl_pct", 0.0)
        raw = [
            price,
            price,
            price,
            price,  # open/high/low/close приближённые значения
            rec.get("volume_ratio", 1.0) * 1000,  # прокси для объёма
            ind.get("rsi", 50.0),
            ind.get("macd", 0.0),
            ind.get("macd_signal", 0.0),
            ind.get("bb_upper", price * 1.02),
            ind.get("bb_middle", price),
            ind.get("bb_lower", price * 0.98),
            0.0 if use_exit else float(rec.get("action") == "buy"),
            pnl if use_exit else 0.0,
            pnl,
            # Новые фичи (14-20) — уже нормализованы, не нормализовать повторно
            ind.get("funding_rate", 0.0) * 1000.0,
            ind.get("ob_imbalance", 0.0),
            min(max(ind.get("pcr", 1.0) / 3.0, 0.0), 1.0),
            ind.get("fear_greed", 50.0) / 100.0,
            min(max(ind.get("iv_skew", 0.0) / 20.0, -1.0), 1.0),
            min(max(ind.get("basis_pct", 0.0) / 5.0, -1.0), 1.0),
            ind.get("google_trends", 50.0) / 100.0,
        ]
        obs = np.array(raw, dtype=np.float32)
        # Нормализация только для первых 11 элементов (OHLCV + индикаторы)
        for i, col in enumerate(_market_cols):
            if col in norm_stats:
                mu, sd = norm_stats[col]
                obs[i] = (obs[i] - mu) / (sd + 1e-8)
        return obs

    added = 0
    for rec in records:
        try:
            obs = _to_obs(rec, use_exit=False)
            next_obs = _to_obs(rec, use_exit=True)
            action_val = np.array(
                [1.0 if rec["action"] == "buy" else -1.0], dtype=np.float32
            )
            model.replay_buffer.add(
                obs=obs.reshape(1, -1),
                next_obs=next_obs.reshape(1, -1),
                action=action_val.reshape(1, -1),
                reward=np.array([rec.get("pnl_pct", 0.0)], dtype=np.float32),
                done=np.array([False]),
                infos=[{}],
            )
            added += 1
        except Exception as exc:
            logger.debug("Пропускаем запись опыта: %s", exc)

    if added < 10 or model.replay_buffer.size() < model.batch_size:
        logger.info("Fine-tune: недостаточно данных в буфере, пропускаем")
        return

    gradient_steps = min(added * 2, 500)
    model.train(gradient_steps=gradient_steps, batch_size=model.batch_size)
    logger.info(
        "Fine-tune завершён: %d gradient steps на %d опытах", gradient_steps, added
    )


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
    _min_candles = int(os.getenv("TRAIN_MIN_CANDLES", "2880"))
    if len(df) < _min_candles:
        raise ValueError(
            f"DataFrame too short: {len(df)} candles < {_min_candles} minimum"
        )

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.monitor import Monitor
    except ImportError as e:
        logger.error(f"stable-baselines3 не установлен: {e}")
        raise

    from reinforcement_learning.rl_env import TradingEnv

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Разделение train/test — строго по времени, без перемешивания
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

    # Загружаем лучшие гиперпараметры Optuna, если файл существует
    hp_path = "models/best_hyperparams.json"
    policy_kwargs: dict = {}
    sac_kwargs: dict = {"verbose": 1}
    if os.path.exists(hp_path):
        with open(hp_path) as _f:
            best_hp = json.load(_f)
        arch_map = {"small": [64, 64], "medium": [256, 256], "large": [400, 300]}
        arch_name = best_hp.pop("net_arch", "medium")
        policy_kwargs = {"net_arch": arch_map.get(arch_name, [256, 256])}
        sac_kwargs.update(best_hp)
        logger.info("Загружены Optuna-гиперпараметры из %s", hp_path)

    try:
        env = Monitor(TradingEnv(train_df))  # type: ignore[var-annotated]
        model = SAC(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs or None,
            **sac_kwargs,
        )
        model.learn(
            total_timesteps=total_timesteps,
            callback=_ProgressCallback(total_timesteps),
        )
    except Exception as e:
        logger.error(f"Ошибка обучения SAC: {e}", exc_info=True)
        return None

    # Оценка на тестовой выборке (out-of-sample)
    eval_result = _evaluate_model(model, test_df)

    # Дообучение на реальных сделках, накопленных ботом
    norm_stats = _compute_norm_stats(train_df)
    _finetune_on_experiences(model, norm_stats)

    # Резервная копия перед перезаписью
    backup_path = ""
    if save_backup:
        backup_path = _backup_existing_model(model_path)

    # Сохранение модели (norm_stats уже вычислен выше на train-данных)
    save_path = model_path.replace(".zip", "")
    model.save(save_path)
    logger.info(f"Модель SAC сохранена → {save_path}.zip")

    _save_norm_stats(model_path, norm_stats)

    result = {**eval_result, "backup": backup_path, "model": model_path}
    print(f"TRAIN_RESULT:{json.dumps(result)}", file=sys.stdout, flush=True)

    return model_path


def train_walk_forward(
    df: pd.DataFrame,
    model_path: str = MODEL_PATH,
    train_months: int = _WF_TRAIN_MONTHS,
    step_months: int = _WF_STEP_MONTHS,
    total_timesteps: int = 100_000,
) -> str | None:
    """
    Walk-forward retraining: скользящее окно train→validate.

    Обучает SAC на окне train_months, оценивает на следующем step_months,
    сдвигает окно, повторяет. Последняя итерация сохраняет финальную модель.

    :param df: Полный датафрейм OHLCV.
    :param model_path: Куда сохранить итоговую модель.
    :param train_months: Размер тренировочного окна (в месяцах).
    :param step_months: Шаг сдвига окна (в месяцах).
    :param total_timesteps: Шагов обучения за итерацию.
    :return: Путь к финальной модели или None при ошибке.
    """
    _min_candles = int(os.getenv("TRAIN_MIN_CANDLES", "2880"))
    if len(df) < _min_candles:
        raise ValueError(
            f"DataFrame too short: {len(df)} candles < {_min_candles} minimum"
        )

    train_size = train_months * _CANDLES_PER_MONTH
    step_size = step_months * _CANDLES_PER_MONTH
    n = len(df)

    if n < train_size + step_size:
        logger.error(
            "Недостаточно данных для walk-forward: %d свечей, нужно %d",
            n,
            train_size + step_size,
        )
        return None

    starts = list(range(0, n - train_size - step_size + 1, step_size))
    logger.info(
        "Walk-forward: %d итераций, окно=%d, шаг=%d свечей",
        len(starts),
        train_size,
        step_size,
    )

    last_saved = None
    for i, start in enumerate(starts, 1):
        train_df = df.iloc[start : start + train_size].reset_index(drop=True)
        val_df = df.iloc[
            start + train_size : start + train_size + step_size
        ].reset_index(drop=True)

        logger.info(
            "WF [%d/%d] train=[%d:%d] val=[%d:%d]",
            i,
            len(starts),
            start,
            start + train_size,
            start + train_size,
            start + train_size + step_size,
        )

        norm_stats = _compute_norm_stats(train_df)
        try:
            from stable_baselines3 import SAC  # noqa: I001
            from stable_baselines3.common.monitor import Monitor

            from reinforcement_learning.rl_env import TradingEnv

            env = Monitor(TradingEnv(train_df))  # type: ignore[var-annotated]
            model = SAC("MlpPolicy", env, verbose=0)
            model.learn(total_timesteps=total_timesteps)
        except Exception as exc:
            logger.error("WF [%d/%d] ошибка обучения: %s", i, len(starts), exc)
            continue

        wf_eval = _evaluate_model(model, val_df)

        # Финальная итерация — сохраняем как основную модель
        if i == len(starts):
            save_path = model_path.replace(".zip", "")
            model.save(save_path)
            _save_norm_stats(model_path, norm_stats)
            last_saved = model_path
            logger.info("Walk-forward завершён → %s.zip", save_path)
            backup_wf = _backup_existing_model(model_path)
            result_wf = {**wf_eval, "backup": backup_wf, "model": model_path}
            print(f"TRAIN_RESULT:{json.dumps(result_wf)}", file=sys.stdout, flush=True)

    return last_saved


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    from src.data_loader import DataLoader

    async def _run() -> None:
        from src.bybit_api import BybitAPI
        from src.market_scanner import MarketScanner

        months = int(os.getenv("BT_MONTHS", "6"))
        timeframe = os.getenv("BT_TIMEFRAME", "15m")
        top_n = int(os.getenv("TRAIN_TOP_N", "20"))

        api = BybitAPI()
        await api.initialize(
            api_key=os.getenv("BYBIT_API_KEY", ""),
            api_secret=os.getenv("BYBIT_API_SECRET", ""),
        )
        loader = DataLoader()
        await loader.initialize(
            api_key=os.getenv("BYBIT_API_KEY", ""),
            api_secret=os.getenv("BYBIT_API_SECRET", ""),
        )
        scanner = MarketScanner(api, loader)
        symbols = await scanner.get_top_symbols(top_n)
        await api.close()

        frames = []
        for sym in symbols:
            try:
                df_sym = await loader.get_paginated_history(
                    sym, timeframe, months=months
                )
                if df_sym is None or df_sym.empty:
                    continue
                # Нормализуем ценовые колонки к % доходности от первой свечи,
                # чтобы BTC ($64k) и малые монеты ($0.06) были в одном масштабе
                price_cols = [
                    c for c in ("open", "high", "low", "close") if c in df_sym.columns
                ]
                base = df_sym[price_cols[0]].iloc[0]
                if base > 0:
                    for col in price_cols:
                        df_sym[col] = df_sym[col] / base
                frames.append(df_sym)
                logger.info("Загружено %s: %d свечей", sym, len(df_sym))
            except Exception as exc:
                logger.warning("Пропускаем %s: %s", sym, exc)

        await loader.close()

        if not frames:
            logger.error("Данные не загружены — прерываем")
            return

        combined = pd.concat(frames, ignore_index=True)
        logger.info(
            "Объединённый датасет: %d свечей из %d символов",
            len(combined),
            len(frames),
        )
        train(combined)

    import asyncio

    asyncio.run(_run())
