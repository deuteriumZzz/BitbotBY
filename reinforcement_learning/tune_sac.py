"""
Поиск гиперпараметров Optuna для SAC-агента.

Подбирает: learning_rate, batch_size, tau, gamma, net_arch.
Сохраняет лучшие параметры в models/best_hyperparams.json.
train_sac.py загружает этот файл автоматически, если он существует.

Использование:
    PYTHONPATH=. python3 reinforcement_learning/tune_sac.py
    make tune
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

HYPERPARAMS_PATH = "models/best_hyperparams.json"
_TUNE_TIMESTEPS = 50_000
_N_TRIALS = int(os.getenv("OPTUNA_TRIALS", "30"))
_TRAIN_SPLIT = 0.7


def _net_arch(trial: object) -> list[int]:
    """Преобразует категориальный параметр trial → размеры скрытых слоёв."""
    import optuna  # noqa: I001

    _trial: "optuna.Trial" = trial  # type: ignore[assignment]
    arch = _trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    return {
        "small": [64, 64],
        "medium": [256, 256],
        "large": [400, 300],
    }[arch]


def objective(trial: object, df: object) -> float:
    """
    Целевая функция Optuna: обучает SAC на _TUNE_TIMESTEPS шагах,
    возвращает отрицательный итоговый PnL.

    :param trial: объект optuna.Trial.
    :param df: OHLCV DataFrame.
    :return: Отрицательная итоговая стоимость портфеля
             (меньше = лучше для минимизации).
    """
    import optuna  # noqa: I001
    import pandas as pd
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor

    from reinforcement_learning.rl_env import TradingEnv

    _trial: "optuna.Trial" = trial  # type: ignore[assignment]
    _df: pd.DataFrame = df  # type: ignore[assignment]

    lr = _trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = _trial.suggest_categorical("batch_size", [64, 128, 256])
    tau = _trial.suggest_float("tau", 0.001, 0.05, log=True)
    gamma = _trial.suggest_float("gamma", 0.95, 0.999)
    net_arch = _net_arch(_trial)

    split_idx = int(len(_df) * _TRAIN_SPLIT)
    train_df = _df.iloc[:split_idx].reset_index(drop=True)
    val_df = _df.iloc[split_idx:].reset_index(drop=True)

    try:
        env = Monitor(TradingEnv(train_df))
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=lr,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            policy_kwargs={"net_arch": net_arch},
            verbose=0,
        )
        model.learn(total_timesteps=_TUNE_TIMESTEPS)
    except Exception as exc:
        logger.warning("Попытка %d завершилась ошибкой: %s", _trial.number, exc)
        return 0.0

    env_val = TradingEnv(val_df, initial_balance=10_000.0)
    obs, _ = env_val.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env_val.step(action)

    final_value = env_val.current_value
    logger.info(
        "Trial %d: lr=%.2e bs=%d tau=%.4f gamma=%.4f arch=%s → $%.0f",
        _trial.number,
        lr,
        batch_size,
        tau,
        gamma,
        net_arch,
        final_value,
    )
    return -float(final_value)


def tune(df: object, n_trials: int = _N_TRIALS) -> dict:
    """
    Запускает поиск гиперпараметров Optuna и сохраняет лучшие в JSON.

    :param df: Полный OHLCV DataFrame.
    :param n_trials: Количество попыток.
    :return: Словарь лучших гиперпараметров.
    """
    try:
        import optuna
    except ImportError:
        logger.error("optuna не установлен: pip install optuna>=3.6.0")
        raise

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: objective(trial, df),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    best_value = -study.best_value
    logger.info("Лучшая попытка: $%.0f | параметры: %s", best_value, json.dumps(best))

    os.makedirs("models", exist_ok=True)
    with open(HYPERPARAMS_PATH, "w") as f:
        json.dump(best, f, indent=2)
    logger.info("Лучшие гиперпараметры сохранены → %s", HYPERPARAMS_PATH)

    return best


if __name__ == "__main__":
    import asyncio
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    from src.data_loader import DataLoader

    async def _run() -> None:
        loader = DataLoader()
        await loader.initialize(
            api_key=os.getenv("BYBIT_API_KEY", ""),
            api_secret=os.getenv("BYBIT_API_SECRET", ""),
        )
        symbol = os.getenv("TRADING_SYMBOL", "BTC/USDT")
        df = await loader.load_ohlcv(symbol, "15m", limit=5760)
        if df is None or df.empty:
            logger.error("Нет данных для тюнинга")
            return
        tune(df, n_trials=_N_TRIALS)

    asyncio.run(_run())
