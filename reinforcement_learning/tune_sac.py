"""
Поиск гиперпараметров Optuna для SAC-агента.

Подбирает: learning_rate, batch_size, tau, gamma, net_arch.
Сохраняет лучшие параметры в models/best_hyperparams.json.
train_sac.py загружает этот файл автоматически, если он существует.

Профили сезонов (SAC_PROFILE):
  bluechip — BTC, ETH, SOL, BNB, XRP, ADA, AVAX, DOGE (крупные монеты)
  altcoin  — топ по объёму исключая bluechip (средние монеты)
  <пусто>  — топ по объёму без фильтра

Использование:
    PYTHONPATH=. python3 reinforcement_learning/tune_sac.py
    SAC_PROFILE=bluechip make tune
    SAC_PROFILE=altcoin  make tune
"""

from __future__ import annotations

import json
import logging
import os
from typing import List

logger = logging.getLogger(__name__)

_SAC_PROFILE = os.getenv("SAC_PROFILE", "")

from src.runtime_config import BLUECHIP_BASES  # noqa: E402

_ALTCOIN_EXCLUDE: set = BLUECHIP_BASES | {"USDC", "USDT", "BUSD", "TUSD", "DAI"}
_TUNE_TOP_N = int(os.getenv("TUNE_TOP_N", "5"))  # символов для тюнинга
HYPERPARAMS_PATH = (
    f"models/best_hyperparams_{_SAC_PROFILE}.json"
    if _SAC_PROFILE
    else "models/best_hyperparams.json"
)
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
        env = Monitor(TradingEnv(train_df))  # type: ignore[var-annotated]
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


async def _get_tune_symbols(api: "Any", loader: "Any") -> List[str]:
    """
    Возвращает список символов для тюнинга согласно SAC_PROFILE.

    bluechip → фиксированный список крупных монет
    altcoin  → топ TUNE_TOP_N по объёму, исключая bluechip-монеты
    <пусто>  → топ TUNE_TOP_N по объёму без фильтра
    Переопределяется через TUNE_SYMBOLS="BTC/USDT,ETH/USDT" в .env.
    """
    override = os.getenv("TUNE_SYMBOLS", "").strip()
    if override:
        symbols = [s.strip() for s in override.split(",") if s.strip()]
        logger.info("TUNE_SYMBOLS override: %s", symbols)
        return symbols

    from src.market_scanner import MarketScanner

    scanner = MarketScanner(api, loader)

    if _SAC_PROFILE == "bluechip":
        # Из топ-50 оставляем только те что классифицируются как bluechip
        all_symbols = await scanner.get_top_symbols(50)
        symbols = [s for s in all_symbols if s.split("/")[0] in BLUECHIP_BASES][:_TUNE_TOP_N]
        logger.info("Season=bluechip: %d symbols from top-50 %s", len(symbols), symbols)
        return symbols

    all_symbols = await scanner.get_top_symbols(50)

    if _SAC_PROFILE == "altcoin":
        # Из топа исключаем все bluechip — только настоящие альты
        symbols = [
            s for s in all_symbols
            if s.split("/")[0] not in _ALTCOIN_EXCLUDE
        ][:_TUNE_TOP_N]
        logger.info("Season=altcoin: top %d alts %s", len(symbols), symbols)
    else:
        symbols = all_symbols[:_TUNE_TOP_N]
        logger.info("Season=default: top %d by volume %s", len(symbols), symbols)

    return symbols


if __name__ == "__main__":
    import asyncio
    import logging as _logging
    from typing import Any

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    from src.bybit_api import BybitAPI
    from src.data_loader import DataLoader

    async def _run() -> None:
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

        symbols = await _get_tune_symbols(api, loader)
        await api.close()

        if not symbols:
            logger.error("No symbols found for tuning — aborting")
            return

        frames = []
        import pandas as pd
        for sym in symbols:
            try:
                df_sym = await loader.get_paginated_history(sym, "15m", months=3)
                if df_sym is None or df_sym.empty:
                    logger.warning("No data for %s — skipping", sym)
                    continue
                # Нормализуем к % доходности как в train_sac.py
                price_cols = [c for c in ("open", "high", "low", "close") if c in df_sym.columns]
                base = df_sym[price_cols[0]].iloc[0]
                if base > 0:
                    for col in price_cols:
                        df_sym[col] = df_sym[col] / base
                frames.append(df_sym)
                logger.info("Loaded %s: %d candles", sym, len(df_sym))
            except Exception as exc:
                logger.warning("Skipping %s: %s", sym, exc)

        await loader.close()

        if not frames:
            logger.error("No data loaded — aborting")
            return

        combined = pd.concat(frames, ignore_index=True)
        logger.info(
            "Tuning on %d symbols, %d total candles (profile=%r)",
            len(frames),
            len(combined),
            _SAC_PROFILE or "default",
        )
        tune(combined, n_trials=_N_TRIALS)

    asyncio.run(_run())
