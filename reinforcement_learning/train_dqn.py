"""
Обучение DQN-модели на исторических данных Bybit.

Запуск:
    python reinforcement_learning/train_dqn.py

Данные кешируются в data/cache/{symbol}_{timeframe}.csv (TTL 24ч).
При повторном запуске догружаются только новые свечи.
Результат → DQN_MODEL_PATH (по умолч. models/dqn_model.pth).
После обучения установите MODE=dqn или MODE=hybrid в .env.
"""

import asyncio
import logging
import os
import sys

import pandas as pd

sys.path.insert(
    0, os.path.dirname(os.path.dirname(__file__))
)

from config import Config
from reinforcement_learning.rl_agent import RLAgent
from reinforcement_learning.rl_env import TradingEnv
from src.bybit_api import BybitAPI
from src.data_loader import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

EPISODES = int(os.getenv("DQN_EPISODES", "300"))
DQN_MONTHS = int(os.getenv("DQN_MONTHS", "6"))
SYMBOL = Config.SYMBOL
TIMEFRAME = Config.TIMEFRAME
# Доля данных для финальной валидации (не участвует в обучении)
TRAIN_RATIO = 0.80


async def fetch_data() -> pd.DataFrame:
    """
    Загружает исторические OHLCV через пагинацию.

    Первый запуск: скачивает DQN_MONTHS месяцев батчами по 200.
    Повторный: загружает из CSV-кэша, догружает только новые свечи.
    """
    api = BybitAPI()
    loader = DataLoader()
    await api.initialize(
        Config.BYBIT_API_KEY, Config.BYBIT_API_SECRET
    )
    await loader.initialize(
        Config.BYBIT_API_KEY, Config.BYBIT_API_SECRET
    )
    try:
        df = await loader.get_paginated_history(
            SYMBOL, TIMEFRAME, months=DQN_MONTHS
        )
        return df
    finally:
        await api.close()
        await loader.close()


def split_train_val(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Делит DataFrame на train (80%) и validation (20%).

    Модель обучается только на train; val используется
    для финальной оценки (не участвует в обучении).
    """
    split = int(len(df) * TRAIN_RATIO)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def evaluate(agent: RLAgent, df: pd.DataFrame) -> float:
    """
    Запускает один эпизод без обучения (epsilon=0).

    :return: Финальная стоимость портфеля.
    """
    env = TradingEnv(df, Config.INITIAL_BALANCE)
    obs, _ = env.reset()
    done = False
    info = {}
    saved_eps = agent.epsilon
    agent.epsilon = 0.0  # greedy

    while not done:
        action = agent.choose_action(obs)
        obs, _, done, _, info = env.step(action)

    agent.epsilon = saved_eps
    return info.get("value", 0.0)


def train(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> RLAgent:
    """
    Обучает DQN на df_train, валидирует на df_val.

    Каждые 50 эпизодов выводит метрики и сохраняет
    лучшую по val-стоимости модель.

    :param df_train: Обучающая выборка (80%).
    :param df_val: Валидационная выборка (20%).
    :return: Лучший RLAgent по val-метрике.
    """
    env = TradingEnv(df_train, Config.INITIAL_BALANCE)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = RLAgent(state_size, action_size)
    logger.info(
        f"Training DQN: "
        f"state={state_size}, actions={action_size}, "
        f"episodes={EPISODES}, "
        f"train={len(df_train)} / val={len(df_val)} candles"
    )

    best_val_value = 0.0
    best_state = None

    for ep in range(1, EPISODES + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done, _, info = (
                env.step(action)
            )
            agent.remember(
                obs, action, reward, next_obs, done
            )
            agent.replay()
            obs = next_obs
            total_reward += reward

        train_value = info.get("value", 0.0)

        if ep % 50 == 0 or ep == EPISODES:
            val_value = evaluate(agent, df_val)
            logger.info(
                f"Ep {ep:>4}/{EPISODES} | "
                f"train=${train_value:.2f} | "
                f"val=${val_value:.2f} | "
                f"reward={total_reward:+.1f} | "
                f"eps={agent.epsilon:.3f}"
            )
            if val_value > best_val_value:
                best_val_value = val_value
                # Сохраняем state_dict лучшей модели
                import copy
                best_state = copy.deepcopy(
                    agent.policy_net.state_dict()
                )
                logger.info(
                    f"  ✓ New best val=${val_value:.2f}"
                )

    # Восстанавливаем лучшие веса
    if best_state is not None:
        agent.policy_net.load_state_dict(best_state)
        agent.target_net.load_state_dict(best_state)
        logger.info(
            f"Restored best model (val=${best_val_value:.2f})"
        )

    return agent


def save_model(agent: RLAgent) -> None:
    path = Config.DQN_MODEL_PATH
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    agent.save_model(path)
    logger.info(f"Model saved → {path}")
    logger.info(
        "Set MODE=dqn or MODE=hybrid in .env to use it."
    )


async def main() -> None:
    logger.info(
        f"DQN Training | {SYMBOL} {TIMEFRAME} | "
        f"{DQN_MONTHS} months | {EPISODES} episodes"
    )
    df = await fetch_data()

    if df.empty or len(df) < 100:
        logger.error(
            f"Not enough data ({len(df)} candles). "
            "Check BYBIT_API_KEY and TRADING_SYMBOL."
        )
        return

    logger.info(
        f"Total candles: {len(df)} "
        f"({len(df) * 15 // 60 // 24} days for 15m)"
    )

    df_train, df_val = split_train_val(df)
    agent = train(df_train, df_val)

    # Финальная оценка
    val_final = evaluate(agent, df_val)
    roi = (val_final - Config.INITIAL_BALANCE) / Config.INITIAL_BALANCE
    logger.info(
        f"Final validation: "
        f"${val_final:.2f} "
        f"(ROI {roi:+.1%} vs buy-and-hold)"
    )

    save_model(agent)


if __name__ == "__main__":
    asyncio.run(main())
