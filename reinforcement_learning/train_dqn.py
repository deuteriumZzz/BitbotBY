"""
Обучение DQN-модели на исторических данных Bybit.

Запуск:
    python reinforcement_learning/train_dqn.py

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

EPISODES = int(os.getenv("DQN_EPISODES", "200"))
SYMBOL = Config.SYMBOL
TIMEFRAME = Config.TIMEFRAME
LIMIT = 1000


async def fetch_data() -> pd.DataFrame:
    """Загружает OHLCV + индикаторы через DataLoader."""
    api = BybitAPI()
    loader = DataLoader()
    await api.initialize(
        Config.BYBIT_API_KEY, Config.BYBIT_API_SECRET
    )
    await loader.initialize(
        Config.BYBIT_API_KEY, Config.BYBIT_API_SECRET
    )
    try:
        df = await loader.get_market_data(
            SYMBOL, TIMEFRAME, limit=LIMIT
        )
        df = loader.calculate_technical_indicators(df)
        logger.info(
            f"Loaded {len(df)} candles "
            f"for {SYMBOL} ({TIMEFRAME})"
        )
        return df
    finally:
        await api.close()
        await loader.close()


def train(df: pd.DataFrame) -> RLAgent:
    """
    Обучает DQN на TradingEnv с историческими данными.

    Каждый эпизод — один прогон по всему DataFrame.
    Агент учится максимизировать portfolio value.

    :param df: DataFrame с OHLCV + индикаторами.
    :return: Обученный RLAgent.
    """
    env = TradingEnv(df, Config.INITIAL_BALANCE)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = RLAgent(state_size, action_size)
    logger.info(
        f"Training DQN: state_size={state_size}, "
        f"actions={action_size}, episodes={EPISODES}"
    )

    best_value = 0.0

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

        final_value = info.get("value", 0)
        if final_value > best_value:
            best_value = final_value

        if ep % 20 == 0 or ep == EPISODES:
            logger.info(
                f"Ep {ep:>4}/{EPISODES} | "
                f"reward={total_reward:+.1f} | "
                f"value=${final_value:.2f} | "
                f"best=${best_value:.2f} | "
                f"eps={agent.epsilon:.3f}"
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
        f"DQN Training | {SYMBOL} | "
        f"{TIMEFRAME} | {EPISODES} episodes"
    )
    df = await fetch_data()
    if df.empty or len(df) < 50:
        logger.error(
            "Not enough data. "
            "Check BYBIT_API_KEY and TRADING_SYMBOL."
        )
        return
    agent = train(df)
    save_model(agent)


if __name__ == "__main__":
    asyncio.run(main())
