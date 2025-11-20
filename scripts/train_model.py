import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.rl_env import TradingEnv
from src.data_loader import DataLoader
from src.strategies import strategies
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

def main(strategy='ppo'):
    try:
        # Загрузка данных
        loader = DataLoader()
        data = loader.load_historical_data('BTC/USDT', '1h', limit=5000)
        if data.empty:
            raise ValueError("No data loaded")

        # Параметры стратегии
        params = strategies.get(strategy, strategies['ppo'])

        # Создание среды
        env = DummyVecEnv([lambda: TradingEnv(data, strategy=strategy, **params)])

        # Модель PPO
        model = PPO("MlpPolicy", env, verbose=1)

        # Тренировка
        model.learn(total_timesteps=10000)

        # Сохранение
        os.makedirs("models", exist_ok=True)
        model.save(f"models/ppo_{strategy}")
        logging.info(f"Model saved: models/ppo_{strategy}.zip")
    except Exception as e:
        logging.error(f"Error in training: {e}")

if __name__ == "__main__":
    main()
