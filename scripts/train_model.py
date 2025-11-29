import os  # Убедитесь, что это есть
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import DataLoader
from src.rl_env import TradingEnv
from src.strategies import strategies
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd

def main(strategy="ppo"):
    try:
        # Загрузка данных
        data_loader = DataLoader()
        data = data_loader.load_data("BTC/USDT", "1h", limit=10000)  # Увеличьте limit для обучения

        # Получение параметров стратегии
        params = strategies.get(strategy, strategies["ppo"])

        # Создание среды (уберите **params из TradingEnv)
        env = DummyVecEnv([lambda: TradingEnv(data, strategy=strategy)])

        # Создание и обучение модели (передайте params в PPO)
        model = PPO("MlpPolicy", env, **params)
        model.learn(total_timesteps=50000)  # Увеличьте для лучшего обучения

        # Сохранение модели
        os.makedirs("models", exist_ok=True)  # exist_ok=True для безопасности
        model.save(f"models/{strategy}_model.zip")
        print(f"Model saved to models/{strategy}_model.zip")
    except Exception as e:
        print(f"Error in training: {e}")

if __name__ == "__main__":
    main()
