import os  # Добавлено для устранения 'os' is not defined
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import DataLoader
from src.rl_env import TradingEnv
from src.strategies import strategies
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import logging

logging.basicConfig(level=logging.INFO)

def main(strategy="ppo"):
    try:
        logging.info("Starting training...")
        # Загрузка данных
        data_loader = DataLoader()
        data = data_loader.load_data("BTC/USDT", "1h", limit=10000)  # Увеличено для лучшего обучения

        # Получение параметров стратегии
        params = strategies.get(strategy, strategies["ppo"])

        # Создание среды (убраны **params из TradingEnv)
        env = DummyVecEnv([lambda: TradingEnv(data, strategy=strategy)])

        # Создание и обучение модели (params переданы в PPO)
        model = PPO("MlpPolicy", env, **params)
        model.learn(total_timesteps=50000)  # Увеличено для лучшего обучения

        # Сохранение модели
        os.makedirs("models", exist_ok=True)
        model.save(f"models/{strategy}_model.zip")
        logging.info(f"Model saved to models/{strategy}_model.zip")
    except Exception as e:
        logging.error(f"Error in training: {e}")

if __name__ == "__main__":
    main()
