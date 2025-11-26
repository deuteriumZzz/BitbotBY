import logging
import os

from dotenv import load_dotenv
from stable_baselines3 import PPO

from src.data_loader import DataLoader
from src.rl_env import TradingEnv
from src.strategies import strategies

load_dotenv()
logging.basicConfig(level=logging.INFO)


def main(strategy="ppo"):
    try:
        # Загрузка данных
        loader = DataLoader()
        data = loader.load_historical_data("BTC/USDT", "1h", limit=1000)
        if data.empty:
            raise ValueError("No data loaded")

        # Параметры стратегии
        params = strategies.get(strategy, strategies["ppo"])

        # Создание среды
        env = TradingEnv(data, strategy=strategy, **params)

        # Загрузка модели
        model_path = f"models/ppo_{strategy}.zip"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = PPO.load(model_path)

        # Бэктестинг
        obs, _ = env.reset()
        total_reward = 0
        for _ in range(len(data) - 1):
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done:
                break

        logging.info(f"Backtest total reward: {total_reward}")
        env.render()  # Показать итоги
    except Exception as e:
        logging.error(f"Error in backtest: {e}")


if __name__ == "__main__":
    main()
