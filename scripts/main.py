import asyncio
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from src.rl_env import TradingEnv
from src.news_analyzer import NewsAnalyzer
from src.bybit_api import BybitAPI
from src.strategies import strategies
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

async def main(strategy='ppo'):
    try:
        # Загрузка данных (инициализация)
        api = BybitAPI()
        data = await api.fetch_historical_data_async('BTC/USDT', '1h', limit=100)

        # Параметры стратегии
        params = strategies.get(strategy, strategies['ppo'])

        # Создание среды
        env = TradingEnv(data, strategy=strategy, **params)

        # Анализатор новостей
        news_analyzer = NewsAnalyzer()

        # Загрузка модели
        model_path = f"models/ppo_{strategy}.zip"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = PPO.load(model_path)

        # Live-торговля
        obs, _ = env.reset()
        while True:
            # Обновление данных и obs
            sentiment = await news_analyzer.analyze_news_async(['crypto news placeholder'])  # Замените на реальный источник
            obs = await env.update_obs_live_async(sentiment, strategy)

            # Предсказание и шаг
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)

            logging.info(f"Action: {action}, Reward: {reward}")
            await asyncio.sleep(60)  # Каждую минуту
    except Exception as e:
        logging.error(f"Error in live trading: {e}")

if __name__ == "__main__":
    asyncio.run(main())
