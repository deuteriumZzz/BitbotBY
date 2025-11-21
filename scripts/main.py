import asyncio
import logging
import os

import redis
from dotenv import load_dotenv
from stable_baselines3 import PPO

from src.bybit_api import BybitAPI
from src.news_analyzer import NewsAnalyzer
from src.rl_env import TradingEnv
from src.strategies import strategies

load_dotenv()
logging.basicConfig(level=logging.INFO)


async def main(strategy="ppo"):
    try:
        # Подключение к Redis
        r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))

        # Загрузка данных
        api = BybitAPI()
        data = await api.fetch_historical_data_async("BTC/USDT", "1h", limit=100)

        # Параметры стратегии
        params = strategies.get(strategy, strategies["ppo"])

        # Создание среды
        env = TradingEnv(data, strategy=strategy, **params)

        # Анализатор новостей
        news_analyzer = NewsAnalyzer(r)  # Передаём Redis

        # Загрузка модели
        model_path = f"models/ppo_{strategy}.zip"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = PPO.load(model_path)

        # Live-торговля
        obs, _ = env.reset()
        while True:
            # Обновление данных и obs с кэшированием
            sentiment = await news_analyzer.analyze_news_async()
            obs = await env.update_obs_live_async(sentiment, strategy)

            # Сохраняем последний сентимент в Redis для отладки
            r.setex("last_sentiment", 300, str(sentiment))  # TTL 5 мин

            # Предсказание и шаг
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)

            logging.info(f"Action: {action}, Reward: {reward}, Sentiment: {sentiment}")
            await asyncio.sleep(60)
    except Exception as e:
        logging.error(f"Error in live trading: {e}")


if __name__ == "__main__":
    asyncio.run(main())
