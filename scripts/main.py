import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.bybit_api import BybitAPI
from src.news_analyzer import NewsAnalyzer
from src.rl_env import TradingEnv
from src.strategies import strategies
from stable_baselines3 import PPO
import logging

logging.basicConfig(level=logging.INFO)

async def main(strategy="ppo"):
    try:
        # Инициализация
        api = BybitAPI()
        news_analyzer = NewsAnalyzer()
        await api.initialize_async()  # Инициализация CCXT

        # Загрузка модели
        model_path = f"models/{strategy}_model.zip"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_path} not found. Train first.")
        model = PPO.load(model_path)

        # Загрузка данных для среды (больше данных для live)
        data = await api.fetch_historical_data_async("BTC/USDT", "1h", limit=1000)

        # Создание среды (уберите **params)
        env = TradingEnv(data, strategy=strategy)

        position = 0  # 0: нет позиции, 1: long, -1: short

        while True:
            # Анализ новостей
            sentiment = await news_analyzer.analyze_news_async()

            # Обновление наблюдений (пока placeholder, реализуйте как предлагал)
            obs = await env.update_obs_live_async(sentiment, strategy)

            # Предсказание действия
            action, _ = model.predict(obs)

            # Симуляция/логика торговли (замените на реальную, как в предыдущем ответе)
            if action == 1 and position == 0:
                logging.info("Simulated: Open long position")
                position = 1
            elif action == 2 and position == 0:
                logging.info("Simulated: Open short position")
                position = -1
            elif action == 0 and position != 0:
                logging.info("Simulated: Close position")
                position = 0

            await asyncio.sleep(3600)  # Каждые 1h

    except Exception as e:
        logging.error(f"Error in live trading: {e}")
    finally:
        await api.close_async()  # Закройте CCXT сессию

if __name__ == "__main__":
    asyncio.run(main())
