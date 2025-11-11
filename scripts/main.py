import asyncio
import logging
import time
import aiohttp
import json
import os
from src.news_analyzer import analyze_news_sentiment
from src.rl_env import TradingEnv
from stable_baselines3 import PPO
import ccxt.async_support as ccxt
import redis.asyncio as redis
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные переменные
model = PPO.load("models/ppo_trading_model.zip")
env = TradingEnv()
r = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=int(os.getenv('REDIS_PORT', 6379)), decode_responses=True)
NEWS_API_KEY = os.getenv('NEWS_API_KEY', 'YOUR_NEWSAPI_KEY')  # Из .env
NEWS_URL = 'https://newsapi.org/v2/everything?q=bitcoin+trading&apiKey={}&pageSize=5'  # Пример запроса

async def fetch_news_async():
    """Асинхронно получает свежие новости из API."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(NEWS_URL.format(NEWS_API_KEY)) as response:
                data = await response.json()
                articles = data.get('articles', [])
                return [{'title': art['title'], 'description': art['description'], 'id': art['url']} for art in articles]
        except Exception as e:
            logger.error(f"News fetch error: {e}")
            return []

async def predict_price_async(obs):
    """Асинхронное предсказание RL."""
    start = time.perf_counter()
    action, _ = model.predict(obs)
    logger.info(f"Prediction time: {time.perf_counter() - start:.4f}s")
    return int(action)

async def execute_trade_async(action, symbol='BTC/USDT'):
    """Асинхронное выполнение трейда."""
    bybit = ccxt.bybit({
        'apiKey': os.getenv('BYBIT_API_KEY', 'YOUR_API_KEY'),
        'secret': os.getenv('BYBIT_API_SECRET', 'YOUR_SECRET'),
        'test': True,  # Тестовый режим
        'enableRateLimit': True,
    })
    
    try:
        balance = await bybit.fetch_balance()
        free_usdt = balance['USDT']['free']
        max_loss = free_usdt * 0.1
        
        if action == 1 and free_usdt > 100:
            volume = min(free_usdt * 0.05, 0.01)
            ticker = await bybit.fetch_ticker(symbol)
            price = ticker['last']
            order = await bybit.create_limit_buy_order(symbol, volume, price)
            logger.info(f"Buy order placed: {order}")
            return order
        elif action == 2:
            balance_btc = balance['BTC']['free']
            if balance_btc > 0.0001:
                order = await bybit.create_market_sell_order(symbol, balance_btc)
                logger.info(f"Sell order placed: {order}")
                return order
    except Exception as e:
        logger.error(f"Trade error: {e}")
    finally:
        await bybit.close()
    return None

async def main_loop():
    """Основной асинхронный цикл: получает новости, анализирует, предсказывает, трейдит."""
    obs = env.reset()
    last_sentiment = 0.0
    step_count = 0
    
    while True:
        try:
            start_cycle = time.perf_counter()
            
            # Получаем новости асинхронно
            news_list = await fetch_news_async()
            if not news_list:
                await asyncio.sleep(10)  # Ждём 10 сек перед следующим запросом
                continue
            
            # Анализируем каждую новость и усредняем настроение
            sentiments = []
            for news in news_list:
                text = f"{news['title']} {news['description']}"
                sentiment = analyze_news_sentiment(text)
                sentiments.append(sentiment)
                # Кэшируем в Redis
                await r.setex(f"sentiment:{news['id']}", 3600, sentiment)
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0.0
            
            # Триггер на значимое изменение
            if abs(avg_sentiment - last_sentiment) > 0.2:
                # Обновляем obs с учётом настроения
                obs = await env.update_obs_live_async(avg_sentiment)
                
                # Параллельные задачи
                predict_task = asyncio.create_task(predict_price_async(obs))
                action = await predict_task
                trade_task = asyncio.create_task(execute_trade_async(action))
                await trade_task
                
                # Онлайн-обучение (опционально, для тестов; комментируй в проде)
                # model.learn(total_timesteps=1)  # Обновляем модель на 1 шаг (рискованно!)
                step_count += 1
                if step_count % 1000 == 0:
                    model.save("models/ppo_trading_model_updated.zip")  # Сохраняем обновления
            
            last_sentiment = avg_sentiment
            logger.info(f"Cycle time: {time.perf_counter() - start_cycle:.4f}s, Sentiment: {avg_sentiment:.2f}")
            
            await asyncio.sleep(10)  # Пауза между циклами
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            await asyncio.sleep(10)

async def main():
    """Запуск основного цикла."""
    logger.info("Trading bot with API integration running...")
    await main_loop()

if __name__ == "__main__":
    asyncio.run(main())
