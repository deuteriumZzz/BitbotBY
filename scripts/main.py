import asyncio
import logging
import os
import sys
import time
from argparse import ArgumentParser

import aiohttp
import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.news_analyzer import analyze_news_sentiment
from src.rl_env import TradingEnv
from src.strategies import get_strategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные переменные
model = None
env = None
r = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True,
)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_NEWSAPI_KEY")
NEWS_URL = "https://newsapi.org/v2/everything?q=bitcoin+trading&apiKey={}&pageSize=5"
CSV_PATH = "data/historical_btc.csv"

# Функция для создания/обновления CSV
def update_historical_csv(order, symbol="BTC/USDT"):
    if order is None:
        return
    try:
        # Получаем последние OHLCV-данные (последние 1 бар для простоты)
        bybit = ccxt.bybit({"enableRateLimit": True})
        ohlcv = bybit.fetch_ohlcv(symbol, "1m", limit=1)  # Последний 1-минутный бар
        bybit.close()
        if not ohlcv:
            logger.warning("No OHLCV data to add to CSV")
            return

        # Преобразуем в DataFrame
        df_new = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit="ms")

        # Проверяем, существует ли файл
        if not os.path.exists(CSV_PATH):
            df_new.to_csv(CSV_PATH, index=False)
            logger.info(f"Created {CSV_PATH} with initial data")
        else:
            # Аппендим новые данные (без дубликатов по timestamp)
            df_existing = pd.read_csv(CSV_PATH)
            df_existing["timestamp"] = pd.to_datetime(df_existing["timestamp"])
            df_combined = pd.concat([df_existing, df_new]).drop_duplicates(subset="timestamp").sort_values("timestamp")
            df_combined.to_csv(CSV_PATH, index=False)
            logger.info(f"Updated {CSV_PATH} with new data after trade")
    except Exception as e:
        logger.error(f"Error updating CSV: {e}")

async def fetch_news_async():
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(NEWS_URL.format(NEWS_API_KEY)) as response:
                data = await response.json()
                articles = data.get("articles", [])
                return [
                    {
                        "title": art["title"],
                        "description": art["description"],
                        "id": art["url"],
                    }
                    for art in articles
                ]
        except Exception as e:
            logger.error(f"News fetch error: {e}")
            return []

async def predict_price_async(obs, strategy):
    start = time.perf_counter()
    action, _ = model.predict(obs)
    logger.info(f"Prediction time: {time.perf_counter() - start:.4f}s")
    return int(action)

async def execute_trade_async(action, symbol="BTC/USDT", strategy=None):
    bybit = ccxt.bybit(
        {
            "apiKey": os.getenv("BYBIT_API_KEY", "YOUR_API_KEY"),
            "secret": os.getenv("BYBIT_API_SECRET", "YOUR_SECRET"),
            "test": True,
            "enableRateLimit": True,
        }
    )

    try:
        balance = await bybit.fetch_balance()
        free_usdt = balance["USDT"]["free"]
        max_loss = free_usdt * strategy.get("max_loss_pct", 0.1)

        ticker = await bybit.fetch_ticker(symbol)
        price = ticker["last"]

        order = None
        if action == 1 and free_usdt > 100:  # Buy
            volume = min(free_usdt * strategy.get("buy_pct", 0.05), strategy.get("max_volume", 0.01))
            order = await bybit.create_limit_buy_order(symbol, volume, price)
            logger.info(f"Buy order placed: {order}")
        elif action == 2:
            balance_btc = balance["BTC"]["free"]
            if balance_btc > 0.0001:
                order = await bybit.create_market_sell_order(symbol, balance_btc)
                logger.info(f"Sell order placed: {order}")

        # Обновляем CSV после сделки
        if order:
            update_historical_csv(order, symbol)
    except Exception as e:
        logger.error(f"Trade error: {e}")
    finally:
        await bybit.close()
    return order

async def main_loop(strategy_name):
    strategy = get_strategy(strategy_name)
    obs = env.reset()
    last_sentiment = 0.0
    step_count = 0
    training_data = []  # Для онлайн-обучения

    # WebSocket для реального времени (упрощён для примера)
    ws = await ccxt.bybit().watch_ticker("BTC/USDT")

    while True:
        try:
            start_cycle = time.perf_counter()

            # Получаем новости
            news_list = await fetch_news_async()
            sentiments = []
            for news in news_list:
                text = f"{news['title']} {news['description']}"
                sentiment = analyze_news_sentiment(text)
                sentiments.append(sentiment)
                await r.setex(f"sentiment:{news['id']}", 3600, sentiment)

            avg_sentiment = np.mean(sentiments) if sentiments else 0.0

            # Обновляем obs с реальными данными
            obs = await env.update_obs_live_async(avg_sentiment, strategy)

            # Проверяем триггер (изменение цены/настроения)
            if abs(avg_sentiment - last_sentiment) > strategy.get("sentiment_threshold", 0.2):
                action = await predict_price_async(obs, strategy)
                await execute_trade_async(action, strategy=strategy)

                # Онлайн-обучение: сохраняем данные для обучения
                training_data.append((obs, action, 0.1))  # Награда placeholder
                if len(training_data) >= strategy.get("train_batch", 100):
                    # Обучаем модель на новых данных
                    for obs_t, action_t, reward_t in training_data:
                        env.step(action_t)  # Симулируем шаг для обучения
                    model.learn(total_timesteps=len(training_data))
                    model.save(f"models/ppo_{strategy_name}_updated.zip")
                    training_data = []  # Сброс
                    logger.info("Model updated online")

            last_sentiment = avg_sentiment
            step_count += 1
            logger.info(f"Cycle time: {time.perf_counter() - start_cycle:.4f}s, Sentiment: {avg_sentiment:.2f}")

            await asyncio.sleep(strategy.get("cycle_pause", 10))
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            await asyncio.sleep(10)

async def main(strategy_name):
    global model, env
    strategy = get_strategy(strategy_name)
    env = TradingEnv(strategy=strategy)
    model_path = f"models/ppo_{strategy_name}.zip"
    if os.path.exists(model_path):
        model = PPO.load(model_path)
    else:
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=strategy.get("initial_train_steps", 10000))
        model.save(model_path)

    logger.info(f"Trading bot with {strategy_name} strategy running...")
    await main_loop(strategy_name)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--strategy", default="scalping", choices=["scalping", "pipsing", "intraday"], help="Choose trading strategy")
    args = parser.parse_args()
    asyncio.run(main(args.strategy))


def auto_select_strategy():
    """Выбирает стратегию на основе волатильности и тренда."""
    # Загружаем последние данные
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH).tail(100)  # Последние 100 баров
        if len(df) < 50:
            return "intraday"  # По умолчанию, если мало данных
        
        # Волатильность: std отклонение цены
        volatility = df["close"].pct_change().std()
        # Тренд: разница EMA
        ema_diff = df["ema"].diff().mean()
        
        if volatility > 0.02:  # Высокая волатильность -> scalping
            return "scalping"
        elif abs(ema_diff) > 0.001:  # Сильный тренд -> intraday
            return "intraday"
        else:  # Средняя -> pipsing
            return "pipsing"
    return "intraday"  # Fallback

async def main(strategy_name=None):
    global model, env
    if not strategy_name:
        strategy_name = auto_select_strategy()
        print(f"Auto-selected strategy: {strategy_name}")
    strategy = get_strategy(strategy_name)
    env = TradingEnv(strategy=strategy)
    model_path = f"models/ppo_{strategy_name}.zip"
    if os.path.exists(model_path):
        model = PPO.load(model_path)
    else:
        # Если модель не найдена, обучаем (хотя run_bot.py должен это обработать)
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=strategy.get("initial_train_steps", 10000))
        model.save(model_path)

    logger.info(f"Trading bot with {strategy_name} strategy running...")
    await main_loop(strategy_name)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--strategy", choices=["scalping", "pipsing", "intraday"], help="Override auto-selection")
    args = parser.parse_args()
    asyncio.run(main(args.strategy))