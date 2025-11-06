import asyncio
import websockets
import json
import logging
import time
from news_analyzer import analyze_news_sentiment
from rl_env import TradingEnv
from stable_baselines3 import PPO
import ccxt.async_support as ccxt
import redis.asyncio as redis
import numpy as np

# Логирование для отладки
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные переменные (загружаем заранее для скорости)
model = PPO.load("models/ppo_trading_model.zip")  # Загрузи обученную модель
env = TradingEnv()
r = redis.Redis(host='localhost', port=6379, decode_responses=True)  # Redis для кэша

async def predict_price_async(obs):
    """Асинхронное предсказание действия RL (модель на CPU/GPU)."""
    start = time.perf_counter()
    action, _ = model.predict(obs)
    logger.info(f"Prediction time: {time.perf_counter() - start:.4f}s")
    return int(action)

async def execute_trade_async(action, symbol='BTC/USDT'):
    """Асинхронное выполнение трейда с лимитами."""
    bybit = ccxt.bybit({
        'apiKey': 'YOUR_API_KEY',  # Замени на свои ключи
        'secret': 'YOUR_SECRET',
        'test': True,  # Тестовый режим
        'enableRateLimit': True,
    })
    
    try:
        balance = await bybit.fetch_balance()
        free_usdt = balance['USDT']['free']
        max_loss = free_usdt * 0.1  # Лимит: не терять >10% баланса
        
        if action == 1 and free_usdt > 100:  # Buy (если достаточно средств)
            volume = min(free_usdt * 0.05, 0.01)  # Малый объём для теста
            ticker = await bybit.fetch_ticker(symbol)
            price = ticker['last']
            order = await bybit.create_limit_buy_order(symbol, volume, price)
            logger.info(f"Buy order placed: {order}")
            return order
        elif action == 2:  # Sell (продаём весь BTC)
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

async def news_handler(websocket, path):
    """Обработчик WebSocket для новостей."""
    await websocket.send("Connected to trading bot")
    last_sentiment = 0.0
    obs = env.reset()  # Начальное состояние RL
    
    while True:
        try:
            start_cycle = time.perf_counter()
            data = await websocket.recv()
            news_data = json.loads(data)
            text = news_data.get('news_text', '')
            
            # Анализ новости (~10ms)
            sentiment = analyze_news_sentiment(text)
            
            # Кэшируем в Redis
            await r.setex(f"sentiment:{news_data.get('id', 'latest')}", 3600, sentiment)
            
            # Триггер на значимое изменение (>0.2) для минимизации задержек
            if abs(sentiment - last_sentiment) > 0.2:
                # Обновляем obs асинхронно
                obs = await env.update_obs_async()
                
                # Параллельные задачи: предсказание и трейдинг
                predict_task = asyncio.create_task(predict_price_async(obs))
                action = await predict_task
                trade_task = asyncio.create_task(execute_trade_async(action))
                await trade_task  # Ждём завершения
                
                # Отправляем ответ клиенту
                response = {"sentiment": sentiment, "action": action, "cycle_time": time.perf_counter() - start_cycle}
                await websocket.send(json.dumps(response))
            
            last_sentiment = sentiment
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            break

async def main():
    """Основной асинхронный цикл."""
    # WebSocket сервер для новостей (порт 8765)
    server = await websockets.serve(news_handler, "localhost", 8765)
    logger.info("Trading bot running on ws://localhost:8765")
    
    # Параллельно запускаем backtest в фоне (если нужно)
    backtest_task = asyncio.create_task(run_backtest_async())
    
    # Держим сервер запущенным
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
