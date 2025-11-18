import os
import sys
from argparse import ArgumentParser

import ccxt
import numpy as np
import pandas as pd
import ta
from stable_baselines3 import PPO

from src.news_analyzer import analyze_news_sentiment
from src.rl_env import TradingEnv
from src.strategies import get_strategy

def load_historical_data(symbol="BTC/USDT", days=30):
    """Загружает исторические OHLCV-данные с Bybit."""
    bybit = ccxt.bybit({"enableRateLimit": True})
    since = bybit.milliseconds() - (days * 24 * 60 * 60 * 1000)
    ohlcv = bybit.fetch_ohlcv(symbol, "1h", since=since, limit=1000)  # 1h бары для обучения
    bybit.close()
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def add_technical_indicators(df):
    """Добавляет тех. индикаторы."""
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    df["ema"] = ta.trend.EMAIndicator(df["close"]).ema_indicator()
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    return df.dropna()

def add_news_sentiment(df, news_list):
    """Интегрирует новостной sentiment (упрощённо: средний за период)."""
    sentiments = []
    for news in news_list:
        text = f"{news['title']} {news['description']}"
        sentiments.append(analyze_news_sentiment(text))
    avg_sentiment = np.mean(sentiments) if sentiments else 0.0
    df["sentiment"] = avg_sentiment  # Добавляем как константу для простоты; можно усреднить по времени
    return df

def main(auto=False):
    strategy_name = "scalping" if not auto else None  # Если auto, обучаем для всех стратегий
    strategies = ["scalping", "pipsing", "intraday"] if auto else [strategy_name]
    
    for strat in strategies:
        strategy = get_strategy(strat)
        env = TradingEnv(strategy=strategy)
        
        # Загружаем/создаём данные
        csv_path = "data/historical_btc.csv"
        if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
            print(f"Loading historical data for {strat}...")
            df = load_historical_data()
            df = add_technical_indicators(df)
            # Загружаем новости (упрощённо; используйте fetch_news_async из main.py)
            news_list = []  # В реале интегрируйте aiohttp
            df = add_news_sentiment(df, news_list)
            df.to_csv(csv_path, index=False)
        
        # Гибридное обучение: PPO на данных с тех. анализом и sentiment
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=strategy.get("initial_train_steps", 10000))
        model.save(f"models/ppo_{strat}.zip")
        print(f"Model for {strat} trained and saved.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="Auto-train for all strategies")
    parser.add_argument("strategy", nargs="?", default="scalping", choices=["scalping", "pipsing", "intraday"])
    args = parser.parse_args()
    main(auto=args.auto)
