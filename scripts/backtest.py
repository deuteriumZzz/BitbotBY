import backtrader as bt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from src.rl_env import TradingEnv
import os
import sys

csv_path = "data/historical_btc.csv"
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found. Run the bot first to generate data.")
    sys.exit(1)

def run_backtest():
    """Backtesting с backtrader и RL-моделью."""
    cerebro = bt.Cerebro()

    # Загружаем исторические данные
    data = bt.feeds.PandasData(
        dataname=pd.read_csv(
            "data/historical_btc.csv", index_col="timestamp", parse_dates=True
        )
    )
    cerebro.adddata(data)

    # Стратегия с RL
    class RLStrategy(bt.Strategy):
        def __init__(self):
            self.model = PPO.load("models/ppo_trading_model.zip")
            self.env = TradingEnv()  # Создаем среду, но в backtest она может не использоваться

            # Добавляем индикаторы
            self.rsi = bt.indicators.RSI(self.data.close, period=14)
            self.macd = bt.indicators.MACD(self.data.close)
            # MACD имеет несколько компонентов: macd.macd, macd.signal, macd.hist
            # Используем основную линию MACD

        def next(self):
            # Ждем, пока индикаторы будут готовы (RSI и MACD требуют достаточных данных)
            if not self.rsi or not self.macd.macd:
                return
            
            obs = np.array([
                self.rsi[0] / 100.0,  # Нормализуем RSI (0-100) к 0-1
                self.macd.macd[0] / 10.0,  # Нормализуем MACD (предполагаем деление на 10 для масштаба)
                0.0  # Sentiment (0 для backtest)
            ])
            action, _ = self.model.predict(obs)

            if action == 1 and not self.position:  # Buy
                self.buy()
            elif action == 2 and self.position:  # Sell
                self.sell()

    cerebro.addstrategy(RLStrategy)
    cerebro.broker.setcash(10000.0)
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    results = cerebro.run()
    returns_analysis = results[0].analyzers.returns.get_analysis()
    print(f"Backtest returns: {returns_analysis}")


if __name__ == "__main__":
    run_backtest()
