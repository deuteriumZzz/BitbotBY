import backtrader as bt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.rl_env import TradingEnv


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
            self.env = TradingEnv()

        def next(self):
            obs = np.array(
                [self.data.rsi[0] / 100.0, self.data.macd[0] / 10.0, 0.0]
            )  # Индикаторы + sentiment (0 для backtest)
            action, _ = self.model.predict(obs)

            if action == 1 and not self.position:  # Buy
                self.buy()
            elif action == 2 and self.position:  # Sell
                self.sell()

    cerebro.addstrategy(RLStrategy)
    cerebro.broker.setcash(10000.0)
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    results = cerebro.run()
    print(f"Backtest returns: {results[0].analyzers.returns.get_analysis()}")


if __name__ == "__main__":
    run_backtest()
