import backtrader as bt
import asyncio
import pandas as pd
from rl_env import TradingEnv
from stable_baselines3 import PPO

async def run_backtest_async():
    """Асинхронный backtesting с backtrader и RL-моделью."""
    cerebro = bt.Cerebro()
    
    # Загружаем исторические данные (пример; используй свои CSV)
    data = bt.feeds.PandasData(dataname=pd.read_csv('data/historical_btc.csv', index_col='timestamp', parse_dates=True))
    cerebro.adddata(data)
    
    # Стратегия с RL
    class RLStrategy(bt.Strategy):
        def __init__(self):
            self.model = PPO.load("models/ppo_trading_model.zip")
            self.env = TradingEnv()
        
        def next(self):
            obs = np.array([self.data.rsi[0] / 100.0, self.data.macd[0] / 10.0])  # Индикаторы
            action, _ = self.model.predict(obs)
            
            if action == 1 and not self.position:  # Buy
                self.buy()
            elif action == 2 and self.position:  # Sell
                self.sell()
    
    cerebro.addstrategy(RLStrategy)
    cerebro.broker.setcash(10000.0)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    results = cerebro.run()
    print(f"Backtest returns: {results[0].analyzers.returns.get_analysis()}")
