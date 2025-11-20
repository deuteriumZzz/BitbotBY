import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
import os
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

class TradingEnv(gym.Env):
    def __init__(self, data, strategy='ppo', initial_balance=1000, commission=0.001, leverage=1):
        super().__init__()
        self.data = data.copy()
        self.strategy = strategy
        self.initial_balance = float(os.getenv('INITIAL_BALANCE', initial_balance))
        self.balance = self.initial_balance
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0
        self.commission = float(os.getenv('COMMISSION', commission))
        self.leverage = int(os.getenv('LEVERAGE', leverage))
        self.current_step = 0

        # Пространства
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy/long, 2: sell/short
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        # Расчёт индикаторов
        self._calculate_indicators()

    def _calculate_indicators(self):
        self.data['rsi'] = ta.rsi(self.data['close'], length=14)
        macd = ta.macd(self.data['close'])
        self.data['macd'] = macd['MACD_12_26_9']
        self.data['ema'] = ta.ema(self.data['close'], length=20)
        bb = ta.bbands(self.data['close'], length=20)
        self.data['bb_upper'] = bb['BBU_20_2.0'] if bb is not None else self.data['close']
        self.data['bb_lower'] = bb['BBL_20_2.0'] if bb is not None else self.data['close']
        self.data['vol_ratio'] = self.data['volume'] / self.data['volume'].rolling(20).mean()
        self.data.bfill(inplace=True)  # Замена на bfill()

    def _get_obs(self, sentiment=0.0):
        row = self.data.iloc[self.current_step]
        price = row['close']
        rsi_norm = (row['rsi'] - 50) / 50 if not np.isnan(row['rsi']) else 0
        macd_norm = row['macd'] / price if not np.isnan(row['macd']) else 0
        ema_diff = (price - row['ema']) / price if not np.isnan(row['ema']) else 0
        bb_upper_diff = (row['bb_upper'] - price) / price if not np.isnan(row['bb_upper']) else 0
        bb_lower_diff = (price - row['bb_lower']) / price if not np.isnan(row['bb_lower']) else 0
        vol_ratio = row['vol_ratio'] if not np.isnan(row['vol_ratio']) else 1
        balance_norm = self.balance / self.initial_balance
        return np.array([rsi_norm, macd_norm, ema_diff, bb_upper_diff, bb_lower_diff, vol_ratio, balance_norm, self.position, sentiment], dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        row = self.data.iloc[self.current_step]
        price = row['close']
        reward = 0
        done = self.current_step >= len(self.data) - 1

        if action == 1 and self.position == 0:  # Buy/Long
            self.position = 1
            self.entry_price = price
            reward -= self.commission * self.balance
        elif action == 2 and self.position == 0:  # Sell/Short
            self.position = -1
            self.entry_price = price
            reward -= self.commission * self.balance
        elif action == 0 and self.position != 0:  # Hold/Sell if in position
            pnl = (price - self.entry_price) * self.position * self.leverage
            reward += pnl - self.commission * abs(pnl)
            self.balance += reward
            self.position = 0
            self.entry_price = 0
        else:  # Hold
            unrealized_pnl = (price - self.entry_price) * self.position * self.leverage if self.position != 0 else 0
            reward += unrealized_pnl * 0.01  # Малый reward за unrealized

        self.current_step += 1
        obs = self._get_obs()
        return obs, reward, done, False, {}

    async def update_obs_live_async(self, sentiment, strategy):
        # Placeholder: обновите с реальными данными из BybitAPI
        # Например, fetch новые данные и пересчитать индикаторы
        return self._get_obs(sentiment)

    def render(self):
        logging.info(f"Final balance: {self.balance}, Position: {self.position}")
