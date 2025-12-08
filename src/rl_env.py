import numpy as np
import pandas as pd
from src.indicators import add_indicators
import logging

logging.basicConfig(level=logging.INFO)

class TradingEnv:
    def __init__(self, data, strategy="ppo", **params):
        self.data = data
        self.strategy = strategy
        self.current_step = 0
        self.position = 0  # 0: no position, 1: long, -1: short
        self.balance = 10000  # Начальный баланс
        self.data = add_indicators(self.data.copy())  # Добавление индикаторов

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.balance = 10000
        return self._get_obs()

    def step(self, action):
        reward = 0
        current_price = self.data.iloc[self.current_step]['close']
        
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = current_price
        elif action == 2 and self.position == 0:  # Sell
            self.position = -1
            self.entry_price = current_price
        elif action == 0 and self.position != 0:  # Close
            # Расчет прибыли/убытка
            if self.position == 1:
                profit = current_price - self.entry_price
            else:
                profit = self.entry_price - current_price
                
            reward = profit / self.entry_price * 100  # ROI в процентах
            self.position = 0

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Награда за избегание убытков
        if done and self.position != 0:
            reward = -10  # Штраф за незакрытую позицию
            
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        if self.current_step >= len(self.data):
            return np.zeros(5)  # Fallback
            
        row = self.data.iloc[self.current_step]
        obs = np.array([
            row['close'], 
            row['rsi'], 
            row['macd'], 
            row['bb_upper'],
            self.position
        ])
        return obs

    def render(self):
        # Простая визуализация результатов
        print(f"Final balance: {self.balance}")
        print(f"Final position: {self.position}")
