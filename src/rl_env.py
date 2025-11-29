import numpy as np
import pandas as pd
from src.indicators import add_indicators
from src.bybit_api import BybitAPI  # Для доступа к API в live
import logging

logging.basicConfig(level=logging.INFO)

class TradingEnv:
    def __init__(self, data, strategy="ppo"):
        self.data = data
        self.strategy = strategy
        self.current_step = 0
        self.position = 0  # 0: no position, 1: long, -1: short
        self.balance = 10000  # Начальный баланс
        self.api = BybitAPI()  # Для live-обновлений
        self.data = add_indicators(self.data)  # Добавление индикаторов

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.balance = 10000
        return self._get_obs()

    def step(self, action):
        # Логика шага (упрощённая)
        reward = 0
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
        elif action == 2 and self.position == 0:  # Sell
            self.position = -1
        elif action == 0 and self.position != 0:  # Close
            self.position = 0
            reward = 10  # Placeholder reward

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        # Получение наблюдений (индикаторы + позиция)
        obs = self.data.iloc[self.current_step][['close', 'rsi', 'macd', 'bb_upper']].values
        obs = np.append(obs, self.position)
        return obs

    async def update_obs_live_async(self, sentiment, strategy):
        try:
            # Загрузка новых данных (пример: последние 100 свечей)
            new_data = await self.api.fetch_historical_data_async("BTC/USDT", "1h", limit=100)
            # Обновление данных: конкатенация и обрезка до последних 1000
            self.data = pd.concat([self.data, new_data]).tail(1000)
            self.data = add_indicators(self.data)  # Пересчёт индикаторов

            # Обновление current_step (например, на последний)
            self.current_step = len(self.data) - 1

            # Получение obs + добавление sentiment
            obs = self._get_obs()
            obs = np.append(obs, sentiment)
            return obs
        except Exception as e:
            logging.error(f"Error updating obs: {e}")
            return self._get_obs()  # Fallback
