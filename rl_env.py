import gym
from gym import spaces
import numpy as np
import pandas as pd
import pandas_ta as ta
import ccxt.async_support as ccxt
import asyncio

class TradingEnv(gym.Env):
    """Асинхронная RL-среда для трейдинга (совместима с stable-baselines3)."""
    
    def __init__(self):
        super(TradingEnv, self).__init__()
        self.bybit = ccxt.bybit({'enableRateLimit': True})
        self.symbol = 'BTC/USDT'
        self.data = []  # Кэш OHLCV
        self.current_step = 0
        
        # Пространства: obs - 2 индикатора (rsi, macd), actions - 3 (hold, buy, sell)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        
        # Награда: простая (прибыль/убыток)
        self.initial_balance = 10000  # Стартовый баланс для симуляции
        self.balance = self.initial_balance
        self.position = 0  # 0: нет позиции, 1: long
    
    async def update_obs_async(self):
        """Асинхронно обновляет наблюдение (индикаторы)."""
        ohlcv = await self.bybit.fetch_ohlcv(self.symbol, '1m', limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Вычисляем индикаторы быстро
        df['rsi'] = ta.rsi(df['close'], length=14)
        macd = ta.macd(df['close'])
        df['macd'] = macd['MACD_12_26_9']
        
        # Последнее obs (нормализуем для RL)
        obs = np.array([df['rsi'].iloc[-1] / 100.0, df['macd'].iloc[-1] / 10.0], dtype=np.float32)
        await self.bybit.close()
        return obs
    
    def reset(self):
        """Сброс среды."""
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        return np.zeros(2, dtype=np.float32)
    
    def step(self, action):
        """Шаг RL (для обучения; в проде используем update_obs_async)."""
        # Простая симуляция (для обучения)
        reward = 0
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            reward = 1  # Награда за вход
        elif action == 2 and self.position == 1:  # Sell
            self.position = 0
            reward = 10  # Награда за выход (симулируем прибыль)
        else:
            reward = -0.1  # Штраф за hold
        
        self.current_step += 1
        done = self.current_step > 1000  # Эпизод заканчивается
        obs = np.random.rand(2)  # В обучении используй реальные данные
        return obs, reward, done, {}
    
    def render(self, mode='human'):
        pass
