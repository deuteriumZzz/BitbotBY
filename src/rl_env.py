import gym
from gym import spaces
import numpy as np
import pandas as pd
import pandas_ta as ta
import ccxt
import asyncio

class TradingEnv(gym.Env):
    def __init__(self, strategy=None):
        super(TradingEnv, self).__init__()
        self.strategy = strategy or {}
        self.bybit = ccxt.bybit({"enableRateLimit": True})
        self.symbol = "BTC/USDT"
        self.data = None
        self.current_step = 0
        self.max_steps = self.strategy.get("max_steps", 1000)

        # Расширенные индикаторы
        self.indicators = {}

        # Наблюдение: rsi, macd, ema, bb_upper, bb_lower, volume_ratio, sentiment
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell

        self.initial_balance = 10000.0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.total_value = self.initial_balance
        self.stop_loss = self.strategy.get("stop_loss_pct", 0.05)
        self.take_profit = self.strategy.get("take_profit_pct", 0.1)

    def _compute_indicators(self, df):
        self.indicators['rsi'] = ta.rsi(df['close'], length=14)
        macd = ta.macd(df['close'])
        self.indicators['macd'] = macd['MACD_12_26_9'] if macd is not None else pd.Series([0]*len(df))
        self.indicators['ema'] = ta.ema(df['close'], length=20)
        bb = ta.bbands(df['close'], length=20)
        self.indicators['bb_upper'] = bb['BBU_20_2.0'] if bb is not None else pd.Series([0]*len(df))
        self.indicators['bb_lower'] = bb['BBL_20_2.0'] if bb is not None else pd.Series([0]*len(df))
        self.indicators['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    def _get_obs(self, sentiment=0.0):
        if self.current_step >= len(self.data):
            return np.zeros(7, dtype=np.float32)
        obs = [
            self.indicators['rsi'].iloc[self.current_step] / 100.0,
            self.indicators['macd'].iloc[self.current_step] / 10.0,
            self.indicators['ema'].iloc[self.current_step] / self.data['close'].iloc[self.current_step],
            self.indicators['bb_upper'].iloc[self.current_step] / self.data['close'].iloc[self.current_step],
            self.indicators['bb_lower'].iloc[self.current_step] / self.data['close'].iloc[self.current_step],
            self.indicators['volume_ratio'].iloc[self.current_step],
            sentiment
        ]
        return np.array(obs, dtype=np.float32)

    def _calculate_reward(self, action, current_price):
        reward = 0.0
        if action == 1 and self.position == 0:  # Buy
            volume = self.balance / current_price
            self.position = volume
            self.entry_price = current_price
            self.balance = 0.0
        elif action == 2 and self.position > 0:  # Sell
            sell_value = self.position * current_price
            profit = (sell_value - (self.entry_price * self.position)) / self.initial_balance
            reward = profit * 100
            self.balance = sell_value
            self.position = 0
        else:  # Hold
            if self.position > 0:
                pnl = (current_price - self.entry_price) / self.entry_price
                if pnl <= -self.stop_loss or pnl >= self.take_profit:
                    reward = pnl * 10  # Автоматический выход
                    self.balance += self.position * current_price
                    self.position = 0
                else:
                    reward = pnl * 5
            else:
                reward = -0.01
        reward -= 0.001  # Комиссия
        return reward

    def reset(self):
        ohlcv = self.bybit.fetch_ohlcv(self.symbol, self.strategy.get("timeframe", "1m"), limit=self.max_steps)
        self.data = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        self._compute_indicators(self.data)
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.total_value = self.initial_balance
        self.current_step = 0
        return self._get_obs()

    def step(self, action):
        if self.current_step >= len(self.data) - 1:
            return self._get_obs(), 0.0, True, False, {}
        current_price = self.data['close'].iloc[self.current_step]
        reward = self._calculate_reward(action, current_price)
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        obs = self._get_obs()
        return obs, reward, done, False, {"balance": self.balance, "position": self.position}

    async def update_obs_live_async(self, sentiment=0.0, strategy=None):
        bybit_async = ccxt.bybit({"enableRateLimit": True, "asyncio_loop": asyncio.get_event_loop()})
        try:
            ohlcv = await bybit_async.fetch_ohlcv(self.symbol, strategy.get("timeframe", "1m"), limit=100)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            self._compute_indicators(df)
            obs = self._get_obs(sentiment)
            return obs
        finally:
            await bybit_async.close()

    def render(self, mode="human"):
        current_price = self.data['close'].iloc[self.current_step] if self.current_step < len(self.data) else 0
        print(f"Step: {self.current_step}, Price: {current_price:.2f}, Balance: {self.balance:.2f}, Position: {self.position:.6f}")

    def close(self):
        self.bybit.close()
