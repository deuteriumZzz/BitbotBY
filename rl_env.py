import gym
from gym import spaces
import numpy as np
import pandas as pd
import pandas_ta as ta
import ccxt  # Используем синхронную версию для совместимости с Gym

class TradingEnv(gym.Env):
    """RL-среда для трейдинга (совместима с stable-baselines3). 
    Использует исторические данные для обучения. Для live-трейдинга используйте update_obs_live_async."""
    
    def __init__(self):
        super(TradingEnv, self).__init__()
        self.bybit = ccxt.bybit({'enableRateLimit': True})
        self.symbol = 'BTC/USDT'
        self.data = None  # OHLCV данные
        self.rsi = None
        self.macd = None
        self.current_step = 0
        self.max_steps = 1000  # Максимум шагов в эпизоде
        
        # Пространства: obs - 3 элемента (rsi нормализованный, macd нормализованный, sentiment)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        
        # Состояние портфеля
        self.initial_balance = 10000.0  # USDT
        self.balance = self.initial_balance  # USDT
        self.position = 0  # 0: нет позиции, 1: long (количество BTC)
        self.entry_price = 0.0
        self.total_value = self.initial_balance  # Для расчета награды
    
    def _get_obs(self, sentiment=0.0):
        """Получает текущее наблюдение на основе текущего шага."""
        if self.current_step >= len(self.data):
            # Fallback, если данные кончились
            rsi_norm = 0.5
            macd_norm = 0.0
        else:
            rsi_norm = self.rsi.iloc[self.current_step] / 100.0
            macd_norm = self.macd.iloc[self.current_step] / 10.0  # Примерная нормализация
        return np.array([rsi_norm, macd_norm, sentiment], dtype=np.float32)
    
    def _calculate_reward(self, action, current_price):
        """Расчет награды на основе действия и изменения цены."""
        reward = 0.0
        if action == 1 and self.position == 0:  # Buy
            self.position = self.balance / current_price  # Покупаем на весь баланс
            self.entry_price = current_price
            self.balance = 0.0
            reward = 0.0  # Нет немедленной награды
        elif action == 2 and self.position > 0:  # Sell
            sell_value = self.position * current_price
            self.balance = sell_value
            self.position = 0
            profit = (sell_value - (self.entry_price * self.position)) / self.initial_balance
            reward = profit * 100  # Масштабируем для RL (пример)
        else:  # Hold или invalid action
            if self.position > 0:
                unrealized_pnl = (current_price - self.entry_price) / self.entry_price
                reward = unrealized_pnl * 10  # Небольшая награда за удержание с PnL
            else:
                reward = -0.01  # Маленький штраф за бездействие
        
        # Общая награда с учетом slippage/commission (симуляция -0.1%)
        reward -= 0.001
        return reward
    
    def _update_total_value(self, current_price):
        """Обновляет общую стоимость портфеля."""
        if self.position > 0:
            self.total_value = self.balance + (self.position * current_price)
        else:
            self.total_value = self.balance
    
    def reset(self):
        """Сброс среды: загружаем исторические данные и сбрасываем состояние."""
        # Загружаем свежие исторические данные (1m свечи)
        ohlcv = self.bybit.fetch_ohlcv(self.symbol, '1m', limit=self.max_steps)
        self.data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Вычисляем индикаторы (для всего датасета)
        self.rsi = ta.rsi(self.data['close'], length=14)
        macd_df = ta.macd(self.data['close'])
        self.macd = macd_df['MACD_12_26_9'] if macd_df is not None else pd.Series([0.0] * len(self.data))
        
        # Сбрасываем состояние
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.total_value = self.initial_balance
        self.current_step = 0
        
        return self._get_obs(sentiment=0.0)
    
    def step(self, action):
        """Выполняет шаг: обновляет состояние, рассчитывает награду."""
        if self.current_step >= len(self.data) - 1:
            done = True
            reward = 0.0
            obs = self._get_obs(0.0)
            return obs, reward, done, {}
        
        current_price = self.data['close'].iloc[self.current_step]
        self._update_total_value(current_price)
        
        # Рассчитываем награду
        reward = self._calculate_reward(action, current_price)
        
        # Переходим к следующему шагу
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        truncated = False  # Для Gym v0.26+; можно игнорировать если не нужно
        
        # Новое наблюдение
        obs = self._get_obs(sentiment=0.0)
        
        info = {'balance': self.balance, 'position': self.position, 'total_value': self.total_value}
        
        return obs, reward, done, truncated, info  # Совместимо с Gym 0.26+
    
    async def update_obs_live_async(self, sentiment=0.0):
        """Асинхронный метод для получения живого наблюдения (для inference/prod).
        Используйте async ccxt для этого."""
        bybit_async = ccxt.bybit({'enableRateLimit': True, 'asyncio_loop': asyncio.get_event_loop()})
        try:
            ohlcv = await bybit_async.fetch_ohlcv(self.symbol, '1m', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            df['rsi'] = ta.rsi(df['close'], length=14)
            macd = ta.macd(df['close'])
            macd_val = macd['MACD_12_26_9'].iloc[-1] if macd is not None and not macd.empty else 0.0
            
            rsi_norm = df['rsi'].iloc[-1] / 100.0 if not df['rsi'].empty else 0.5
            macd_norm = macd_val / 10.0
            
            obs = np.array([rsi_norm, macd_norm, sentiment], dtype=np.float32)
            return obs
        finally:
            await bybit_async.close()
    
    def render(self, mode='human'):
        """Рендеринг состояния (для отладки)."""
        current_price = self.data['close'].iloc[self.current_step] if self.data is not None and self.current_step < len(self.data) else 0
        print(f"Step: {self.current_step}, Price: {current_price:.2f}, Balance: {self.balance:.2f}, "
              f"Position: {self.position:.6f}, Total Value: {self.total_value:.2f}")
    
    def close(self):
        """Закрытие соединения."""
        if hasattr(self, 'bybit'):
            self.bybit.close()
