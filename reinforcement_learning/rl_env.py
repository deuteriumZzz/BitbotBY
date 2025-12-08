import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from src.indicators import add_indicators, normalize_data
from config import Config

class TradingEnv(gym.Env):
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0):
        super(TradingEnv, self).__init__()
        
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.current_step = 0
        
        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: OHLCV + indicators
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(len(self._get_observation().flatten()),)
        )
        
        self.reset()
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        if self.current_step >= len(self.data):
            return np.zeros(self.observation_space.shape)
        
        current_data = self.data.iloc[self.current_step]
        observation = np.array([
            current_data['open'],
            current_data['high'],
            current_data['low'],
            current_data['close'],
            current_data['volume'],
            current_data.get('rsi', 50),
            current_data.get('macd', 0),
            current_data.get('macd_signal', 0),
            current_data.get('bb_upper', 0),
            current_data.get('bb_middle', 0),
            current_data.get('bb_lower', 0),
            self.balance,
            self.position,
            self.current_value
        ])
        
        return observation
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.current_step = 0
        self.done = False
        self.current_value = self.initial_balance
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        if self.done:
            return self._get_observation(), 0, True, {}
        
        current_price = self.data.iloc[self.current_step]['close']
        prev_value = self.current_value
        
        # Execute action
        if action == 1:  # Buy
            if self.balance > 0:
                self.position = self.balance / current_price
                self.entry_price = current_price
                self.balance = 0
        
        elif action == 2:  # Sell
            if self.position > 0:
                self.balance = self.position * current_price
                self.position = 0
                self.entry_price = 0
        
        # Update current portfolio value
        self.current_value = self.balance + (self.position * current_price)
        
        # Calculate reward
        reward = self.current_value - prev_value
        
        # Move to next step
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True
        
        info = {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'value': self.current_value,
            'price': current_price
        }
        
        return self._get_observation(), reward, self.done, info
    
    def render(self, mode='human'):
        """Render environment state"""
        current_price = self.data.iloc[self.current_step]['close']
        print(f"Step: {self.current_step}, Price: {current_price:.2f}, "
              f"Balance: {self.balance:.2f}, Position: {self.position:.4f}, "
              f"Value: {self.current_value:.2f}")
