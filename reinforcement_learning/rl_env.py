import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Any

from src.indicators import add_indicators, normalize_data
from config import Config

logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0):
        super(TradingEnv, self).__init__()
        
        self.df = add_indicators(df)
        self.normalized_df = normalize_data(self.df)
        self.initial_balance = initial_balance
        self.current_step = 0
        
        # Define action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(len(self.normalized_df.columns) + 2,),  # +2 for balance and position
            dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.position = 0
        self.position_price = 0
        self.current_step = 0
        self.trades = []
        self.profits = []
        
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        if self.current_step >= len(self.normalized_df):
            return np.zeros(self.observation_space.shape)
        
        market_data = self.normalized_df.iloc[self.current_step].values
        account_state = np.array([self.balance / self.initial_balance, self.position])
        
        return np.concatenate([market_data, account_state]).astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.current_step >= len(self.df) - 1:
            done = True
            return self._get_obs(), 0, done, False, {}
        
        current_price = self.df.iloc[self.current_step]['close']
        prev_balance = self.balance
        
        reward = 0
        fee = 0.001  # 0.1% trading fee
        
        # Execute action
        if action == 1:  # Buy
            if self.balance > 0 and self.position == 0:
                self.position = self.balance / current_price
                self.position_price = current_price
                self.balance = 0
                self.trades.append({'step': self.current_step, 'action': 'buy', 'price': current_price})
                
        elif action == 2:  # Sell
            if self.position > 0:
                self.balance = self.position * current_price * (1 - fee)
                profit = self.balance - prev_balance
                reward = profit / self.initial_balance
                self.position = 0
                self.position_price = 0
                self.trades.append({'step': self.current_step, 'action': 'sell', 'price': current_price, 'profit': profit})
                self.profits.append(profit)
        
        # Calculate unrealized PnL
        unrealized_pnl = 0
        if self.position > 0:
            unrealized_pnl = (current_price - self.position_price) * self.position
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.df) - 1
        
        # Additional reward for holding profitable position
        if self.position > 0 and unrealized_pnl > 0:
            reward += unrealized_pnl / self.initial_balance * 0.1
        
        # Penalty for inactivity
        if action == 0 and self.position == 0:
            reward -= 0.001
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'unrealized_pnl': unrealized_pnl,
            'step': self.current_step
        }
        
        return self._get_obs(), reward, done, False, info

    def render(self, mode='human'):
        if mode == 'human':
            current_price = self.df.iloc[self.current_step]['close']
            print(f'Step: {self.current_step}, Price: {current_price:.2f}, '
                  f'Balance: {self.balance:.2f}, Position: {self.position:.6f}')

    def get_stats(self) -> Dict[str, Any]:
        """Get trading statistics"""
        total_trades = len([t for t in self.trades if t['action'] == 'sell'])
        profitable_trades = len([t for t in self.trades if t.get('profit', 0) > 0])
        
        return {
            'final_balance': self.balance,
            'total_return': (self.balance - self.initial_balance) / self.initial_balance,
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': profitable_trades / total_trades if total_trades > 0 else 0,
            'max_profit': max(self.profits) if self.profits else 0,
            'max_loss': min(self.profits) if self.profits else 0
        }
