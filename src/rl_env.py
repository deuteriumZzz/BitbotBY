import asyncio
import ccxt
import gymnasium as gym
import numpy as np
import pandas as pd
import pandas_ta as ta
from gymnasium import spaces

class TradingEnv(gym.Env):
    """
    A trading environment for reinforcement learning using Bybit API and technical indicators.
    Actions: 0 - Hold, 1 - Buy, 2 - Sell.
    Observation: Normalized indicators + balance + position.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, strategy=None):
        super(TradingEnv, self).__init__()
        
        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: rsi, macd, ema, bb_upper, bb_lower, volume_ratio, balance, position, sentiment
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )
        
        # Bybit exchange
        self.bybit = ccxt.bybit({
            'apiKey': '',  # Add your API key here for live trading
            'secret': '',  # Add your secret
            'sandbox': True,  # Use sandbox for testing
            'enableRateLimit': True,
        })
        
        # Trading parameters
        self.symbol = strategy.get('symbol', 'BTC/USDT') if strategy else 'BTC/USDT'
        self.max_steps = strategy.get('max_steps', 1000) if strategy else 1000
        self.initial_balance = strategy.get('initial_balance', 10000) if strategy else 10000
        self.lot_size = strategy.get('lot_size', 0.001) if strategy else 0.001  # BTC lot size
        
        # State
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # 0: no position, 1: long
        self.entry_price = 0
        self.data = pd.DataFrame()
        self.indicators = {}
        
        # Reset on init
        self.reset()

    def _compute_indicators(self, df):
        """
        Compute technical indicators using pandas-ta.
        Handles empty or insufficient data with fallbacks.
        """
        if df.empty or len(df) < 20:
            print(f"Warning: Insufficient data for indicators (len={len(df)}). Using defaults.")
            default_len = len(df) if not df.empty else 20
            self.indicators = {
                "rsi": pd.Series([50] * default_len),  # Neutral RSI
                "macd": pd.Series([0] * default_len),
                "ema": pd.Series([df["close"].mean()] * default_len) if not df.empty else pd.Series([0] * default_len),
                "bb_upper": pd.Series([0] * default_len),
                "bb_lower": pd.Series([0] * default_len),
                "volume_ratio": pd.Series([1] * default_len),
            }
            return
        
        # RSI
        self.indicators["rsi"] = ta.rsi(df["close"], length=14)
        
        # MACD
        macd = ta.macd(df["close"])
        self.indicators["macd"] = (
            macd["MACD_12_26_9"] if macd is not None and not macd.empty and "MACD_12_26_9" in macd.columns
            else pd.Series([0] * len(df))
        )
        
        # EMA
        self.indicators["ema"] = ta.ema(df["close"], length=20)
        
        # Bollinger Bands (fix for KeyError)
        bb = ta.bbands(df["close"], length=20)
        if bb is not None and not bb.empty:
            if "BBU_20_2.0" in bb.columns and "BBL_20_2.0" in bb.columns:
                self.indicators["bb_upper"] = bb["BBU_20_2.0"]
                self.indicators["bb_lower"] = bb["BBL_20_2.0"]
            else:
                print(f"Warning: BB columns not found in bb DataFrame. Available columns: {list(bb.columns) if not bb.empty else 'None'}")
                self.indicators["bb_upper"] = pd.Series([df["close"].mean()] * len(df))
                self.indicators["bb_lower"] = pd.Series([df["close"].mean()] * len(df))
        else:
            print("Warning: ta.bbands returned None or empty. Using defaults.")
            self.indicators["bb_upper"] = pd.Series([df["close"].mean()] * len(df))
            self.indicators["bb_lower"] = pd.Series([df["close"].mean()] * len(df))
        
        # Volume ratio
        self.indicators["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    def _get_obs(self, sentiment=0.0):
        """
        Get the current observation as a normalized array.
        """
        if self.current_step >= len(self.data):
            # End of data
            obs = np.zeros(9)
        else:
            row = self.data.iloc[self.current_step]
            rsi = self.indicators["rsi"].iloc[self.current_step] if self.current_step < len(self.indicators["rsi"]) else 50
            macd = self.indicators["macd"].iloc[self.current_step] if self.current_step < len(self.indicators["macd"]) else 0
            ema = self.indicators["ema"].iloc[self.current_step] if self.current_step < len(self.indicators["ema"]) else row["close"]
            bb_upper = self.indicators["bb_upper"].iloc[self.current_step] if self.current_step < len(self.indicators["bb_upper"]) else row["close"]
            bb_lower = self.indicators["bb_lower"].iloc[self.current_step] if self.current_step < len(self.indicators["bb_lower"]) else row["close"]
            vol_ratio = self.indicators["volume_ratio"].iloc[self.current_step] if self.current_step < len(self.indicators["volume_ratio"]) else 1.0
            
            # Normalize some values
            rsi_norm = (rsi - 50) / 50 if rsi is not np.nan else 0
            price_norm = (row["close"] - row["close"].min()) / (row["close"].max() - row["close"].min()) if row["close"].max() != row["close"].min() else 0.5
            balance_norm = self.balance / self.initial_balance
            position_norm = self.position  # 0 or 1
            
            obs = np.array([
                rsi_norm, macd / row["close"],  # MACD relative to price
                (row["close"] - ema) / ema if ema != 0 else 0,
                (row["close"] - bb_upper) / bb_upper if bb_upper != 0 else 0,
                (row["close"] - bb_lower) / bb_lower if bb_lower != 0 else 0,
                vol_ratio,
                balance_norm,
                position_norm,
                sentiment
            ], dtype=np.float32)
        
        return obs

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        Fetches new data from Bybit or uses test data.
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        
        try:
            # Fetch OHLCV data from Bybit
            ohlcv = self.bybit.fetch_ohlcv(
                self.symbol, timeframe='1m', limit=self.max_steps
            )
            if not ohlcv:
                raise ValueError("No OHLCV data fetched from Bybit API.")
            
            self.data = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            self.data["timestamp"] = pd.to_datetime(self.data["timestamp"], unit='ms')
            print(f"Data loaded successfully: {len(self.data)} rows from Bybit.")
            print(self.data.head())  # Debug: show first few rows
        except Exception as e:
            print(f"Error fetching data from Bybit: {e}. Falling back to test data.")
            # Generate test data for debugging (replace with CSV load for production)
            np.random.seed(42)  # For reproducibility
            test_len = self.max_steps
            prices = 45000 + np.cumsum(np.random.randn(test_len) * 100)
            self.data = pd.DataFrame({
                "timestamp": pd.date_range(start='2023-01-01', periods=test_len, freq='1min'),
                "open": prices + np.random.randn(test_len) * 10,
                "high": prices + np.abs(np.random.randn(test_len) * 20),
                "low": prices - np.abs(np.random.randn(test_len) * 20),
                "close": prices,
                "volume": np.random.uniform(100, 1000, test_len),
            })
            print(f"Using test data: {len(self.data)} rows.")
        
        # Compute indicators
        self._compute_indicators(self.data)
        
        # Initial observation
        obs = self._get_obs(sentiment=0.0)
        return obs, {}  # Gymnasium returns (obs, info)

    def step(self, action):
        """
        Execute one time step within the environment.
        Returns: obs, reward, done, truncated, info
        """
        if self.current_step >= len(self.data) - 1:
            done = True
            truncated = False
            reward = 0
            obs = np.zeros(9)
            info = {"balance": self.balance, "position": self.position}
            return obs, reward, done, truncated, info
        
        current_price = self.data.iloc[self.current_step]["close"]
        self.current_step += 1
        next_price = self.data.iloc[self.current_step]["close"]
        
        # Execute action
        reward = 0
        if action == 1 and self.position == 0:  # Buy
            if self.balance > 0:
                self.position = 1
                self.entry_price = current_price
                self.balance -= current_price * self.lot_size
                reward -= 0.1  # Transaction fee
        elif action == 2 and self.position == 1:  # Sell
            if self.position == 1:
                profit = (current_price - self.entry_price) * self.lot_size
                self.balance += current_price * self.lot_size + profit
                reward = profit / self.initial_balance  # Normalized profit
                self.position = 0
                reward -= 0.1  # Fee
        # Hold (action 0): no change
        
        # Additional reward: unrealized P&L if holding
        if self.position == 1:
            unrealized = (current_price - self.entry_price) * self.lot_size
            reward += unrealized / self.initial_balance * 0.5  # Partial reward
        
        # Penalty for large drawdown
        drawdown = (self.initial_balance - self.balance) / self.initial_balance
        if drawdown > 0.1:
            reward -= drawdown * 0.5
        
        done = self.current_step >= len(self.data) - 1
        truncated = False  # No truncation in this env
        obs = self._get_obs(sentiment=0.0)  # Sentiment can be external input
        info = {
            "balance": self.balance,
            "position": self.position,
            "price": current_price,
            "step": self.current_step
        }
        
        return obs, reward, done, truncated, info

    def render(self, mode='human'):
        """
        Render the environment.
        """
        if mode == 'human':
            current_price = self.data.iloc[self.current_step]["close"] if self.current_step < len(self.data) else 0
            print(f"Step: {self.current_step}, Price: {current_price:.2f}, Balance: {self.balance:.2f}, Position: {self.position}")

# Example usage (for testing)
if __name__ == "__main__":
    strategy = {"symbol": "BTC/USDT", "max_steps": 100, "initial_balance": 10000}
    env = TradingEnv(strategy)
    obs, info = env.reset()
    print("Initial observation:", obs)
    
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done or truncated:
            break
    print("Final balance:", info["balance"])
