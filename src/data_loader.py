import logging

import pandas as pd

from .bybit_api import BybitAPI
from .redis_client import RedisClient


class DataLoader:
    def __init__(self):
        self.redis = RedisClient()
        self.api = BybitAPI()
        self.logger = logging.getLogger(__name__)

    async def initialize(self, api_key: str, api_secret: str):
        """Initialize data loader"""
        await self.api.initialize(api_key, api_secret)

    async def get_market_data(self, symbol: str, timeframe: str, limit: int = 100):
        """Get market data with caching"""
        try:
            data = await self.api.get_ohlcv(symbol, timeframe, limit)
            return data
        except Exception as e:
            self.logger.error(f"Error loading market data: {e}")
            raise

    async def get_historical_data(self, symbol: str, timeframe: str, days: int = 30):
        """Get historical data"""
        limit = days * 24  # Approximate number of candles
        return await self.get_market_data(symbol, timeframe, limit)

    def calculate_technical_indicators(self, df: pd.DataFrame):
        """Calculate technical indicators"""
        # EMA
        df["ema_short"] = df["close"].ewm(span=12).mean()
        df["ema_long"] = df["close"].ewm(span=26).mean()

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df["close"].ewm(span=12).mean()
        exp2 = df["close"].ewm(span=26).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=9).mean()

        return df

    async def close(self):
        """Close connections"""
        await self.api.close()
