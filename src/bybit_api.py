import logging
from datetime import datetime
from typing import Optional

import ccxt.async_support as ccxt
import pandas as pd

from .redis_client import RedisClient


class BybitAPI:
    def __init__(self):
        self.exchange = None
        self.redis = RedisClient()
        self.logger = logging.getLogger(__name__)

    async def initialize(self, api_key: str, api_secret: str):
        """Initialize Bybit connection"""
        try:
            self.exchange = ccxt.bybit(
                {
                    "apiKey": api_key,
                    "secret": api_secret,
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                }
            )
            await self.exchange.load_markets()
            self.logger.info("Bybit API initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Bybit API: {e}")
            raise

    def _process_ohlcv(self, ohlcv_data):
        """Process OHLCV data into DataFrame"""
        df = pd.DataFrame(
            ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    async def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100):
        """Get OHLCV data with Redis caching"""
        cache_key = f"{symbol}:{timeframe}:{limit}"

        # Try to get from cache
        cached_data = self.redis.load_market_data(cache_key)
        if cached_data is not None:
            self.logger.debug(f"Using cached data for {cache_key}")
            return cached_data

        try:
            # Get fresh data
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = self._process_ohlcv(ohlcv)

            # Save to cache
            self.redis.save_market_data(cache_key, df)

            return df

        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data: {e}")
            raise

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
    ):
        """Create order with locking to avoid conflicts"""
        lock_name = f"order_lock:{symbol}"

        if not self.redis.acquire_lock(lock_name):
            self.logger.warning(f"Could not acquire lock for {symbol}")
            return None

        try:
            order = await self.exchange.create_order(
                symbol, order_type, side, amount, price
            )

            # Save order info to Redis
            order_data = {
                "id": order["id"],
                "symbol": symbol,
                "type": order_type,
                "side": side,
                "amount": amount,
                "price": price,
                "timestamp": datetime.now().isoformat(),
            }
            self.redis.save_trading_state(symbol, order_data)

            self.logger.info(f"Order created: {order_data}")
            return order

        except Exception as e:
            self.logger.error(f"Error creating order: {e}")
            return None
        finally:
            self.redis.release_lock(lock_name)

    async def get_balance(self):
        """Get account balance"""
        try:
            balance = await self.exchange.fetch_balance()
            return balance
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return None

    async def close(self):
        """Close exchange connection"""
        await self.exchange.close()
