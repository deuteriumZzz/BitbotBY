import logging
import os

import ccxt.async_support as ccxt
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)


class BybitAPI:
    def __init__(self):
        self.exchange = ccxt.bybit(
            {
                "apiKey": os.getenv("BYBIT_API_KEY"),
                "secret": os.getenv("BYBIT_API_SECRET"),
                "enableRateLimit": True,
            }
        )

    async def fetch_historical_data_async(self, symbol, timeframe, limit=1000):
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    async def place_order_async(self, symbol, side, amount, price=None):
        try:
            order = await self.exchange.create_order(symbol, "market", side, amount)
            logging.info(f"Order placed: {order}")
            return order
        except Exception as e:
            logging.error(f"Error placing order: {e}")

    async def close(self):
        await self.exchange.close()
