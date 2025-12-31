import logging
from datetime import datetime
from typing import Optional

import ccxt.async_support as ccxt
import pandas as pd

from .redis_client import RedisClient


class BybitAPI:
    """
    Класс для взаимодействия с API Bybit.
    Обеспечивает подключение, получение данных OHLCV, создание ордеров,
    управление балансом и кэшированием данных через Redis.
    """

    def __init__(self):
        """
        Инициализирует объект BybitAPI с подключением к Redis и настройкой логгера.
        """
        self.exchange = None
        self.redis = RedisClient()
        self.logger = logging.getLogger(__name__)

    async def initialize(self, api_key: str, api_secret: str, testnet: bool = False):
        """
        Инициализирует подключение к API Bybit.
        Поддерживает тестнет для безопасного тестирования.

        :param api_key: API-ключ Bybit.
        :param api_secret: Секретный ключ Bybit.
        :param testnet: Флаг использования тестнета (по умолчанию False).
        :raises Exception: В случае ошибки инициализации.
        """
        try:
            # Новое: поддержка тестнета для безопасного тестирования
            exchange_class = ccxt.bybit
            if testnet:
                self.exchange = exchange_class(
                    {
                        "apiKey": api_key,
                        "secret": api_secret,
                        "enableRateLimit": True,
                        "options": {"defaultType": "spot"},
                        "urls": {
                            "api": {
                                "public": "https://api-testnet.bybit.com",
                                "private": "https://api-testnet.bybit.com",
                            }
                        },
                    }
                )
            else:
                self.exchange = exchange_class(
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
        """
        Обрабатывает данные OHLCV в DataFrame pandas.

        :param ohlcv_data: Сырые данные OHLCV.
        :return: DataFrame с обработанными данными.
        """
        df = pd.DataFrame(
            ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    async def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100):
        """
        Получает данные OHLCV с кэшированием в Redis.
        Если данные есть в кэше, возвращает их; иначе загружает свежие и сохраняет в кэш.

        :param symbol: Символ торговой пары (например, "BTC/USDT").
        :param timeframe: Таймфрейм (по умолчанию "1h").
        :param limit: Количество свечей (по умолчанию 100).
        :return: DataFrame с данными OHLCV.
        :raises Exception: В случае ошибки получения данных.
        """
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
        """
        Создает ордер с блокировкой для избежания конфликтов.
        Использует Redis для блокировки и сохранения состояния ордера.

        :param symbol: Символ торговой пары.
        :param order_type: Тип ордера (например, "limit" или "market").
        :param side: Сторона ордера ("buy" или "sell").
        :param amount: Количество.
        :param price: Цена (опционально, для лимитных ордеров).
        :return: Информация об ордере или None в случае ошибки.
        """
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
        """
        Получает баланс аккаунта.

        :return: Словарь с балансом или None в случае ошибки.
        """
        try:
            balance = await self.exchange.fetch_balance()
            return balance
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return None

    # Новое: метод для получения текущей цены (используется в trading_bot.py)
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Получает текущую цену для заданного символа.

        :param symbol: Символ торговой пары.
        :return: Текущая цена или None в случае ошибки.
        """
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker["last"]
        except Exception as e:
            self.logger.error(f"Error fetching current price for {symbol}: {e}")
            return None

    async def close(self):
        """
        Закрывает подключение к обмену.
        """
        await self.exchange.close()
