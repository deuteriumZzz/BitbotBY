"""
Клиент Bybit API с поддержкой retry, кэширования через Redis и управления ордерами.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional

import ccxt.async_support as ccxt
import pandas as pd
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .redis_client import RedisClient


def _is_retryable(exc: Exception) -> bool:
    """
    Определяет, является ли исключение временной сетевой ошибкой биржи.

    :param exc: Исключение для проверки.
    :return: True если ошибка временная и допускает повтор запроса.
    """
    return isinstance(
        exc,
        (
            ccxt.NetworkError,
            ccxt.RequestTimeout,
            ccxt.ExchangeNotAvailable,
        ),
    )


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
async def _fetch_ohlcv_with_retry(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since: Optional[int],
    limit: int,
) -> list:
    """
    Выполняет запрос OHLCV с автоматическим повтором при временных ошибках сети.

    :param exchange: Экземпляр ccxt-биржи.
    :param symbol: Символ торговой пары.
    :param timeframe: Таймфрейм.
    :param since: Начальная метка времени (мс) или None.
    :param limit: Максимальное количество свечей.
    :return: Список OHLCV-данных.
    :raises ccxt.NetworkError: При неустранимой сетевой ошибке.
    :raises ccxt.AuthenticationError: При ошибке авторизации (не повторяется).
    """
    return await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)


class BybitAPI:
    """
    Клиент Bybit API с поддержкой retry, кэширования через Redis и управления ордерами.

    Обеспечивает подключение, получение данных OHLCV, создание ордеров,
    управление балансом и кэширование данных через Redis.
    """

    def __init__(self) -> None:
        """
        Инициализирует объект BybitAPI с подключением к Redis и настройкой логгера.
        """
        self.exchange: Optional[ccxt.Exchange] = None
        self.redis = RedisClient()
        self.logger = logging.getLogger(__name__)

    async def initialize(
        self, api_key: str, api_secret: str, testnet: bool = False
    ) -> None:
        """
        Инициализирует подключение к API Bybit.

        Поддерживает тестнет для безопасного тестирования.

        :param api_key: API-ключ Bybit.
        :param api_secret: Секретный ключ Bybit.
        :param testnet: Флаг использования тестнета (по умолчанию False).
        :raises ccxt.AuthenticationError: При неверных ключах API.
        :raises ccxt.NetworkError: При недоступности биржи.
        :raises Exception: В случае иной ошибки инициализации.
        """
        try:
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
            self.logger.info("Bybit API инициализирован успешно")
        except ccxt.AuthenticationError as e:
            self.logger.critical(f"Ошибка авторизации API — проверьте ключи: {e}")
            raise
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
            self.logger.error(f"Биржа недоступна при инициализации: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Ошибка инициализации Bybit API: {e}", exc_info=True)
            raise

    def _process_ohlcv(self, ohlcv_data: list) -> pd.DataFrame:
        """
        Преобразует сырые данные OHLCV в DataFrame pandas.

        :param ohlcv_data: Сырые данные OHLCV от ccxt.
        :return: DataFrame с колонками timestamp, open, high, low, close, volume.
        """
        df = pd.DataFrame(
            ohlcv_data,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    async def get_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 100
    ) -> pd.DataFrame:
        """
        Получает данные OHLCV с кэшированием в Redis.

        Если данные есть в кэше — возвращает их; иначе загружает свежие
        через retry-обёртку и сохраняет в кэш.

        :param symbol: Символ торговой пары (например, "BTC/USDT").
        :param timeframe: Таймфрейм (по умолчанию "1h").
        :param limit: Количество свечей (по умолчанию 100).
        :return: DataFrame с данными OHLCV.
        :raises ccxt.AuthenticationError: При неверных ключах.
        :raises ccxt.NetworkError: После исчерпания попыток retry.
        """
        cache_key = f"{symbol}:{timeframe}:{limit}"

        cached_data = self.redis.load_market_data(cache_key)
        if cached_data is not None:
            self.logger.debug(f"Кэш Redis: {cache_key}")
            return cached_data

        try:
            ohlcv = await _fetch_ohlcv_with_retry(
                self.exchange, symbol, timeframe, None, limit
            )
            df = self._process_ohlcv(ohlcv)
            self.redis.save_market_data(cache_key, df)
            return df
        except asyncio.CancelledError:
            raise
        except ccxt.AuthenticationError as e:
            self.logger.critical(f"Ошибка авторизации API: {e}")
            raise
        except ccxt.InsufficientFunds as e:
            self.logger.error(f"Недостаточно средств: {e}")
            raise
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable) as e:
            self.logger.warning(f"Временная ошибка сети при получении OHLCV: {e}")
            raise
        except Exception as e:
            self.logger.error(
                f"Неожиданная ошибка при получении OHLCV: {e}", exc_info=True
            )
            raise

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
    ) -> Optional[dict]:
        """
        Создаёт ордер с Redis-блокировкой для предотвращения конфликтов.

        Повторяет попытку до 3 раз с экспоненциальной задержкой.

        :param symbol: Символ торговой пары.
        :param order_type: Тип ордера ("limit" или "market").
        :param side: Сторона ордера ("buy" или "sell").
        :param amount: Количество актива.
        :param price: Цена (для лимитных ордеров).
        :return: Словарь с данными ордера или None при неудаче.
        """
        lock_name = f"order_lock:{symbol}"
        if not self.redis.acquire_lock(lock_name):
            self.logger.warning(f"Не удалось захватить блокировку для {symbol}")
            return None
        try:
            last_error: Optional[Exception] = None
            for attempt in range(3):
                try:
                    order = await self.exchange.create_order(
                        symbol, order_type, side, amount, price
                    )
                    order_data = {
                        "id": order["id"],
                        "symbol": symbol,
                        "type": order_type,
                        "side": side,
                        "amount": amount,
                        "filled": order.get("filled", 0),
                        "price": price,
                        "timestamp": datetime.now().isoformat(),
                        "status": order.get("status", "open"),
                    }
                    self.redis.save_trading_state(symbol, order_data)
                    self.logger.info(f"Ордер создан: {order_data}")
                    return order
                except asyncio.CancelledError:
                    raise
                except ccxt.AuthenticationError as e:
                    self.logger.critical(f"Ошибка авторизации при создании ордера: {e}")
                    raise
                except ccxt.InsufficientFunds as e:
                    self.logger.error(f"Недостаточно средств для ордера {symbol}: {e}")
                    return None
                except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                    last_error = e
                    wait = 2**attempt
                    self.logger.warning(
                        f"Попытка {attempt + 1}/3 создания ордера не удалась: {e}. "
                        f"Повтор через {wait}с"
                    )
                    await asyncio.sleep(wait)
                except Exception as e:
                    last_error = e
                    self.logger.warning(
                        f"Попытка {attempt + 1}/3 не удалась (неожиданная ошибка): {e}"
                    )
                    await asyncio.sleep(2**attempt)
            self.logger.error(
                f"Ордер не создан после 3 попыток: {last_error}", exc_info=True
            )
            return None
        finally:
            self.redis.release_lock(lock_name)

    async def get_balance(self) -> Optional[dict]:
        """
        Получает баланс аккаунта.

        :return: Словарь с балансом или None при ошибке.
        """
        try:
            return await self.exchange.fetch_balance()
        except asyncio.CancelledError:
            raise
        except ccxt.AuthenticationError as e:
            self.logger.critical(f"Ошибка авторизации при получении баланса: {e}")
            raise
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            self.logger.warning(f"Временная ошибка сети при получении баланса: {e}")
            return None
        except Exception as e:
            self.logger.error(
                f"Неожиданная ошибка при получении баланса: {e}", exc_info=True
            )
            return None

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Получает текущую рыночную цену для заданного символа.

        :param symbol: Символ торговой пары.
        :return: Текущая цена или None при ошибке.
        """
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker["last"]
        except asyncio.CancelledError:
            raise
        except ccxt.AuthenticationError as e:
            self.logger.critical(f"Ошибка авторизации при получении цены: {e}")
            raise
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            self.logger.warning(
                f"Временная ошибка сети при получении цены {symbol}: {e}"
            )
            return None
        except Exception as e:
            self.logger.error(
                f"Неожиданная ошибка при получении цены {symbol}: {e}", exc_info=True
            )
            return None

    async def fetch_order_status(self, order_id: str, symbol: str) -> Optional[dict]:
        """
        Возвращает статус ордера по его ID.

        :param order_id: Идентификатор ордера.
        :param symbol: Символ торговой пары.
        :return: Словарь с данными ордера или None при ошибке.
        """
        try:
            return await self.exchange.fetch_order(order_id, symbol)
        except asyncio.CancelledError:
            raise
        except ccxt.AuthenticationError as e:
            self.logger.critical(
                f"Ошибка авторизации при получении статуса ордера: {e}"
            )
            raise
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            self.logger.warning(
                f"Временная ошибка сети при получении ордера {order_id}: {e}"
            )
            return None
        except Exception as e:
            self.logger.error(
                f"Неожиданная ошибка fetch_order_status: {e}", exc_info=True
            )
            return None

    async def place_exchange_sl_tp(
        self,
        symbol: str,
        close_side: str,
        qty: float,
        sl_price: float,
        tp_price: float,
    ) -> tuple[str | None, str | None]:
        """
        Ставит SL (stop-market) и TP (limit) ордера на бирже.
        Возвращает (sl_order_id, tp_order_id) — None если не удалось.
        Служит защитой позиции при падении бота.
        """
        sl_id: str | None = None
        tp_id: str | None = None

        if sl_price and sl_price > 0:
            try:
                sl_order = await self.exchange.create_order(
                    symbol,
                    "stop_market",
                    close_side,
                    qty,
                    None,
                    {"stopPrice": sl_price, "triggerPrice": sl_price},
                )
                sl_id = sl_order.get("id")
                self.logger.info(
                    f"Exchange SL placed: {symbol} @ {sl_price} id={sl_id}"
                )
            except Exception as e:
                self.logger.error(f"Failed to place exchange SL for {symbol}: {e}")

        if tp_price and tp_price > 0:
            try:
                tp_order = await self.exchange.create_order(
                    symbol, "limit", close_side, qty, tp_price
                )
                tp_id = tp_order.get("id")
                self.logger.info(
                    f"Exchange TP placed: {symbol} @ {tp_price} id={tp_id}"
                )
            except Exception as e:
                self.logger.error(f"Failed to place exchange TP for {symbol}: {e}")

        return sl_id, tp_id

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Отменяет ордер на бирже.

        :param order_id: Идентификатор ордера.
        :param symbol: Символ торговой пары.
        :return: True если ордер отменён, False при ошибке.
        """
        try:
            await self.exchange.cancel_order(order_id, symbol)
            self.logger.info(f"Ордер {order_id} для {symbol} отменён")
            return True
        except asyncio.CancelledError:
            raise
        except ccxt.AuthenticationError as e:
            self.logger.critical(f"Ошибка авторизации при отмене ордера: {e}")
            raise
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            self.logger.warning(
                f"Временная ошибка сети при отмене ордера {order_id}: {e}"
            )
            return False
        except Exception as e:
            self.logger.error(f"Неожиданная ошибка cancel_order: {e}", exc_info=True)
            return False

    async def close(self) -> None:
        """
        Закрывает соединение с биржей.
        """
        if self.exchange is not None:
            await self.exchange.close()
