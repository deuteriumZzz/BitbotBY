"""
Клиент Bybit API с поддержкой retry, кэширования через Redis и управления ордерами.

Обеспечивает подключение к бирже Bybit через ccxt, получение данных OHLCV
с Redis-кэшем, создание и отмену ордеров, получение баланса и позиций.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional

import ccxt.async_support as ccxt
import pandas as pd
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from config import Config

from .redis_client import RedisClient


def _is_retryable(exc: BaseException) -> bool:
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
        self.exchange: Optional[ccxt.Exchange] = None
        self.redis = RedisClient()
        self.logger = logging.getLogger(__name__)
        self._leverage_cache: dict[str, int] = {}  # {symbol: last_set_leverage}

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
                        "options": {"defaultType": Config.MARKET_TYPE},
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
                        "options": {"defaultType": Config.MARKET_TYPE},
                    }
                )
            await self.exchange.load_markets()
            self.logger.info("Bybit API инициализирован успешно")
        except ccxt.AuthenticationError as e:
            self.logger.critical("Ошибка авторизации API — проверьте ключи: %s", e)
            raise
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
            self.logger.error("Биржа недоступна при инициализации: %s", e)
            raise
        except Exception as e:
            self.logger.error("Ошибка инициализации Bybit API: %s", e, exc_info=True)
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
            self.logger.debug("Кэш Redis: %s", cache_key)
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
            self.logger.critical("Ошибка авторизации API: %s", e)
            raise
        except ccxt.InsufficientFunds as e:
            self.logger.error("Недостаточно средств: %s", e)
            raise
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable) as e:
            self.logger.warning("Временная ошибка сети при получении OHLCV: %s", e)
            raise
        except Exception as e:
            self.logger.error(
                "Неожиданная ошибка при получении OHLCV: %s", e, exc_info=True
            )
            raise

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        lock_suffix: str = "trade",
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
        lock_name = f"order_{lock_suffix}:{symbol}"
        if not self.redis.acquire_lock(lock_name):
            self.logger.warning("Не удалось захватить блокировку для %s", symbol)
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
                    self.logger.info("Ордер создан: %s", order_data)
                    return order
                except asyncio.CancelledError:
                    raise
                except ccxt.AuthenticationError as e:
                    self.logger.critical(
                        "Ошибка авторизации при создании ордера: %s", e
                    )
                    raise
                except ccxt.InsufficientFunds as e:
                    self.logger.error(
                        "Недостаточно средств для ордера %s: %s", symbol, e
                    )
                    return None
                except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                    last_error = e
                    wait = 2**attempt
                    self.logger.warning(
                        "Попытка %d/3 создания ордера не удалась: %s. Повтор через %dс",
                        attempt + 1,
                        e,
                        wait,
                    )
                    await asyncio.sleep(wait)
                except Exception as e:
                    last_error = e
                    self.logger.warning(
                        "Попытка %d/3 не удалась (неожиданная ошибка): %s",
                        attempt + 1,
                        e,
                    )
                    await asyncio.sleep(2**attempt)
            self.logger.error(
                "Ордер не создан после 3 попыток: %s", last_error, exc_info=True
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
            self.logger.critical("Ошибка авторизации при получении баланса: %s", e)
            raise
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            self.logger.warning("Временная ошибка сети при получении баланса: %s", e)
            return None
        except Exception as e:
            self.logger.error(
                "Неожиданная ошибка при получении баланса: %s", e, exc_info=True
            )
            return None

    async def fetch_positions(self) -> list:
        """
        Возвращает список активных позиций на бирже (для реконсиляции).

        ccxt format: [{symbol, side, contracts, entryPrice, ...}]
        Фильтрует нулевые/пустые позиции на стороне клиента.

        :return: Список позиций или [] при ошибке.
        """
        try:
            raw = await self.exchange.fetch_positions() or []
            return [p for p in raw if float(p.get("contracts") or 0) > 0]
        except asyncio.CancelledError:
            raise
        except ccxt.AuthenticationError as e:
            self.logger.critical("Ошибка авторизации fetch_positions: %s", e)
            raise
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            self.logger.warning("Временная ошибка сети fetch_positions: %s", e)
            return []
        except Exception as e:
            self.logger.error("fetch_positions: %s", e, exc_info=True)
            return []

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
            self.logger.critical("Ошибка авторизации при получении цены: %s", e)
            raise
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            self.logger.warning(
                "Временная ошибка сети при получении цены %s: %s", symbol, e
            )
            return None
        except Exception as e:
            self.logger.error(
                "Неожиданная ошибка при получении цены %s: %s", symbol, e, exc_info=True
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
                "Ошибка авторизации при получении статуса ордера: %s", e
            )
            raise
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            self.logger.warning(
                "Временная ошибка сети при получении ордера %s: %s", order_id, e
            )
            return None
        except Exception as e:
            self.logger.error(
                "Неожиданная ошибка fetch_order_status: %s", e, exc_info=True
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

        :param symbol: Символ торговой пары.
        :param close_side: Сторона закрытия ("buy" или "sell").
        :param qty: Количество актива.
        :param sl_price: Цена стоп-лосса.
        :param tp_price: Цена тейк-профита.
        :return: Кортеж (sl_order_id, tp_order_id).
        """
        sl_id: str | None = None
        tp_id: str | None = None

        # На фьючерсном рынке ордера закрытия должны иметь reduceOnly=True,
        # чтобы не открыть новую позицию если основная уже закрыта.
        extra: dict = {"reduceOnly": True} if Config.MARKET_TYPE != "spot" else {}

        if sl_price and sl_price > 0:
            try:
                sl_order = await self.exchange.create_order(
                    symbol,
                    "stop_market",
                    close_side,
                    qty,
                    None,
                    {"stopPrice": sl_price, "triggerPrice": sl_price, **extra},
                )
                sl_id = sl_order.get("id")
                self.logger.info(
                    "Exchange SL placed: %s @ %s id=%s", symbol, sl_price, sl_id
                )
            except Exception as e:
                self.logger.error("Failed to place exchange SL for %s: %s", symbol, e)

        if tp_price and tp_price > 0:
            try:
                tp_order = await self.exchange.create_order(
                    symbol, "limit", close_side, qty, tp_price, extra
                )
                tp_id = tp_order.get("id")
                self.logger.info(
                    "Exchange TP placed: %s @ %s id=%s", symbol, tp_price, tp_id
                )
            except Exception as e:
                self.logger.error("Failed to place exchange TP for %s: %s", symbol, e)

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
            self.logger.info("Ордер %s для %s отменён", order_id, symbol)
            return True
        except asyncio.CancelledError:
            raise
        except ccxt.AuthenticationError as e:
            self.logger.critical("Ошибка авторизации при отмене ордера: %s", e)
            raise
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            self.logger.warning(
                "Временная ошибка сети при отмене ордера %s: %s", order_id, e
            )
            return False
        except Exception as e:
            self.logger.error("Неожиданная ошибка cancel_order: %s", e, exc_info=True)
            return False

    def round_quantity(self, symbol: str, quantity: float) -> float:
        """
        Округляет quantity до шага лота биржи и проверяет минимальный размер.

        Использует exchange.amount_to_precision() из ccxt — тот же метод что
        биржа применяет внутри create_order(). Предотвращает отклонение ордера
        из-за неправильного числа знаков.

        :param symbol: Торговый символ ('BTC/USDT').
        :param quantity: Исходное количество.
        :return: Округлённое количество, или 0.0 если меньше минимума.
        """
        if self.exchange is None or symbol not in (self.exchange.markets or {}):
            return quantity
        try:
            rounded = float(self.exchange.amount_to_precision(symbol, quantity))
            market = self.exchange.markets[symbol]
            min_qty = float(
                (market.get("limits") or {}).get("amount", {}).get("min") or 0
            )
            if min_qty and rounded < min_qty:
                self.logger.warning(
                    "round_quantity: %s qty=%.8f < min=%.8f → skip",
                    symbol,
                    rounded,
                    min_qty,
                )
                return 0.0
            return rounded
        except Exception as e:
            self.logger.warning("round_quantity failed for %s: %s", symbol, e)
            return quantity

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Устанавливает плечо для фьючерсного символа.
        Для спота — no-op. Для paper trading — no-op.
        Кэш per-symbol: если плечо уже выставлено в это значение — пропускаем.

        :param symbol: Торговый символ.
        :param leverage: Плечо (1 = без плеча).
        :return: True всегда (исключение не роняет бота).
        """
        if Config.MARKET_TYPE == "spot":
            return True
        if Config.PAPER_TRADING:
            return True
        if self._leverage_cache.get(symbol) == leverage:
            return True
        try:
            await self.exchange.set_leverage(leverage, symbol)
            self.logger.debug("Leverage set to %dx for %s", leverage, symbol)
            self._leverage_cache[symbol] = leverage
        except Exception as e:
            # Leverage already at this value — not an error
            self.logger.debug("set_leverage %s: %s", symbol, e)
            self._leverage_cache[symbol] = leverage
        return True

    async def close(self) -> None:
        """Закрывает соединение с биржей."""
        if self.exchange is not None:
            await self.exchange.close()
