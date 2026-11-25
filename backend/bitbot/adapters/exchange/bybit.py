"""
Клиент Bybit API с поддержкой retry, кэширования через Redis и управления ордерами.

Обеспечивает подключение к бирже Bybit через ccxt, получение данных OHLCV
с Redis-кэшем, создание и отмену ордеров, получение баланса и позиций.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Optional

import ccxt.async_support as ccxt
import pandas as pd
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from bitbot.adapters.exchange import constants as ex
from bitbot.adapters.exchange.paper import PaperExchange, Realism
from bitbot.config.settings import Config
from bitbot.domain.order_states import (
    OrderOutcome,
    Placed,
    Rejected,
    Unknown,
    classify_order,
)
from bitbot.storage.redis_client import RedisClient


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


def _new_client_order_id() -> str:
    """Генерирует clientOrderId (Bybit orderLinkId) для одной логической попытки
    размещения ордера. Используется ОДИН И ТОТ ЖЕ id на все повторы одного и
    того же ордера — иначе ccxt сам сгенерирует новый uuid на каждый вызов, и
    серверная дедупликация Bybit по orderLinkId не сможет поймать повторную
    отправку потерянного ответа."""
    return f"bb-{uuid.uuid4().hex}"


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

    def __init__(self, runtime_config: "Any | None" = None) -> None:
        self.exchange: Optional[ccxt.Exchange] = None
        self.redis = RedisClient()
        self.logger = logging.getLogger(__name__)
        # Paper: торговые вызовы исполняет симулятор, данные — настоящие (см. paper.py).
        self.paper = PaperExchange(
            self.get_current_price,
            Realism(Config.PAPER_SLIPPAGE_PCT, Config.PAPER_LATENCY_S),
        )
        self._leverage_cache: dict[str, int] = {}  # {symbol: last_set_leverage}
        self._runtime_config = runtime_config

    def _current_market_type(self) -> str:
        """Тип рынка: из RuntimeConfig (переключаемо на лету) либо статичный Config."""
        rc = getattr(self, "_runtime_config", None)
        if rc is not None:
            return rc.get_market_type()
        return Config.MARKET_TYPE

    def _ccxt_options(self) -> dict:
        """Подпись запросов устойчива к расхождению часов: ccxt сверяет время с
        биржей (adjustForTimeDifference), окно приёма подписи — 5 с (recvWindow)."""
        return {
            "defaultType": self._current_market_type(),
            "adjustForTimeDifference": True,
            "recvWindow": ex.RECV_WINDOW_MS,
        }

    def market_type(self) -> str:
        """Текущий тип рынка (spot | linear | ...), публичный доступ для сверки."""
        return self._current_market_type()

    def sync_market_type(self) -> None:
        """Синхронизирует defaultType уже подключённого ccxt-клиента с текущим
        рантайм-режимом — вызывать перед ордер-специфичными вызовами биржи,
        чтобы переключение spot/futures через Telegram подхватывалось без
        пересоздания соединения."""
        if self.exchange is not None:
            self.exchange.options["defaultType"] = self._current_market_type()

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
                        "timeout": ex.CCXT_TIMEOUT_MS,
                        "options": self._ccxt_options(),
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
                        "timeout": ex.CCXT_TIMEOUT_MS,
                        "options": self._ccxt_options(),
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
        except ccxt.BadSymbol as e:
            self.logger.warning("Символ недоступен на этой бирже: %s", e)
            raise
        except Exception as e:
            self.logger.error(
                "Неожиданная ошибка при получении OHLCV: %s", e, exc_info=True
            )
            raise

    async def _find_order_by_client_id(
        self, symbol: str, client_order_id: str
    ) -> Optional[dict]:
        """Ищет ордер по clientOrderId среди открытых и закрытых.

        Используется перед повтором после сетевой ошибки/дублирования — чтобы
        не отправить второй ордер поверх уже исполненного первого. Аккаунт на
        UTA (Unified Trading Account) — fetch_order/fetch_orders там не
        работают без биржевого id, поэтому ищем через fetch_open_orders/
        fetch_closed_orders с фильтром по orderLinkId.

        :param symbol: Символ торговой пары.
        :param client_order_id: clientOrderId, переданный при создании ордера.
        :return: Словарь ордера если найден, иначе None (в т.ч. при ошибке
            самого поиска — fail-open, чтобы не блокировать основной retry).
        """
        try:
            for o in await self.exchange.fetch_open_orders(
                symbol, params={"orderLinkId": client_order_id}
            ):
                if o.get("clientOrderId") == client_order_id:
                    return o
            for o in await self.exchange.fetch_closed_orders(
                symbol, params={"orderLinkId": client_order_id}
            ):
                if o.get("clientOrderId") == client_order_id:
                    return o
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.warning(
                "Поиск ордера по clientOrderId %s (%s) не удался: %s",
                client_order_id,
                symbol,
                e,
            )
        return None

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        lock_suffix: str = "trade",
        reduce_only: bool = False,
    ) -> OrderOutcome:
        """
        Создаёт ордер с Redis-блокировкой для предотвращения конфликтов.

        Повторяет попытку до 3 раз с экспоненциальной задержкой. Все попытки
        одного вызова используют один и тот же clientOrderId — если ответ на
        успешно исполненный ордер потеряется (таймаут/сетевой сбой), перед
        повтором ищем уже созданный ордер по этому id вместо слепой повторной
        отправки, которая задвоила бы позицию.

        :param symbol: Символ торговой пары.
        :param order_type: Тип ордера ("limit" или "market").
        :param side: Сторона ордера ("buy" или "sell").
        :param amount: Количество актива.
        :param price: Цена (для лимитных ордеров).
        :param reduce_only: Ордер только закрывает позицию. На фьючерсах
            ставит reduceOnly — если позицию уже закрыл биржевой SL/TP, ордер
            не откроет встречную. На споте не применяется.
        :return: Placed(ответ биржи) | Rejected(причина) — ордера точно нет |
            Unknown(clientOrderId) — сеть оборвалась, исход выяснит сверка.
        """
        if Config.PAPER_TRADING:
            return await self.paper.create_order(
                symbol, order_type, side, amount, price, lock_suffix, reduce_only
            )
        order_params: dict = {}
        if reduce_only and self._current_market_type() != "spot":
            order_params["reduceOnly"] = True
        lock_name = f"order_{lock_suffix}:{symbol}"
        if not self.redis.acquire_lock(lock_name):
            self.logger.warning("Не удалось захватить блокировку для %s", symbol)
            return Rejected("по символу уже идёт другой ордер (блокировка)")
        client_order_id = _new_client_order_id()
        try:
            last_error: Optional[Exception] = None
            for attempt in range(ex.ORDER_ATTEMPTS):
                try:
                    order = await self.exchange.create_order(
                        symbol,
                        order_type,
                        side,
                        amount,
                        price,
                        params={"clientOrderId": client_order_id, **order_params},
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
                    return Placed(order)
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
                    return Rejected(f"недостаточно средств: {e}")
                except (ccxt.BadRequest, ccxt.InvalidOrder) as e:
                    last_error = e
                    if "duplicate" in str(e).lower():
                        self.logger.warning(
                            "Bybit сообщил о дубликате clientOrderId %s для %s — "
                            "ищу уже созданный ордер вместо повтора: %s",
                            client_order_id,
                            symbol,
                            e,
                        )
                        existing = await self._find_order_by_client_id(
                            symbol, client_order_id
                        )
                        if existing is not None:
                            self.redis.save_trading_state(
                                symbol,
                                {
                                    "id": existing.get("id"),
                                    "symbol": symbol,
                                    "type": order_type,
                                    "side": side,
                                    "amount": amount,
                                    "filled": existing.get("filled", 0),
                                    "price": price,
                                    "timestamp": datetime.now().isoformat(),
                                    "status": existing.get("status", "open"),
                                },
                            )
                            self.logger.info(
                                "Найден существующий ордер по clientOrderId: %s",
                                existing,
                            )
                            return Placed(existing)
                        self.logger.error(
                            "Дубликат clientOrderId %s, но ордер не найден ни в "
                            "открытых, ни в закрытых — не повторяю с тем же id",
                            client_order_id,
                        )
                        break
                    self.logger.warning(
                        "Попытка %d/%d создания ордера не удалась (BadRequest/"
                        "InvalidOrder): %s",
                        attempt + 1,
                        ex.ORDER_ATTEMPTS,
                        e,
                    )
                    await asyncio.sleep(2**attempt)
                except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                    last_error = e
                    existing = await self._find_order_by_client_id(
                        symbol, client_order_id
                    )
                    if existing is not None:
                        self.redis.save_trading_state(
                            symbol,
                            {
                                "id": existing.get("id"),
                                "symbol": symbol,
                                "type": order_type,
                                "side": side,
                                "amount": amount,
                                "filled": existing.get("filled", 0),
                                "price": price,
                                "timestamp": datetime.now().isoformat(),
                                "status": existing.get("status", "open"),
                            },
                        )
                        self.logger.info(
                            "Ордер после сетевой ошибки найден по clientOrderId, "
                            "повтор не требуется: %s",
                            existing,
                        )
                        return Placed(existing)
                    wait = 2**attempt
                    self.logger.warning(
                        "Попытка %d/%d создания ордера не удалась: %s. Повтор через %dс",
                        attempt + 1,
                        ex.ORDER_ATTEMPTS,
                        e,
                        wait,
                    )
                    await asyncio.sleep(wait)
                except Exception as e:
                    last_error = e
                    self.logger.warning(
                        "Попытка %d/%d не удалась (неожиданная ошибка): %s",
                        attempt + 1,
                        ex.ORDER_ATTEMPTS,
                        e,
                    )
                    await asyncio.sleep(2**attempt)
            self.logger.error(
                "Ордер не создан после %d попыток: %s",
                ex.ORDER_ATTEMPTS,
                last_error,
                exc_info=True,
            )
            if isinstance(last_error, ccxt.NetworkError):
                # Сеть оборвалась, а биржа могла принять ордер: исход неизвестен,
                # сверка выяснит по clientOrderId.
                self._note_ambiguous(symbol, client_order_id, side, amount)
                return Unknown(client_order_id)
            return Rejected(
                str(last_error) or f"ордер не создан после {ex.ORDER_ATTEMPTS} попыток"
            )
        finally:
            self.redis.release_lock(lock_name)

    async def get_balance(self) -> Optional[dict]:
        """
        Получает баланс аккаунта.

        :return: Словарь с балансом или None при ошибке.
        """
        try:
            return await self.exchange.fetch_balance({"type": "UNIFIED"})
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

    async def fetch_positions_checked(self) -> list:
        """Позиции биржи ИЛИ исключение.

        В отличие от `fetch_positions` не выдаёт ошибку (сеть, таймаут, спот без
        API позиций) за «позиций нет»: вызывающий код сверки при пустом ответе
        иначе счёл бы все позиции бота закрытыми.
        """
        raw = await self.exchange.fetch_positions()
        if raw is None:
            raise RuntimeError("fetch_positions: пустой ответ биржи")
        return [p for p in raw if float(p.get("contracts") or 0) > 0]

    # ── Ордера с неизвестным исходом ──────────────────────────────────────────

    def _note_ambiguous(
        self, symbol: str, client_order_id: str, side: str, amount: float
    ) -> None:
        pending = self.__dict__.setdefault("ambiguous_orders", {})
        pending[client_order_id] = {
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "ts": time.time(),
        }

    async def resolve_ambiguous(self, max_age_s: float = 600.0) -> list:
        """Выясняет судьбу ордеров, чей исход после сетевой ошибки был неизвестен.

        Для каждого ищет ордер по clientOrderId. Найден — возвращает его
        состояние и убирает из списка. Не найден или поиск не удался — остаётся
        до `max_age_s`, затем снимается как UNRESOLVED (не «ордера нет»).
        """
        pending = self.__dict__.get("ambiguous_orders") or {}
        results = []
        for cid, info in list(pending.items()):
            order = await self._find_order_by_client_id(info["symbol"], cid)
            if order is not None:
                state = classify_order(order)
                pending.pop(cid, None)
                results.append(
                    {
                        "client_id": cid,
                        "symbol": info["symbol"],
                        "state": state.value,
                        "filled": float(order.get("filled") or 0),
                        "order_id": order.get("id"),
                    }
                )
            elif time.time() - info["ts"] > max_age_s:
                pending.pop(cid, None)
                results.append(
                    {
                        "client_id": cid,
                        "symbol": info["symbol"],
                        "state": "UNRESOLVED",
                        "filled": None,
                        "order_id": None,
                    }
                )
        return results

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
        if Config.PAPER_TRADING:
            return await self.paper.place_exchange_sl_tp(
                symbol, close_side, qty, sl_price, tp_price
            )
        sl_id: str | None = None
        tp_id: str | None = None

        # Биржа отклоняет цены не по шагу цены (tickSize): округляем по контракту.
        sl_price = self._round_price(symbol, sl_price)
        tp_price = self._round_price(symbol, tp_price)

        # На фьючерсном рынке ордера закрытия должны иметь reduceOnly=True,
        # чтобы не открыть новую позицию если основная уже закрыта.
        extra: dict = (
            {"reduceOnly": True} if self._current_market_type() != "spot" else {}
        )

        if sl_price and sl_price > 0:
            sl_id = await self._place_protective(
                symbol,
                "stop_market",
                close_side,
                qty,
                None,
                {"stopPrice": sl_price, "triggerPrice": sl_price, **extra},
                "SL",
            )

        if tp_price and tp_price > 0:
            tp_id = await self._place_protective(
                symbol, "limit", close_side, qty, tp_price, extra, "TP"
            )

        return sl_id, tp_id

    async def _place_protective(
        self,
        symbol: str,
        order_type: str,
        side: str,
        qty: float,
        price: Optional[float],
        params: dict,
        label: str,
    ) -> Optional[str]:
        """Ставит SL/TP идемпотентно: один clientOrderId на все попытки.

        Если ответ потерялся (таймаут), но биржа ордер приняла, повтор с тем же
        id Bybit отклонит как дубликат, а поиск по id вернёт уже стоящий ордер —
        второго стопа на бирже не появится. Исход неизвестен → сверка
        (resolve_ambiguous), а не «ордера нет».
        """
        # Прошлая постановка по этому символу кончилась неизвестностью: если тот
        # ордер всё-таки стоит — снимаем его, чтобы не было двух стопов.
        pending = self.__dict__.setdefault("_pending_protective", {})
        stale = pending.pop((symbol, label), None)
        if stale:
            old = await self._find_order_by_client_id(symbol, stale)
            if old is not None and old.get("status") == "open" and old.get("id"):
                await self.cancel_order(old["id"], symbol)

        client_order_id = _new_client_order_id()
        for attempt in range(ex.PROTECTIVE_ORDER_ATTEMPTS):
            try:
                order = await self.exchange.create_order(
                    symbol,
                    order_type,
                    side,
                    qty,
                    price,
                    params={**params, "clientOrderId": client_order_id},
                )
                oid = order.get("id")
                self.logger.info(
                    "Exchange %s placed: %s @ %s id=%s",
                    label,
                    symbol,
                    price or params.get("stopPrice"),
                    oid,
                )
                return oid
            except asyncio.CancelledError:
                raise
            except (ccxt.NetworkError, ccxt.InvalidOrder, ccxt.BadRequest) as e:
                network = isinstance(e, ccxt.NetworkError)
                if not network and "duplicate" not in str(e).lower():
                    self.logger.error(
                        "Failed to place exchange %s for %s: %s", label, symbol, e
                    )
                    return None
                existing = await self._find_order_by_client_id(symbol, client_order_id)
                if existing is not None:
                    self.logger.info(
                        "Exchange %s for %s already placed (found by clientOrderId)",
                        label,
                        symbol,
                    )
                    return existing.get("id")
                if not network:
                    self.logger.error(
                        "Duplicate %s clientOrderId for %s but order not found",
                        label,
                        symbol,
                    )
                    return None
                self.logger.warning(
                    "Exchange %s for %s: attempt %d/%d failed: %s",
                    label,
                    symbol,
                    attempt + 1,
                    ex.PROTECTIVE_ORDER_ATTEMPTS,
                    e,
                )
                await asyncio.sleep(ex.PROTECTIVE_RETRY_PAUSE_S)
            except Exception as e:
                self.logger.error(
                    "Failed to place exchange %s for %s: %s", label, symbol, e
                )
                return None
        self._note_ambiguous(symbol, client_order_id, side, qty)
        pending[(symbol, label)] = client_order_id
        return None

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Отменяет ордер на бирже.

        :param order_id: Идентификатор ордера.
        :param symbol: Символ торговой пары.
        :return: True если ордер отменён, False при ошибке.
        """
        if Config.PAPER_TRADING:
            return await self.paper.cancel_order(order_id, symbol)
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

    def _market_for(self, symbol: str) -> Optional[tuple]:
        """(ключ, рынок ccxt) для символа; на фьючерсах сначала 'SYM:USDT'.

        Берётся из уже загруженных `exchange.markets` — без запросов к бирже.
        """
        markets = getattr(self.exchange, "markets", None) or {}
        candidates = (
            [f"{symbol}:USDT", symbol]
            if self._current_market_type() != "spot"
            else [symbol]
        )
        for key in candidates:
            if key in markets:
                return key, markets[key]
        return None

    def get_contract_spec(self, symbol: str) -> Optional[dict]:
        """Спецификация контракта из загруженных рынков.

        `tick_size`/`qty_step` — как отдаёт ccxt (число знаков или шаг, зависит от
        precisionMode биржи); `funding_interval_min` — фактический интервал
        funding конкретного контракта (минуты), его нельзя считать 8 часами.
        """
        found = self._market_for(symbol)
        if not found:
            return None
        key, market = found
        precision = market.get("precision") or {}
        amount_limits = (market.get("limits") or {}).get("amount") or {}
        return {
            "symbol": key,
            "type": market.get("type"),
            "tick_size": precision.get("price"),
            "qty_step": precision.get("amount"),
            "min_qty": amount_limits.get("min"),
            "funding_interval_min": (market.get("info") or {}).get("fundingInterval"),
        }

    def _round_price(self, symbol: str, price: float) -> float:
        """Округляет цену до шага цены контракта; при любой проблеме — как есть."""
        found = self._market_for(symbol)
        if not found or not price:
            return price
        try:
            return float(self.exchange.price_to_precision(found[0], price))
        except Exception as exc:
            self.logger.warning("round_price failed for %s: %s", symbol, exc)
            return price

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
        Для спота — no-op. Для paper — симулятор (всегда успешно).
        Кэш per-symbol: если плечо уже выставлено в это значение — пропускаем.

        :param symbol: Торговый символ.
        :param leverage: Плечо (1 = без плеча).
        :return: True всегда (исключение не роняет бота).
        """
        if self._current_market_type() == "spot":
            return True
        if Config.PAPER_TRADING:
            return await self.paper.set_leverage(symbol, leverage)
        if self._leverage_cache.get(symbol) == leverage:
            return True
        try:
            await self.exchange.set_leverage(leverage, symbol)
            self.logger.debug("Leverage set to %dx for %s", leverage, symbol)
        except Exception as e:
            # 110043 «leverage not modified» — плечо уже такое, это не ошибка.
            if (
                ex.LEVERAGE_NOT_MODIFIED not in str(e)
                and "not modified" not in str(e).lower()
            ):
                # Иначе на бирже осталось прежнее плечо: позиция с ним — другой риск.
                self.logger.warning(
                    "set_leverage %s → %dx failed: %s", symbol, leverage, e
                )
                return False
            self.logger.debug("set_leverage %s: %s", symbol, e)
        self._leverage_cache[symbol] = leverage
        return True

    async def close(self) -> None:
        """Закрывает соединение с биржей."""
        if self.exchange is not None:
            await self.exchange.close()
