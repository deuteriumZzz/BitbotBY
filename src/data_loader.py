"""
Загрузчик рыночных данных с пагинацией, кэшированием в CSV и расчётом индикаторов.

Получает данные OHLCV из Bybit API через BybitAPI, кэширует их в Redis и CSV,
стандартизирует названия колонок и рассчитывает технические индикаторы.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import ccxt.async_support as ccxt
import pandas as pd
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .bybit_api import BybitAPI, _is_retryable
from .redis_client import RedisClient

# Максимум свечей за один ccxt-запрос (ограничение Bybit)
_FETCH_LIMIT = 200
# TTL CSV-кэша в секундах (24 часа)
_CSV_CACHE_TTL = 86_400


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
async def _fetch_batch_with_retry(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since: int,
    limit: int,
) -> list:
    """
    Выполняет один батч-запрос OHLCV с retry при сетевых ошибках.

    :param exchange: Экземпляр ccxt-биржи.
    :param symbol: Символ торговой пары.
    :param timeframe: Таймфрейм ccxt.
    :param since: Начальная метка времени в мс.
    :param limit: Максимальное количество свечей в батче.
    :return: Список OHLCV-записей.
    :raises ccxt.NetworkError: После исчерпания попыток retry.
    """
    return await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)


class DataLoader:
    """
    Загрузчик рыночных данных с поддержкой пагинации, CSV-кэша и технических индикаторов.

    Обеспечивает получение данных OHLCV из API Bybit, кэширование через Redis,
    стандартизацию данных и расчёт технических индикаторов.
    """

    def __init__(self) -> None:
        """Инициализирует DataLoader с экземплярами RedisClient и BybitAPI."""
        self.redis = RedisClient()
        self.api = BybitAPI()
        self.logger = logging.getLogger(__name__)

    async def initialize(self, api_key: str, api_secret: str) -> None:
        """
        Инициализирует DataLoader, включая подключение к API Bybit.

        :param api_key: API-ключ Bybit.
        :param api_secret: Секретный ключ Bybit.
        """
        await self.api.initialize(api_key, api_secret)

    async def get_market_data(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> pd.DataFrame:
        """
        Получает рыночные данные OHLCV с кэшированием через Redis.

        Стандартизирует названия колонок и логирует информацию о данных.

        :param symbol: Символ торговой пары (например, "BTC/USDT").
        :param timeframe: Таймфрейм (например, "1h").
        :param limit: Количество свечей (по умолчанию 100).
        :return: DataFrame с рыночными данными.
        :raises ccxt.AuthenticationError: При неверных ключах.
        :raises Exception: В случае иной ошибки загрузки данных.
        """
        try:
            data = await self.api.get_ohlcv(symbol, timeframe, limit)
            self.logger.debug("Сырые колонки: %s", data.columns.tolist())
            self.logger.debug("Размер данных: %s", data.shape)
            if not data.empty:
                self.logger.debug("Первая строка: %s", data.iloc[0].to_dict())
            data = self._standardize_column_names(data)
            return data
        except asyncio.CancelledError:
            raise
        except ccxt.AuthenticationError:
            raise
        except Exception as e:
            self.logger.error(
                "Ошибка загрузки рыночных данных: %s",
                e,
                exc_info=True,
            )
            raise

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Стандартизирует названия колонок DataFrame к ожидаемому формату.

        Автоматически определяет и переименовывает колонки по ключевым
        словам или порядку (первая — timestamp, вторая — open и т.д.).

        :param df: Исходный DataFrame.
        :return: DataFrame с стандартизированными названиями колонок.
        """
        if df.empty:
            return df

        column_mapping: dict[str, str] = {}
        cols = df.columns.tolist()

        for i, col in enumerate(cols):
            col_lower = str(col).lower()
            if any(k in col_lower for k in ("timestamp", "time", "date")):
                column_mapping[col] = "timestamp"
            elif any(k in col_lower for k in ("open", "opening")):
                column_mapping[col] = "open"
            elif any(k in col_lower for k in ("high", "highest")):
                column_mapping[col] = "high"
            elif any(k in col_lower for k in ("low", "lowest")):
                column_mapping[col] = "low"
            elif any(k in col_lower for k in ("close", "closing", "last")):
                column_mapping[col] = "close"
            elif any(k in col_lower for k in ("volume", "vol", "amount")):
                column_mapping[col] = "volume"
            elif i == 0:
                column_mapping[col] = "timestamp"
            elif i == 1:
                column_mapping[col] = "open"
            elif i == 2:
                column_mapping[col] = "high"
            elif i == 3:
                column_mapping[col] = "low"
            elif i == 4:
                column_mapping[col] = "close"
            elif i == 5:
                column_mapping[col] = "volume"

        if column_mapping:
            df = df.rename(columns=column_mapping)
            self.logger.info("Стандартизированные колонки: %s", df.columns.tolist())
        return df

    async def get_historical_data(
        self, symbol: str, timeframe: str, days: int = 30
    ) -> pd.DataFrame:
        """
        Получает исторические данные OHLCV за указанное количество дней.

        Рассчитывает лимит свечей исходя из таймфрейма и глубины истории.

        :param symbol: Символ торговой пары.
        :param timeframe: Таймфрейм ("1m", "5m", "15m", "1h", "4h", "1d" и т.д.).
        :param days: Количество дней истории (по умолчанию 30).
        :return: DataFrame с историческими данными.
        """
        timeframe_minutes = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "6h": 360,
            "8h": 480,
            "12h": 720,
            "1d": 1440,
            "3d": 4320,
            "1w": 10080,
            "1M": 43200,
        }
        minutes_per_day = 24 * 60
        timeframe_min = timeframe_minutes.get(timeframe, 60)
        candles_per_day = minutes_per_day // timeframe_min
        limit = days * candles_per_day
        return await self.get_market_data(symbol, timeframe, limit)

    # ── Paginated history + CSV cache ─────────────────────────────

    _SYMBOL_RE = re.compile(r"^[A-Z0-9]{1,20}(?:[/:][A-Z0-9]{1,20}){0,2}$")
    _TF_RE = re.compile(r"^[0-9]{1,4}[mhdwM]$")

    @staticmethod
    def _csv_cache_path(symbol: str, timeframe: str) -> str:
        """Возвращает путь к CSV-кэшу для пары символ/таймфрейм."""
        if not DataLoader._SYMBOL_RE.match(symbol):
            raise ValueError(f"Invalid symbol: {symbol!r}")
        if not DataLoader._TF_RE.match(timeframe):
            raise ValueError(f"Invalid timeframe: {timeframe!r}")
        safe = symbol.replace("/", "_").replace(":", "_")
        cache_dir = Path("data/cache").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = (cache_dir / f"{safe}_{timeframe}.csv").resolve()
        if not str(path).startswith(str(cache_dir)):
            raise ValueError(f"Path traversal detected: {path}")
        return str(path)

    @staticmethod
    def _load_csv_cache(path: str) -> Optional[pd.DataFrame]:
        """
        Загружает CSV-кэш если файл свежее _CSV_CACHE_TTL секунд.

        :param path: Путь к файлу кэша.
        :return: DataFrame или None если кэш устарел или отсутствует.
        """
        if not os.path.exists(path):
            return None
        age = time.time() - os.path.getmtime(path)
        if age > _CSV_CACHE_TTL:
            return None
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            return df
        except (OSError, pd.errors.ParserError, ValueError):
            return None

    @staticmethod
    def _save_csv_cache(df: pd.DataFrame, path: str) -> None:
        """
        Сохраняет DataFrame в CSV-файл кэша.

        :param df: DataFrame для сохранения.
        :param path: Путь к файлу.
        """
        try:
            df.to_csv(path)
        except OSError:
            pass

    async def get_paginated_history(
        self,
        symbol: str,
        timeframe: str,
        months: int = 6,
    ) -> pd.DataFrame:
        """
        Загружает историю через пагинацию и кэширует результат в CSV.

        Алгоритм:
        1. Проверяем CSV-кэш (TTL 24ч).
        2. Если свежий — догружаем только новые свечи с последней timestamp.
        3. Если устарел — скачиваем всё заново батчами по _FETCH_LIMIT свечей.
        4. Сохраняем обновлённый CSV.
        5. Считаем индикаторы и возвращаем.

        :param symbol: Торговая пара ('BTC/USDT').
        :param timeframe: Таймфрейм ccxt ('15m', '1h', ...).
        :param months: Глубина истории в месяцах (по умолчанию 6).
        :return: DataFrame с OHLCV и техническими индикаторами.
        """
        cache_path = self._csv_cache_path(symbol, timeframe)
        cached = self._load_csv_cache(cache_path)

        now_ms = int(datetime.now().timestamp() * 1000)
        since_ms = int(
            (datetime.now() - timedelta(days=months * 30)).timestamp() * 1000
        )

        if cached is not None and not cached.empty:
            last_ts = int(cached["timestamp"].max())
            self.logger.info(
                "CSV-кэш найден для %s %s. Загружаем только новые свечи.",
                symbol,
                timeframe,
            )
            new_df = await self._fetch_batches(
                symbol, timeframe, since_ms=last_ts + 1, until_ms=now_ms
            )
            if not new_df.empty:
                combined = (
                    pd.concat([cached, new_df], ignore_index=True)
                    .drop_duplicates(subset=["timestamp"])
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
            else:
                combined = cached
        else:
            self.logger.info(
                "Загружаем историю %dм для %s %s (пагинация)...",
                months,
                symbol,
                timeframe,
            )
            combined = await self._fetch_batches(
                symbol, timeframe, since_ms=since_ms, until_ms=now_ms
            )

        if combined.empty:
            self.logger.warning("Данные для %s не получены", symbol)
            return combined

        self._save_csv_cache(combined, cache_path)
        self.logger.info(
            "История: %d свечей для %s (%s), сохранено в %s",
            len(combined),
            symbol,
            timeframe,
            cache_path,
        )
        return self.calculate_technical_indicators(combined)

    async def _fetch_batches(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int,
        until_ms: int,
    ) -> pd.DataFrame:
        """
        Постранично скачивает OHLCV через fetch_ohlcv(since=...).

        Каждый батч — _FETCH_LIMIT свечей. Цикл продолжается пока
        не достигнем until_ms или данные не закончатся.

        :param symbol: Символ торговой пары.
        :param timeframe: Таймфрейм ccxt.
        :param since_ms: Начальная метка времени в мс.
        :param until_ms: Конечная метка времени в мс.
        :return: Объединённый DataFrame со всеми загруженными свечами.
        """
        all_frames: list[pd.DataFrame] = []
        current = since_ms

        while current < until_ms:
            try:
                raw = await _fetch_batch_with_retry(
                    self.api.exchange, symbol, timeframe, current, _FETCH_LIMIT
                )
            except asyncio.CancelledError:
                raise
            except ccxt.AuthenticationError as e:
                self.logger.critical(
                    "Ошибка авторизации при пагинации %s: %s", symbol, e
                )
                raise
            except (
                ccxt.NetworkError,
                ccxt.RequestTimeout,
                ccxt.ExchangeNotAvailable,
            ) as e:
                self.logger.error("Сетевая ошибка при загрузке батча %s: %s", symbol, e)
                break
            except Exception as e:
                self.logger.error(
                    "Неожиданная ошибка fetch_ohlcv %s: %s", symbol, e, exc_info=True
                )
                break

            if not raw:
                break

            df_batch = pd.DataFrame(
                raw,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            all_frames.append(df_batch)

            last_ts = int(raw[-1][0])
            if last_ts <= current:
                break  # защита от бесконечного цикла
            current = last_ts + 1

            if len(raw) < _FETCH_LIMIT:
                break  # последний батч

        if not all_frames:
            return pd.DataFrame()

        result = (
            pd.concat(all_frames, ignore_index=True)
            .drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        self.logger.info("Загружено %d свечей всего", len(result))
        return result

    # ── Technical indicators ───────────────────────────────────────

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает полный набор технических индикаторов через indicators.py.

        Добавляет: RSI, MACD, Bollinger Bands, SMA 20/50, EMA 12/26, ATR,
        volatility, momentum, volume_sma, volume_ratio.
        Алиасы ema_short/ema_long добавляются для совместимости со стратегиями.

        :param df: DataFrame с колонками OHLCV.
        :return: DataFrame с добавленными техническими индикаторами.
        :raises ValueError: Если отсутствует обязательная колонка 'close'.
        :raises Exception: В случае ошибки расчёта индикаторов.
        """
        from src.indicators import add_indicators

        try:
            if "close" not in df.columns:
                raise ValueError(
                    "Обязательная колонка 'close' отсутствует. "
                    "Доступные: %s" % df.columns.tolist()
                )

            df = add_indicators(df)

            if "ema_12" in df.columns:
                df["ema_short"] = df["ema_12"]
            if "ema_26" in df.columns:
                df["ema_long"] = df["ema_26"]

            self.logger.info("Технические индикаторы рассчитаны успешно")
            return df
        except ValueError:
            raise
        except Exception as e:
            self.logger.error("Ошибка расчёта индикаторов: %s", e, exc_info=True)
            self.logger.error("Колонки DataFrame: %s", df.columns.tolist())
            self.logger.error("Размер DataFrame: %s", df.shape)
            if not df.empty:
                self.logger.error("Пример данных:\n%s", df.head(2))
            raise

    async def close(self) -> None:
        """Закрывает соединения, включая подключение к API Bybit."""
        await self.api.close()
