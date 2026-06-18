import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from .bybit_api import BybitAPI
from .redis_client import RedisClient

# Максимум свечей за один ccxt-запрос (ограничение Bybit)
_FETCH_LIMIT = 200
# TTL CSV-кэша в секундах (24 часа)
_CSV_CACHE_TTL = 86_400


class DataLoader:
    """
    Класс для загрузки и обработки рыночных данных.
    Обеспечивает получение данных OHLCV из API Bybit, кэширование через Redis,
    стандартизацию данных и расчет технических индикаторов.
    """

    def __init__(self):
        """
        Инициализирует объект DataLoader с экземплярами RedisClient и BybitAPI.
        """
        self.redis = RedisClient()
        self.api = BybitAPI()
        self.logger = logging.getLogger(__name__)

    async def initialize(self, api_key: str, api_secret: str):
        """
        Инициализирует DataLoader, включая подключение к API Bybit.

        :param api_key: API-ключ Bybit.
        :param api_secret: Секретный ключ Bybit.
        """
        await self.api.initialize(api_key, api_secret)

    async def get_market_data(self, symbol: str, timeframe: str, limit: int = 100):
        """
        Получает рыночные данные OHLCV с кэшированием через Redis.
        Стандартизирует названия колонок и логирует информацию о данных.

        :param symbol: Символ торговой пары (например, "BTC/USDT").
        :param timeframe: Таймфрейм (например, "1h").
        :param limit: Количество свечей (по умолчанию 100).
        :return: DataFrame с рыночными данными.
        :raises Exception: В случае ошибки загрузки данных.
        """
        try:
            data = await self.api.get_ohlcv(symbol, timeframe, limit)

            # Добавляем отладку для понимания формата данных
            self.logger.info(f"Raw data columns: {data.columns.tolist()}")
            self.logger.info(f"Data shape: {data.shape}")
            if not data.empty:
                self.logger.info(f"First row sample: {data.iloc[0].to_dict()}")

            # Стандартизируем названия колонок
            data = self._standardize_column_names(data)

            return data
        except Exception as e:
            self.logger.error(f"Error loading market data: {e}")
            raise

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Стандартизирует названия колонок DataFrame к ожидаемому формату.
        Автоматически определяет и переименовывает колонки на основе ключевых слов.

        :param df: Исходный DataFrame.
        :return: DataFrame с стандартизированными названиями колонок.
        """
        if df.empty:
            return df

        column_mapping = {}
        current_columns = df.columns.tolist()

        # Автоматическое определение и маппинг колонок
        for i, col in enumerate(current_columns):
            col_lower = str(col).lower()

            if any(keyword in col_lower for keyword in ["timestamp", "time", "date"]):
                column_mapping[col] = "timestamp"
            elif any(keyword in col_lower for keyword in ["open", "opening"]):
                column_mapping[col] = "open"
            elif any(keyword in col_lower for keyword in ["high", "highest"]):
                column_mapping[col] = "high"
            elif any(keyword in col_lower for keyword in ["low", "lowest"]):
                column_mapping[col] = "low"
            elif any(keyword in col_lower for keyword in ["close", "closing", "last"]):
                column_mapping[col] = "close"
            elif any(keyword in col_lower for keyword in ["volume", "vol", "amount"]):
                column_mapping[col] = "volume"
            elif i == 0:  # первая колонка - обычно timestamp
                column_mapping[col] = "timestamp"
            elif i == 1:  # вторая - open
                column_mapping[col] = "open"
            elif i == 2:  # третья - high
                column_mapping[col] = "high"
            elif i == 3:  # четвертая - low
                column_mapping[col] = "low"
            elif i == 4:  # пятая - close
                column_mapping[col] = "close"
            elif i == 5:  # шестая - volume
                column_mapping[col] = "volume"

        if column_mapping:
            df = df.rename(columns=column_mapping)
            self.logger.info(f"Standardized columns: {df.columns.tolist()}")

        return df

    async def get_historical_data(self, symbol: str, timeframe: str, days: int = 30):
        """
        Получает исторические данные OHLCV за указанное количество дней.
        Использует метод get_market_data с расчетом лимита на основе дней и таймфрейма.

        :param symbol: Символ торговой пары.
        :param timeframe: Таймфрейм (например, "1m", "5m", "15m", "1h", "4h", "1d").
        :param days: Количество дней (по умолчанию 30).
        :return: DataFrame с историческими данными.
        """
        # Map timeframes to minutes
        timeframe_minutes = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
            "1d": 1440, "3d": 4320, "1w": 10080, "1M": 43200
        }

        minutes_per_day = 24 * 60
        timeframe_min = timeframe_minutes.get(timeframe, 60)
        candles_per_day = minutes_per_day // timeframe_min
        limit = days * candles_per_day

        return await self.get_market_data(symbol, timeframe, limit)

    # ── Paginated history + CSV cache ─────────────────────────────

    @staticmethod
    def _csv_cache_path(symbol: str, timeframe: str) -> str:
        """data/cache/BTC_USDT_15m.csv"""
        safe = symbol.replace("/", "_").replace(":", "_")
        os.makedirs("data/cache", exist_ok=True)
        return f"data/cache/{safe}_{timeframe}.csv"

    @staticmethod
    def _load_csv_cache(
        path: str,
    ) -> Optional[pd.DataFrame]:
        """
        Загружает CSV-кэш если файл свежее _CSV_CACHE_TTL.

        :return: DataFrame или None если кэш устарел/отсутствует.
        """
        if not os.path.exists(path):
            return None
        age = time.time() - os.path.getmtime(path)
        if age > _CSV_CACHE_TTL:
            return None
        try:
            df = pd.read_csv(path, index_col=None)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_numeric(
                    df["timestamp"], errors="coerce"
                )
            return df
        except Exception:
            return None

    @staticmethod
    def _save_csv_cache(df: pd.DataFrame, path: str) -> None:
        try:
            save_df = df.reset_index(drop=False)
            save_df.to_csv(path, index=False)
        except Exception:
            pass

    async def get_paginated_history(
        self,
        symbol: str,
        timeframe: str,
        months: int = 6,
    ) -> pd.DataFrame:
        """
        Загружает историю через пагинацию и кеширует в CSV.

        Алгоритм:
        1. Проверяем CSV-кэш (TTL 24ч).
        2. Если свежий — догружаем только новые свечи с
           последней сохранённой timestamp.
        3. Если устарел — скачиваем всё заново батчами
           по _FETCH_LIMIT свечей через параметр since.
        4. Сохраняем обновлённый CSV.
        5. Считаем индикаторы и возвращаем.

        :param symbol: Торговая пара ('BTC/USDT').
        :param timeframe: Таймфрейм ccxt ('15m', '1h', ...).
        :param months: Глубина истории в месяцах.
        :return: DataFrame с OHLCV + индикаторами.
        """
        cache_path = self._csv_cache_path(symbol, timeframe)
        cached = self._load_csv_cache(cache_path)

        now_ms = int(datetime.now().timestamp() * 1000)
        since_ms = int(
            (datetime.now() - timedelta(days=months * 30))
            .timestamp() * 1000
        )

        if cached is not None and not cached.empty:
            # Инкрементальная загрузка: только новые свечи
            if "timestamp" in cached.columns:
                last_ts = int(cached["timestamp"].max())
            elif cached.index.name == "timestamp":
                last_ts = int(cached.index.max())
            else:
                last_ts = 0
            self.logger.info(
                f"CSV cache hit for {symbol} {timeframe}. "
                f"Fetching candles since last timestamp."
            )
            new_df = await self._fetch_batches(
                symbol, timeframe,
                since_ms=last_ts + 1,
                until_ms=now_ms,
            )
            if not new_df.empty:
                combined = pd.concat(
                    [cached, new_df], ignore_index=True
                )
                combined = combined.drop_duplicates(
                    subset=["timestamp"]
                ).sort_values("timestamp").reset_index(
                    drop=True
                )
            else:
                combined = cached
        else:
            # Полная загрузка с нуля
            self.logger.info(
                f"Fetching {months}m history for "
                f"{symbol} {timeframe} (paginated)..."
            )
            combined = await self._fetch_batches(
                symbol, timeframe,
                since_ms=since_ms,
                until_ms=now_ms,
            )

        if combined.empty:
            self.logger.warning(
                f"No data fetched for {symbol}"
            )
            return combined

        self._save_csv_cache(combined, cache_path)
        self.logger.info(
            f"History: {len(combined)} candles "
            f"for {symbol} ({timeframe}), "
            f"saved to {cache_path}"
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

        Каждый батч — _FETCH_LIMIT свечей. Цикл продолжается
        пока не дойдём до until_ms или данные не кончатся.
        """
        all_frames = []
        current = since_ms

        if self.api.exchange is None:
            raise RuntimeError(
                "DataLoader not initialized. "
                "Call await loader.initialize() first."
            )
        while current < until_ms:
            try:
                raw = await self.api.exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=current,
                    limit=_FETCH_LIMIT,
                )
            except Exception as e:
                self.logger.error(
                    f"fetch_ohlcv error: {e}"
                )
                break

            if not raw:
                break

            df_batch = pd.DataFrame(
                raw,
                columns=[
                    "timestamp", "open", "high",
                    "low", "close", "volume",
                ],
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

        result = pd.concat(
            all_frames, ignore_index=True
        ).drop_duplicates(
            subset=["timestamp"]
        ).sort_values("timestamp").reset_index(drop=True)

        self.logger.info(
            f"Fetched {len(result)} candles total"
        )
        return result

    # ── Technical indicators ───────────────────────────────────────

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает полный набор технических индикаторов через indicators.py.

        Добавляет: RSI, MACD, Bollinger Bands, SMA 20/50, EMA 12/26, ATR,
        volatility, momentum, volume_sma, volume_ratio.
        Алиасы ema_short/ema_long добавляются для совместимости со стратегиями.

        :param df: DataFrame с колонками OHLCV.
        :return: DataFrame с добавленными индикаторами.
        :raises ValueError: Если отсутствует колонка 'close'.
        :raises Exception: В случае ошибки расчёта.
        """
        from src.indicators import add_indicators

        try:
            if "close" not in df.columns:
                raise ValueError(
                    f"Required column 'close' not found. "
                    f"Available columns: {df.columns.tolist()}"
                )

            df = add_indicators(df)

            # Алиасы для совместимости со стратегиями
            if "ema_12" in df.columns:
                df["ema_short"] = df["ema_12"]
            if "ema_26" in df.columns:
                df["ema_long"] = df["ema_26"]

            self.logger.info("Technical indicators calculated successfully")
            return df

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            self.logger.error(f"DataFrame columns: {df.columns.tolist()}")
            self.logger.error(f"DataFrame shape: {df.shape}")
            if not df.empty:
                self.logger.error(f"DataFrame sample:\n{df.head(2)}")
            raise

    async def close(self):
        """
        Закрывает соединения, включая подключение к API Bybit.
        """
        if self.api and self.api.exchange:
            await self.api.close()
