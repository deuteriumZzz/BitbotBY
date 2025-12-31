import logging

import pandas as pd

from .bybit_api import BybitAPI
from .redis_client import RedisClient


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
        Использует метод get_market_data с расчетом лимита на основе дней.
        
        :param symbol: Символ торговой пары.
        :param timeframe: Таймфрейм.
        :param days: Количество дней (по умолчанию 30).
        :return: DataFrame с историческими данными.
        """
        limit = days * 24  # Approximate number of candles
        return await self.get_market_data(symbol, timeframe, limit)

    def calculate_technical_indicators(self, df: pd.DataFrame):
        """
        Рассчитывает технические индикаторы (EMA, RSI, MACD) на основе данных OHLCV.
        Проверяет наличие необходимых колонок и логирует ошибки.
        
        :param df: DataFrame с рыночными данными.
        :return: DataFrame с добавленными индикаторами.
        :raises ValueError: Если отсутствуют необходимые колонки.
        :raises Exception: В случае ошибки расчета.
        """
        try:
            # Проверяем наличие необходимых колонок
            required_columns = ["close"]
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(
                        f"Required column '{col}' not found in DataFrame. "
                        f"Available columns: {df.columns.tolist()}"
                    )

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
            df["signal"] = df["macd"].ewm(span=9).mean()  # Исправлено: macd_signal → signal

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
        await self.api.close()
