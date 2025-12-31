import logging
from typing import Dict, Optional

from config import config

from .data_loader import DataLoader  # используем существующий функционал


class DataFetcher:
    """
    Адаптер для использования DataLoader в основном торговом боте.
    Обеспечивает получение рыночных данных в формате, ожидаемом ботом,
    включая расчет технических индикаторов.
    """

    def __init__(self, config):
        """
        Инициализирует объект DataFetcher с конфигурацией и экземпляром DataLoader.

        :param config: Объект конфигурации с настройками API и т.д.
        """
        self.config = config
        self.data_loader = DataLoader()
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """
        Инициализирует DataFetcher, включая подключение DataLoader к API Bybit.
        Использует ключи из конфигурации.
        """
        await self.data_loader.initialize(config.BYBIT_API_KEY, config.BYBIT_API_SECRET)

    async def fetch_market_data(self, symbol: str) -> Optional[Dict]:
        """
        Получает рыночные данные в формате, ожидаемом торговым ботом.
        Включает расчет технических индикаторов и извлечение последнего значения.

        :param symbol: Символ торговой пары (например, "BTC/USDT").
        :return: Словарь с рыночными данными и индикаторами или None в случае ошибки.
        """
        try:
            # Get raw data using existing DataLoader
            raw_data = await self.data_loader.get_market_data(symbol, "1h", 100)
            if raw_data is None or raw_data.empty:
                return None

            # Calculate indicators
            data_with_indicators = self.data_loader.calculate_technical_indicators(
                raw_data
            )

            # Get latest data point
            latest = data_with_indicators.iloc[-1]

            # Format for trading bot
            return {
                "symbol": symbol,
                "price": latest["close"],
                "open": latest["open"],
                "high": latest["high"],
                "low": latest["low"],
                "volume": latest["volume"],
                "rsi": latest.get("rsi", 50),
                "macd": latest.get("macd", 0),
                "macd_signal": latest.get("macd_signal", 0),
                "ema_short": latest.get("ema_short", 0),
                "ema_long": latest.get("ema_long", 0),
                "timestamp": latest.name.isoformat()
                if hasattr(latest.name, "isoformat")
                else str(latest.name),
            }

        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    async def close(self):
        """
        Закрывает соединения, включая подключение DataLoader.
        """
        await self.data_loader.close()
