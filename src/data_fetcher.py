import logging
from typing import Dict, Optional

from .data_loader import DataLoader  # используем существующий функционал
from config import config


class DataFetcher:
    """Адаптер для использования DataLoader в основном боте"""
    
    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader()
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize data fetcher"""
        await self.data_loader.initialize(
            config.BYBIT_API_KEY, 
            config.BYBIT_API_SECRET
        )

    async def fetch_market_data(self, symbol: str) -> Optional[Dict]:
        """Fetch market data in format expected by trading bot"""
        try:
            # Get raw data using existing DataLoader
            raw_data = await self.data_loader.get_market_data(symbol, "1h", 100)
            if raw_data is None or raw_data.empty:
                return None

            # Calculate indicators
            data_with_indicators = self.data_loader.calculate_technical_indicators(raw_data)
            
            # Get latest data point
            latest = data_with_indicators.iloc[-1]
            
            # Format for trading bot
            return {
                'symbol': symbol,
                'price': latest['close'],
                'open': latest['open'],
                'high': latest['high'],
                'low': latest['low'], 
                'volume': latest['volume'],
                'rsi': latest.get('rsi', 50),
                'macd': latest.get('macd', 0),
                'macd_signal': latest.get('macd_signal', 0),
                'ema_short': latest.get('ema_short', 0),
                'ema_long': latest.get('ema_long', 0),
                'timestamp': latest.name.isoformat() if hasattr(latest.name, 'isoformat') else str(latest.name)
            }

        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    async def close(self):
        """Close connections"""
        await self.data_loader.close()
