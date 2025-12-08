import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from data_fetcher import DataFetcher
from trading_engine import TradingEngine


class TestAPI:
    """Тесты для API взаимодействия с биржами"""

    @pytest.fixture
    def data_fetcher(self):
        """Фикстура для создания экземпляра DataFetcher"""
        config = {
            "EXCHANGE": "binance",
            "API_KEY": "test_key",
            "API_SECRET": "test_secret",
            "REQUEST_TIMEOUT": 10,
        }
        return DataFetcher(config)

    @pytest.fixture
    def trading_engine(self):
        """Фикстура для создания экземпляра TradingEngine"""
        config = {
            "TRADING_MODE": "paper",
            "EXCHANGE": "binance",
            "API_KEY": "test_key",
            "API_SECRET": "test_secret",
        }
        return TradingEngine(config)

    @pytest.mark.asyncio
    async def test_fetch_market_data_success(self, data_fetcher):
        """Тест успешного получения рыночных данных"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            # Мокируем успешный ответ
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "symbol": "BTC/USDT",
                "price": 50000.0,
                "volume": 1000.0,
            }
            mock_get.return_value.__aenter__.return_value = mock_response

            data = await data_fetcher.fetch_market_data("BTC/USDT")

            assert data["symbol"] == "BTC/USDT"
            assert data["price"] == 50000.0
            mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_market_data_failure(self, data_fetcher):
        """Тест обработки ошибки при получении данных"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_get.return_value.__aenter__.return_value = mock_response

            with pytest.raises(Exception) as exc_info:
                await data_fetcher.fetch_market_data("BTC/USDT")

            assert "Failed to fetch data" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_trade_paper_mode(self, trading_engine):
        """Тест исполнения трейда в paper режиме"""
        trade_decision = {
            "action": "BUY",
            "symbol": "BTC/USDT",
            "amount": 0.1,
            "price": 50000.0,
        }

        result = await trading_engine.execute_trade(trade_decision)

        assert result["status"] == "executed"
        assert result["mode"] == "paper"
        assert result["symbol"] == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_execute_trade_live_mode(self):
        """Тест исполнения трейда в live режиме"""
        config = {
            "TRADING_MODE": "live",
            "EXCHANGE": "binance",
            "API_KEY": "test_key",
            "API_SECRET": "test_secret",
        }

        trading_engine = TradingEngine(config)
        trade_decision = {
            "action": "BUY",
            "symbol": "BTC/USDT",
            "amount": 0.1,
            "price": 50000.0,
        }

        with patch("ccxt.binance.create_order") as mock_order:
            mock_order.return_value = {"id": "123", "status": "closed"}

            result = await trading_engine.execute_trade(trade_decision)

            assert result["status"] == "closed"
            mock_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_limiting(self, data_fetcher):
        """Тест ограничения частоты запросов"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"price": 50000.0}
            mock_get.return_value.__aenter__.return_value = mock_response

            # Выполняем несколько запросов подряд
            start_time = asyncio.get_event_loop().time()
            for _ in range(5):
                await data_fetcher.fetch_market_data("BTC/USDT")
            end_time = asyncio.get_event_loop().time()

            # Проверяем что между запросами была задержка
            assert end_time - start_time >= 0.2  # 200ms rate limit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
