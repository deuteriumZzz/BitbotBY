import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from cryptotradingbot import CryptoTradingBot
from portfolio_manager import PortfolioManager
from risk_manager import RiskManager


class TestIntegration:
    """Интеграционные тесты взаимодействия компонентов"""
    
    @pytest.fixture
    def bot_config(self):
        """Конфигурация для тестового бота"""
        return {
            'TRADING_MODE': 'paper',
            'INITIAL_BALANCE': 10000.0,
            'TRADING_SYMBOLS': ['BTC/USDT'],
            'TRADING_INTERVAL': 1,
            'RISK_PER_TRADE': 0.02,
            'MAX_POSITION_SIZE': 0.1
        }
    
    @pytest.fixture
    async def trading_bot(self, bot_config):
        """Создание тестового бота"""
        bot = CryptoTradingBot(bot_config)
        await bot.initialize()
        yield bot
        await bot.shutdown()
    
    @pytest.mark.asyncio
    async def test_full_trading_cycle(self, trading_bot):
        """Тест полного цикла торговли"""
        # Мокируем внешние зависимости
        with patch.object(trading_bot.data_fetcher, 'fetch_market_data') as mock_fetch, \
             patch.object(trading_bot.trading_engine, 'execute_trades') as mock_execute:
            
            # Мок рыночных данных
            mock_fetch.return_value = {
                'symbol': 'BTC/USDT',
                'price': 50000.0,
                'volume': 1000.0,
                'timestamp': 1234567890
            }
            
            # Мок успешного исполнения
            mock_execute.return_value = [{
                'status': 'executed',
                'symbol': 'BTC/USDT',
                'amount': 0.1,
                'price': 50000.0
            }]
            
            # Выполняем торговую итерацию
            result = await trading_bot._trading_iteration()
            
            # Проверяем что все компоненты были вызваны
            assert mock_fetch.called
            assert mock_execute.called
            assert result['decisions_made'] == 1
    
    @pytest.mark.asyncio
    async def test_risk_manager_integration(self, trading_bot):
        """Тест интеграции RiskManager с TradingEngine"""
        # Создаем рискованную торговую decision
        risky_decision = {
            'action': 'BUY',
            'symbol': 'BTC/USDT',
            'amount': 2.0,  # Слишком много для портфеля
            'price': 50000.0
        }
        
        # Проверяем что RiskManager заблокирует эту сделку
        is_allowed = await trading_bot.risk_manager.validate_trade(
            risky_decision, 
            trading_bot.portfolio_manager.get_portfolio_state()
        )
        
        assert not is_allowed
        assert 'exceeds max position size' in risky_decision.get('rejection_reason', '')
    
    @pytest.mark.asyncio
    async def test_portfolio_updates(self, trading_bot):
        """Тест обновления портфеля после сделки"""
        initial_balance = trading_bot.portfolio_manager.get_balance()
        
        # Симулируем успешную сделку
        trade_result = {
            'status': 'executed',
            'symbol': 'BTC/USDT',
            'action': 'BUY',
            'amount': 0.1,
            'price': 50000.0,
            'cost': 5000.0,
            'fee': 25.0
        }
        
        await trading_bot.portfolio_manager.update_portfolio(trade_result)
        
        updated_balance = trading_bot.portfolio_manager.get_balance()
        positions = trading_bot.portfolio_manager.get_positions()
        
        assert updated_balance == initial_balance - 5000.0 - 25.0
        assert 'BTC/USDT' in positions
        assert positions['BTC/USDT']['amount'] == 0.1
    
    @pytest.mark.asyncio
    async def test_rl_agent_integration(self, trading_bot):
        """Тест интеграции RL агента с торговой системой"""
        market_data = {
            'symbol': 'BTC/USDT',
            'price': 50000.0,
            'volume': 1000.0,
            'indicators': {'rsi': 60, 'macd': 0.01}
        }
        
        portfolio_state = trading_bot.portfolio_manager.get_portfolio_state()
        
        # Получаем решение от RL агента
        decision = await trading_bot.rl_agent.choose_action(
            market_data, 
            portfolio_state
        )
        
        # Проверяем что решение имеет правильный формат
        assert 'action' in decision
        assert decision['action'] in ['BUY', 'SELL', 'HOLD', 'CLOSE']
        assert 'symbol' in decision
        assert 'confidence' in decision
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, trading_bot):
        """Тест обработки ошибок во всей системе"""
        with patch.object(trading_bot.data_fetcher, 'fetch_market_data') as mock_fetch:
            # Мокируем ошибку получения данных
            mock_fetch.side_effect = Exception("API недоступен")
            
            # Выполняем итерацию и проверяем что ошибка обрабатывается
            result = await trading_bot._trading_iteration()
            
            assert result['success'] == False
            assert 'error' in result
            # Проверяем что портфель не изменился
            assert trading_bot.portfolio_manager.get_portfolio_state()['balance'] == 10000.0

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
