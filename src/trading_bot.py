import asyncio
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from config import Config
from src.bybit_api import BybitAPI
from src.data_loader import DataLoader
from src.portfolio_manager import PortfolioManager
from src.redis_client import RedisClient
from src.risk_management import RiskManager
from src.strategies import TradingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TradingBot:
    """
    Класс для управления торговым ботом криптовалют.
    
    Координирует работу с API Bybit, загрузчиком данных, менеджером портфеля, стратегиями торговли,
    управлением рисками и Redis для хранения состояния. Включает инициализацию, основной торговый цикл,
    выполнение ордеров, анализ рынка и обновление статистики производительности. Использует асинхронные
    методы для эффективной работы и логирование для отслеживания операций и ошибок.
    """
    
    def __init__(self):
        """
        Инициализирует торговый бот.
        
        Создает экземпляры клиентов Redis, API Bybit, загрузчика данных, менеджера портфеля,
        стратегии (изначально None) и менеджера рисков. Устанавливает флаг is_running в False.
        """
        self.redis = RedisClient()
        self.api = BybitAPI()
        self.data_loader = DataLoader()
        self.portfolio_manager = PortfolioManager(Config.INITIAL_BALANCE)
        self.strategy = None
        self.risk_manager = RiskManager(Config.INITIAL_BALANCE, Config.RISK_PER_TRADE)
        self.is_running = False

    async def initialize(self):
        """
        Инициализирует торговый бот.
        
        Выполняет инициализацию API Bybit и загрузчика данных с ключами из конфига,
        создает и инициализирует стратегию торговли, восстанавливает состояние из Redis.
        Логирует успех или ошибки. В случае исключения логирует ошибку и поднимает её.
        
        :raises Exception: Если инициализация не удалась.
        """
        try:
            await self.api.initialize(Config.BYBIT_API_KEY, Config.BYBIT_API_SECRET)
            await self.data_loader.initialize(
                Config.BYBIT_API_KEY, Config.BYBIT_API_SECRET
            )

            # Initialize strategy
            self.strategy = TradingStrategy(Config.DEFAULT_STRATEGY)
            await self.strategy.initialize()

            # Restore state from Redis
            await self._restore_state()

            logger.info("Trading bot initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize trading bot: {e}")
            raise

    async def _restore_state(self):
        """
        Восстанавливает состояние из Redis.
        
        Загружает состояние торговли и портфеля из Redis. Если состояние найдено, обновляет
        баланс и позиции портфеля. Логирует восстановление состояния.
        """
        state = self.redis.load_trading_state(Config.SYMBOL)
        if state:
            logger.info(f"Restored trading state from Redis: {state}")

        # Restore portfolio state
        portfolio_state = self.redis.load_trading_state("portfolio_state")
        if portfolio_state:
            self.portfolio_manager.current_balance = portfolio_state.get(
                "balance", Config.INITIAL_BALANCE
            )
            self.portfolio_manager.positions = portfolio_state.get("positions", {})

    async def _execute_trade(self, signal: dict, market_data: pd.DataFrame):
        """
        Выполняет торговый ордер на основе сигнала.
        
        Проверяет сигнал через менеджер рисков, рассчитывает размер позиции, стоп-лосс,
        проверяет баланс аккаунта перед ордером. Если проверки пройдены, создает ордер
        через API, обновляет портфель. Логирует все шаги, ошибки и предупреждения.
        В случае исключения логирует ошибку.
        
        :param signal: Словарь с торговым сигналом, содержащий "action", "price" и т.д. (dict).
        :param market_data: DataFrame с рыночными данными (pd.DataFrame).
        """
        try:
            if not await self.risk_manager.validate_signal(signal, market_data):
                logger.info("Signal validation failed")
                return

            entry_price = signal.get("price", market_data["close"].iloc[-1])
            stop_loss = await self.risk_manager.calculate_stop_loss(entry_price, signal)
            
            # Получить текущий баланс и цену для расчета position_size
            balance = await self.api.get_balance()
            if not balance:
                logger.error("Не удалось получить баланс аккаунта")
                return
            
            current_price = await self.api.get_current_price(Config.SYMBOL)
            if not current_price:
                logger.error("Не удалось получить текущую цену")
                return
            
            position_size = await self.risk_manager.calculate_position_size(
                self.portfolio_manager.current_balance,  # Используем баланс из portfolio_manager
                current_price,
                stop_loss,
            )

            if position_size <= 0:
                logger.warning("Invalid position size")
                return

            # Проверка баланса перед ордером (новое: предотвращает недостаточный баланс)
            order_side = "buy" if signal["action"] == "buy" else "sell"
            if order_side == "buy":
                cost = position_size * entry_price
                usdt_balance = balance.get('free', {}).get('USDT', 0)
                if cost > usdt_balance:
                    logger.warning(f"Недостаточный баланс USDT: {usdt_balance} < {cost}. Пропускаю ордер.")
                    return
            elif order_side == "sell":
                btc_balance = balance.get('free', {}).get('BTC', 0)
                if btc_balance < position_size:
                    logger.warning(f"Недостаточный баланс BTC: {btc_balance} < {position_size}. Пропускаю ордер.")
                    return

            # Execute order
            order = await self.api.create_order(
                Config.SYMBOL, "limit", order_side, position_size, entry_price
            )

            if order:
                logger.info(f"Order executed: {order}")
                # Update portfolio
                success = await self.portfolio_manager.update_portfolio(
                    Config.SYMBOL, order_side, position_size, entry_price
                )

                if success:
                    logger.info("Portfolio updated successfully")
                else:
                    logger.warning("Failed to update portfolio")
            else:
                logger.error("Failed to create order")

        except Exception as e:
            logger.error(f"Error executing trade: {e}")

    async def _update_performance_stats(self):
        """
        Обновляет статистику производительности.
        
        Получает текущую цену актива, рассчитывает стоимость портфеля, прибыль/убыток,
        собирает статистику (баланс, позиции и т.д.) и сохраняет в Redis. Логирует ошибки,
        если не удалось получить данные.
        
        :raises Exception: Если обновление статистики не удалось.
        """
        try:
            current_price = await self.api.get_current_price(Config.SYMBOL)
            if not current_price:
                logger.error("Не удалось получить текущую цену для статистики")
                return
            
            portfolio_value = await self.portfolio_manager.get_portfolio_value(
                {Config.SYMBOL: current_price}
            )

            stats = {
                "timestamp": datetime.now().isoformat(),
                "current_balance": self.portfolio_manager.current_balance,
                "portfolio_value": portfolio_value,
                "profit_loss": (portfolio_value - Config.INITIAL_BALANCE)
                / Config.INITIAL_BALANCE
                * 100,
                "positions": self.portfolio_manager.get_positions(),
            }
            self.redis.update_performance_stats(stats)

        except Exception as e:
            logger.error(f"Error updating performance stats: {e}")

    async def analyze_market(self, symbol: str, timeframe: str) -> Optional[dict]:
        """
        Анализирует рынок и генерирует торговый сигнал.
        
        Загружает рыночные данные, рассчитывает технические индикаторы,
        получает сигнал от стратегии. Возвращает сигнал или None в случае ошибки.
        Логирует сгенерированный сигнал или ошибки.
        
        :param symbol: Символ актива (str).
        :param timeframe: Таймфрейм данных (str).
        :return: Словарь с сигналом или None (Optional[dict]).
        """
        try:
            # Get market data
            data = await self.data_loader.get_market_data(symbol, timeframe, limit=100)

            # Calculate technical indicators
            data = self.data_loader.calculate_technical_indicators(data)

            # Get trading signal
            signal = await self.strategy.get_signal(data)

            logger.info(f"Generated signal for {symbol}: {signal}")
            return signal

        except Exception as e:
            logger.error(f"Error analyzing market for {symbol}: {e}")
            return None

    async def trading_loop(self):
        """
        Основной торговый цикл.
        
        Запускает бесконечный цикл, анализирует рынок, выполняет ордера при сигналах,
        обновляет статистику и ждет интервала. Останавливается при флаге is_running = False.
        Логирует ошибки и ждет 30 секунд при исключениях.
        """
        self.is_running = True
        logger.info("Starting trading loop")

        while self.is_running:
            try:
                # Analyze market and get signal
                signal = await self.analyze_market(Config.SYMBOL, Config.TIMEFRAME)

                if signal and signal["action"] != "hold":
                    # Get fresh market data for execution
                    market_data = await self.data_loader.get_market_data(
                        Config.SYMBOL, Config.TIMEFRAME, limit=100
                    )
                    market_data = self.data_loader.calculate_technical_indicators(
                        market_data
                    )

                    await self._execute_trade(signal, market_data)

                # Update performance statistics
                await self._update_performance_stats()

                # Wait for next iteration
                await asyncio.sleep(Config.TRADING_INTERVAL)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(30)

    async def stop(self):
        """
        Останавливает торговый бот.
        
        Устанавливает флаг is_running в False, закрывает соединения с API и загрузчиком данных.
        Логирует остановку.
        """
        self.is_running = False
        await self.api.close()
        await self.data_loader.close()
        logger.info("Trading bot stopped")


async def main():
    """
    Главная функция для запуска торгового бота.
    
    Создает экземпляр TradingBot, инициализирует его, запускает торговый цикл.
    Обрабатывает прерывания клавиатуры и фатальные ошибки, в конце останавливает бота.
    """
    bot = TradingBot()

    try:
        await bot.initialize()
        await bot.trading_loop()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
