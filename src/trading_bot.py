import asyncio
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from config import Config
from src.bybit_api import BybitAPI
from src.data_loader import DataLoader
from src.portfolio_manager import PortfolioManager  # Обновленный, как предложено ранее
from src.redis_client import RedisClient
from src.risk_management import RiskManager
from src.strategies import TradingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TradingBot:
    def __init__(self):
        self.redis = RedisClient()
        self.api = BybitAPI()
        self.data_loader = DataLoader()
        self.portfolio_manager = None  # Инициализируем позже с режимом
        self.strategy = None
        self.risk_manager = RiskManager(Config.INITIAL_BALANCE, Config.RISK_PER_TRADE)
        self.is_running = False
        self.trading_mode = None  # "REAL" или "DEMO"

    async def initialize(self, trading_mode: str):
        """Initialize trading bot with selected mode"""
        self.trading_mode = trading_mode.upper()
        try:
            if self.trading_mode == "REAL":
                await self.api.initialize(Config.BYBIT_API_KEY, Config.BYBIT_API_SECRET)
            elif self.trading_mode == "DEMO":
                # Для демо используем демо-ключи (виртуальный баланс на основном endpoint)
                await self.api.initialize(Config.DEMO_BYBIT_API_KEY, Config.DEMO_BYBIT_API_SECRET)
            
            await self.data_loader.initialize(
                Config.BYBIT_API_KEY if self.trading_mode == "REAL" else Config.DEMO_BYBIT_API_KEY,
                Config.BYBIT_API_SECRET if self.trading_mode == "REAL" else Config.DEMO_BYBIT_API_SECRET
            )

            # Initialize portfolio manager with mode
            self.portfolio_manager = PortfolioManager(Config.INITIAL_BALANCE, self.trading_mode)

            # Initialize strategy
            self.strategy = TradingStrategy(Config.DEFAULT_STRATEGY)
            await self.strategy.initialize()

            # Restore state from Redis
            await self._restore_state()

            logger.info(f"Trading bot initialized in {self.trading_mode} mode successfully")

        except Exception as e:
            logger.error(f"Failed to initialize trading bot: {e}")
            raise

    async def _restore_state(self):
        """Restore state from Redis"""
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
        """Execute trade based on signal"""
        try:
            if not await self.risk_manager.validate_signal(signal, market_data):
                logger.info("Signal validation failed")
                return

            entry_price = signal.get("price", market_data["close"].iloc[-1])
            stop_loss = await self.risk_manager.calculate_stop_loss(entry_price, signal)
            
            # Получить текущий баланс и цену
            balance = await self.api.get_balance()
            if not balance:
                logger.error("Не удалось получить баланс аккаунта")
                return
            current_price = await self.api.get_current_price(Config.SYMBOL)
            if not current_price:
                logger.error("Не удалось получить текущую цену")
                return
            
            position_size = await self.risk_manager.calculate_position_size(
                self.portfolio_manager.current_balance,
                current_price,
                stop_loss,
            )

            if position_size <= 0:
                logger.warning("Invalid position size")
                return

            # Проверка баланса перед ордером
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

            # Execute order (теперь работает для обоих режимов через API, в DEMO с виртуальным балансом)
            order = await self.api.create_order(
                Config.SYMBOL, "limit", order_side, position_size, entry_price
            )
            if order:
                logger.info(f"Order executed in {self.trading_mode} mode: {order}")
            else:
                logger.error("Failed to create order")
                return

            # Update portfolio (синхронизируется с Redis для восстановления)
            success = await self.portfolio_manager.update_portfolio(
                Config.SYMBOL, order_side, position_size, entry_price
            )

            if success:
                logger.info("Portfolio updated successfully")
            else:
                logger.warning("Failed to update portfolio")

        except Exception as e:
            logger.error(f"Error executing trade: {e}")

    async def _update_performance_stats(self):
        """Update performance statistics"""
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
        """Analyze market and generate trading signal"""
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
        """Main trading loop"""
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
        """Stop trading bot"""
        self.is_running = False
        await self.api.close()
        await self.data_loader.close()
        logger.info("Trading bot stopped")


async def main():
    """Main function"""
    # Выбор режима — теперь через config.TRADING_MODE (из переменной окружения или по умолчанию "DEMO")
    mode = Config.TRADING_MODE.upper()
    if mode not in ["REAL", "DEMO"]:
        logger.error(f"Неверный TRADING_MODE в config: {mode}. Должен быть 'REAL' или 'DEMO'. Использую 'DEMO'.")
        mode = "DEMO"

    bot = TradingBot()

    try:
        await bot.initialize(mode)
        await bot.trading_loop()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
