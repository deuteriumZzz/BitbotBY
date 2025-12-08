import asyncio
import logging
from datetime import datetime

import pandas as pd

from config import Config
from src.bybit_api import BybitAPI
from src.data_loader import DataLoader
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
        self.strategy = None
        self.risk_manager = RiskManager(Config.INITIAL_BALANCE, Config.RISK_PER_TRADE)
        self.current_balance = Config.INITIAL_BALANCE
        self.is_running = False

    async def initialize(self):
        """Initialize trading bot"""
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
        """Restore state from Redis"""
        state = self.redis.load_trading_state(Config.SYMBOL)
        if state:
            logger.info(f"Restored trading state from Redis: {state}")

        # Restore balance from Redis or use initial balance
        balance_state = self.redis.load_trading_state("balance")
        if balance_state:
            self.current_balance = balance_state.get(
                "current_balance", Config.INITIAL_BALANCE
            )

    async def _execute_trade(self, signal: dict, market_data: pd.DataFrame):
        """Execute trade based on signal"""
        try:
            if not await self.risk_manager.validate_signal(signal, market_data):
                logger.info("Signal validation failed")
                return

            entry_price = signal.get("price", market_data["close"].iloc[-1])
            stop_loss = await self.risk_manager.calculate_stop_loss(entry_price, signal)
            position_size = await self.risk_manager.calculate_position_size(
                self.current_balance, entry_price, stop_loss
            )

            if position_size <= 0:
                logger.warning("Invalid position size")
                return

            # Execute order
            order_side = "buy" if signal["action"] == "buy" else "sell"
            order = await self.api.create_order(
                Config.SYMBOL, "limit", order_side, position_size, entry_price
            )

            if order:
                logger.info(f"Order executed: {order}")
                # Update balance
                if signal["action"] == "buy":
                    self.current_balance -= position_size * entry_price
                else:
                    self.current_balance += position_size * entry_price

                # Save updated balance to Redis
                self.redis.save_trading_state(
                    "balance",
                    {
                        "current_balance": self.current_balance,
                        "last_trade": order,
                        "timestamp": datetime.now().isoformat(),
                    },
                )

        except Exception as e:
            logger.error(f"Error executing trade: {e}")

    async def _update_performance_stats(self):
        """Update performance statistics"""
        try:
            stats = {
                "timestamp": datetime.now().isoformat(),
                "current_balance": self.current_balance,
                "profit_loss": (self.current_balance - Config.INITIAL_BALANCE)
                / Config.INITIAL_BALANCE
                * 100,
                "total_trades": 0,  # Would be updated with actual trade count
                "win_rate": 0.0,
            }
            self.redis.update_performance_stats(stats)

        except Exception as e:
            logger.error(f"Error updating performance stats: {e}")

    async def trading_loop(self):
        """Main trading loop"""
        self.is_running = True
        logger.info("Starting trading loop")

        while self.is_running:
            try:
                # Get market data
                data = await self.data_loader.get_market_data(
                    Config.SYMBOL, Config.TIMEFRAME, limit=100
                )

                # Calculate technical indicators
                data = self.data_loader.calculate_technical_indicators(data)

                # Get trading signal
                signal = await self.strategy.get_signal(data)

                logger.info(f"Generated signal: {signal}")

                # Execute trade if not hold
                if signal["action"] != "hold":
                    await self._execute_trade(signal, data)

                # Update performance statistics
                await self._update_performance_stats()

                # Wait for next iteration
                await asyncio.sleep(60)

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
