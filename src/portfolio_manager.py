import logging
from typing import Dict

import numpy as np

from src.redis_client import RedisClient


class PortfolioManager:
    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions: Dict[str, float] = {}
        self.redis = RedisClient()
        self.logger = logging.getLogger(__name__)

    async def update_portfolio(
        self, symbol: str, action: str, quantity: float, price: float
    ) -> bool:
        """Update portfolio based on trading action"""
        try:
            if action == "buy":
                cost = quantity * price
                if cost <= self.current_balance:
                    self.current_balance -= cost
                    self.positions[symbol] = self.positions.get(symbol, 0) + quantity
                    await self._save_portfolio_state()
                    return True

            elif action == "sell":
                if symbol in self.positions and self.positions[symbol] >= quantity:
                    revenue = quantity * price
                    self.current_balance += revenue
                    self.positions[symbol] -= quantity

                    if self.positions[symbol] <= 0:
                        del self.positions[symbol]

                    await self._save_portfolio_state()
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Error updating portfolio: {e}")
            return False

    async def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        total_value = self.current_balance

        for symbol, quantity in self.positions.items():
            if symbol in current_prices:
                total_value += quantity * current_prices[symbol]

        return total_value

    async def _save_portfolio_state(self):
        """Save portfolio state to Redis"""
        portfolio_state = {
            "timestamp": np.datetime64("now").astype(str),
            "balance": self.current_balance,
            "positions": self.positions,
            "total_value": await self.get_portfolio_value({}),
        }

        self.redis.save_trading_state("portfolio_state", portfolio_state)

    async def get_position_size(self, symbol: str) -> float:
        """Get current position size for symbol"""
        return self.positions.get(symbol, 0.0)

    async def rebalance_portfolio(
        self, target_allocations: Dict[str, float], current_prices: Dict[str, float]
    ):
        """Rebalance portfolio to target allocations"""
        total_value = await self.get_portfolio_value(current_prices)

        for symbol, target_allocation in target_allocations.items():
            target_value = total_value * target_allocation
            current_value = await self.get_position_size(symbol) * current_prices.get(
                symbol, 0
            )

            if current_value < target_value:
                # Need to buy
                buy_value = target_value - current_value
                quantity = buy_value / current_prices[symbol]
                await self.update_portfolio(
                    symbol, "buy", quantity, current_prices[symbol]
                )

            elif current_value > target_value:
                # Need to sell
                sell_value = current_value - target_value
                quantity = sell_value / current_prices[symbol]
                await self.update_portfolio(
                    symbol, "sell", quantity, current_prices[symbol]
                )
