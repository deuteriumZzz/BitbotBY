import logging
from datetime import datetime
from typing import Dict

import numpy as np

from config import Config
from src.redis_client import RedisClient


class PortfolioManager:
    """
    Класс для управления портфелем в торговле криптовалютой.

    Отвечает за отслеживание баланса, позиций, обновление портфеля на основе торговых операций,
    расчет стоимости портфеля и ребалансировку. Использует Redis для сохранения состояния.
    """

    def __init__(self, initial_balance: float):
        """
        Инициализирует экземпляр PortfolioManager.

        :param initial_balance: Начальный баланс портфеля (float).
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions: Dict[str, float] = {}
        self.commission_rate: float = Config.COMMISSION_RATE
        self.total_commissions: float = 0.0
        self.redis = RedisClient()
        self.logger = logging.getLogger(__name__)

    async def update_portfolio(
        self, symbol: str, action: str, quantity: float, price: float
    ) -> bool:
        """
        Обновляет портфель на основе торговой операции.

        Проверяет возможность операции (покупка или продажа), обновляет баланс и позиции,
        сохраняет состояние в Redis. Для покупки проверяет достаточность баланса,
        для продажи - наличие достаточного количества актива.

        :param symbol: Символ актива (str, например, "BTC").
        :param action: Действие ("buy" или "sell").
        :param quantity: Количество актива (float).
        :param price: Цена актива (float).
        :return: True, если операция успешна, иначе False.
        :raises Exception: В случае ошибок при обновлении (логируется в logger).
        """
        try:
            if action == "buy":
                commission = (
                    quantity * price * self.commission_rate
                )
                cost = quantity * price + commission
                if cost <= self.current_balance:
                    self.current_balance -= cost
                    self.total_commissions += commission
                    self.positions[symbol] = (
                        self.positions.get(symbol, 0) + quantity
                    )
                    await self._save_portfolio_state()
                    return True

            elif action == "sell":
                if (
                    symbol in self.positions
                    and self.positions[symbol] >= quantity
                ):
                    commission = (
                        quantity * price * self.commission_rate
                    )
                    revenue = quantity * price - commission
                    self.current_balance += revenue
                    self.total_commissions += commission
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
        """
        Рассчитывает общую стоимость портфеля.

        Суммирует текущий баланс и стоимость всех позиций на основе текущих цен.
        Если цена для символа не указана, позиция игнорируется.

        :param current_prices: Словарь текущих цен по символам (Dict[str, float]).
        :return: Общая стоимость портфеля (float).
        """
        total_value = self.current_balance

        for symbol, quantity in self.positions.items():
            if symbol in current_prices:
                total_value += quantity * current_prices[symbol]

        return total_value

    async def _save_portfolio_state(self):
        """
        Сохраняет состояние портфеля в Redis.

        Создает словарь с текущим timestamp, балансом, позициями и общей стоимостью
        (без текущих цен для позиций) и сохраняет его под ключом "portfolio_state".

        :raises Exception: В случае ошибок при сохранении (не обрабатывается явно).
        """
        portfolio_state = {
            "timestamp": datetime.now().isoformat(),
            "balance": self.current_balance,
            "positions": self.positions,
            "total_commissions": self.total_commissions,
        }
        self.redis.save_trading_state(
            "portfolio_state", portfolio_state
        )

    def get_positions(self) -> Dict[str, float]:
        """
        Возвращает копию текущих позиций.

        :return: Словарь с символами и их количествами (Dict[str, float]).
        """
        return self.positions.copy()

    async def get_position_size(self, symbol: str) -> float:
        """
        Возвращает текущий размер позиции для заданного символа.

        :param symbol: Символ актива (str).
        :return: Количество актива в позиции (float, 0.0 если позиция отсутствует).
        """
        return self.positions.get(symbol, 0.0)

    async def rebalance_portfolio(
        self, target_allocations: Dict[str, float], current_prices: Dict[str, float]
    ):
        """
        Реbalancing портфеля к целевым аллокациям.

        Рассчитывает текущую стоимость портфеля, определяет необходимые покупки или продажи
        для достижения целевых аллокаций и выполняет соответствующие операции.
        Аллокации должны суммироваться к 1.0 (100%).

        :param target_allocations: Словарь целевых аллокаций по символам (Dict[str, float], суммы должны быть 1.0).
        :param current_prices: Словарь текущих цен по символам (Dict[str, float]).
        :raises Exception: В случае ошибок при обновлении портфеля (логируется в update_portfolio).
        """
        total_value = await self.get_portfolio_value(current_prices)

        for symbol, target_allocation in target_allocations.items():
            price = current_prices.get(symbol)
            if price is None or price <= 0:
                continue
            target_value = total_value * target_allocation
            current_value = (
                await self.get_position_size(symbol) * price
            )

            if current_value < target_value:
                # Need to buy
                buy_value = target_value - current_value
                quantity = buy_value / price
                await self.update_portfolio(
                    symbol, "buy", quantity, price
                )

            elif current_value > target_value:
                # Need to sell
                sell_value = current_value - target_value
                quantity = sell_value / price
                await self.update_portfolio(
                    symbol, "sell", quantity, price
                )
