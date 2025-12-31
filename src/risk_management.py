import logging
from typing import Any, Dict

import numpy as np

from src.redis_client import RedisClient


class RiskManager:
    """
    Класс для управления рисками в торговле.
    
    Предоставляет методы для расчета размера позиции, стоп-лосса, тейк-профита и валидации сигналов
    на основе заданных параметров риска. Использует Redis для сохранения расчетов рисков и логирование для отслеживания операций.
    """
    
    def __init__(self, initial_balance: float, risk_per_trade: float = 0.02):
        """
        Инициализирует менеджер рисков.
        
        Устанавливает начальный баланс, процент риска на сделку, клиент Redis и логгер.
        
        :param initial_balance: Начальный баланс счета (float).
        :param risk_per_trade: Процент риска на одну сделку (float, по умолчанию 0.02 - 2%).
        """
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.redis = RedisClient()
        self.logger = logging.getLogger(__name__)

    async def calculate_position_size(
        self, current_balance: float, entry_price: float, stop_loss: float
    ) -> float:
        """
        Рассчитывает размер позиции на основе правил управления рисками.
        
        Определяет сумму риска, вычисляет размер позиции как риск деленный на разницу цен,
        сохраняет данные расчета в Redis под ключом "risk_calculation".
        Если разница цен равна нулю, возвращает 0.
        
        :param current_balance: Текущий баланс счета (float).
        :param entry_price: Цена входа в позицию (float).
        :param stop_loss: Цена стоп-лосса (float).
        :return: Рассчитанный размер позиции (float).
        """
        risk_amount = current_balance * self.risk_per_trade
        price_difference = abs(entry_price - stop_loss)

        if price_difference == 0:
            return 0

        position_size = risk_amount / price_difference

        # Save risk calculation to Redis
        risk_data = {
            "timestamp": np.datetime64("now").astype(str),
            "current_balance": current_balance,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "position_size": position_size,
            "risk_amount": risk_amount,
        }
        self.redis.save_trading_state("risk_calculation", risk_data)

        return position_size

    async def calculate_stop_loss(
        self, entry_price: float, signal: Dict[str, Any]
    ) -> float:
        """
        Рассчитывает цену стоп-лосса.
        
        Для сигнала "buy" устанавливает стоп-лосс на 5% ниже цены входа,
        для "sell" - на 5% выше. Для других сигналов возвращает цену входа.
        
        :param entry_price: Цена входа в позицию (float).
        :param signal: Словарь с данными сигнала, содержащий ключ "action" (Dict[str, Any]).
        :return: Рассчитанная цена стоп-лосса (float).
        """
        if signal["action"] == "buy":
            stop_loss = entry_price * 0.95  # 5% stop loss
        elif signal["action"] == "sell":
            stop_loss = entry_price * 1.05  # 5% stop loss
        else:
            stop_loss = entry_price

        return stop_loss

    async def calculate_take_profit(
        self, entry_price: float, signal: Dict[str, Any]
    ) -> float:
        """
        Рассчитывает цену тейк-профита.
        
        Использует соотношение риска к прибыли 1:2. Рассчитывает риск на основе стоп-лосса,
        затем определяет тейк-профит как цену входа плюс/минус удвоенный риск в зависимости от действия сигнала.
        Для других сигналов возвращает цену входа.
        
        :param entry_price: Цена входа в позицию (float).
        :param signal: Словарь с данными сигнала, содержащий ключ "action" (Dict[str, Any]).
        :return: Рассчитанная цена тейк-профита (float).
        """
        risk_reward_ratio = 2.0  # 1:2 risk-reward ratio

        if signal["action"] == "buy":
            stop_loss = await self.calculate_stop_loss(entry_price, signal)
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * risk_reward_ratio)
        elif signal["action"] == "sell":
            stop_loss = await self.calculate_stop_loss(entry_price, signal)
            risk = stop_loss - entry_price
            take_profit = entry_price - (risk * risk_reward_ratio)
        else:
            take_profit = entry_price

        return take_profit

    async def validate_signal(self, signal: Dict[str, Any], market_data: Any) -> bool:
        """
        Валидирует торговый сигнал на основе правил риска.
        
        Проверяет, что сигнал не "hold", уверенность сигнала не ниже 0.6.
        Логирует предупреждение при низкой уверенности. Возвращает False, если проверки не пройдены.
        Дополнительные проверки риска могут быть добавлены.
        
        :param signal: Словарь с данными сигнала, содержащий ключи "action" и опционально "confidence" (Dict[str, Any]).
        :param market_data: Рыночные данные (любой тип, не используется в текущей реализации).
        :return: True, если сигнал валиден, иначе False.
        """
        if signal["action"] == "hold":
            return False

        # Check if confidence is sufficient
        if signal.get("confidence", 0) < 0.6:
            self.logger.warning("Signal confidence too low")
            return False

        # Additional risk checks can be added here
        return True
