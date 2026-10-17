"""Базовый класс стратегии: отдельно, чтобы реестр и порты не зависели друг от друга."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd


class BaseStrategy(ABC):
    """
    Базовый абстрактный класс для всех торговых стратегий.

    Определяет интерфейс generate_signal() и вспомогательные методы
    для безопасного доступа к значениям DataFrame.
    """

    name: str = ""
    description: str = ""
    recommended_timeframes: List[str] = []
    risk_level: str = "medium"  # low / medium / high
    market_type: str = "any"  # trending / ranging / volatile / any
    # Где стратегия вообще применима: spot (только лонг), futures (нужны шорты
    # или перпетуалы) или both. На споте шорта нет — futures-стратегии там
    # не участвуют ни в отборе, ни в промпте AI.
    markets: str = "both"
    # Экспериментальная: добавлена без доказанного преимущества и по умолчанию
    # выключена; включается вручную (см. RuntimeConfig.ensure_experimental_disabled).
    experimental: bool = False

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Генерирует торговый сигнал на основе рыночных данных.

        :param data: DataFrame с OHLCV и техническими индикаторами.
        :return: Словарь с ключами action, confidence и опционально price.
        """

    def get_info(self) -> Dict[str, Any]:
        """
        Возвращает метаданные стратегии.

        :return: Словарь с name, description, recommended_timeframes,
            risk_level, market_type.
        """
        return {
            "name": self.name,
            "description": self.description,
            "recommended_timeframes": self.recommended_timeframes,
            "risk_level": self.risk_level,
            "market_type": self.market_type,
            "markets": self.markets,
            "experimental": self.experimental,
        }

    def _last(self, data: pd.DataFrame, col: str, default: float = 0.0) -> float:
        """Возвращает последнее значение колонки или default."""
        if col in data.columns and len(data) > 0:
            val = data[col].iloc[-1]
            return float(val) if pd.notna(val) else default
        return default

    def _prev(self, data: pd.DataFrame, col: str, default: float = 0.0) -> float:
        """Возвращает предпоследнее значение колонки или default."""
        if col in data.columns and len(data) > 1:
            val = data[col].iloc[-2]
            return float(val) if pd.notna(val) else default
        return default

    def _prev2(self, data: pd.DataFrame, col: str, default: float = 0.0) -> float:
        """Возвращает третье с конца значение колонки или default."""
        if col in data.columns and len(data) > 2:
            val = data[col].iloc[-3]
            return float(val) if pd.notna(val) else default
        return default
