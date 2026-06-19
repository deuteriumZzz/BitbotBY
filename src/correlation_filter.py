"""
Корреляционный фильтр позиций.

Перед открытием новой позиции проверяет корреляцию её log-returns
с уже открытыми позициями. Высококоррелированные сигналы отклоняются —
это предотвращает удвоение экспозиции на один и тот же рыночный фактор
(например, BTC и ETH в лонге одновременно).
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CorrelationFilter:
    """
    Отслеживает корреляцию log-returns между монетами по скользящему окну.

    Использование:
      1. Вызывать update_from_df() после каждого рыночного скана.
      2. Перед открытием позиции вызывать is_allowed() или max_correlation().
    """

    def __init__(self, window: int = 50, max_corr: float = 0.7) -> None:
        """
        :param window: Количество свечей для расчёта корреляции.
        :param max_corr: Порог: если |corr| ≥ max_corr — открытие запрещено.
        """
        self._window = window
        self._max_corr = max_corr
        # +1 потому что из N цен получаем N-1 доходностей
        self._prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window + 1))

    def update_from_df(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Обновляет историю цен из OHLCV DataFrame.

        :param symbol: Символ ccxt ('BTC/USDT').
        :param df: DataFrame с колонкой 'close'.
        """
        if df is None or df.empty or "close" not in df.columns:
            return
        closes = df["close"].astype(float).tail(self._window + 1).tolist()
        q = self._prices[symbol]
        for price in closes:
            q.append(price)

    def _returns(self, symbol: str) -> Optional[np.ndarray]:
        """Log-returns для символа. None если данных меньше 10 свечей."""
        prices = list(self._prices[symbol])
        if len(prices) < 10:
            return None
        arr = np.array(prices, dtype=float)
        # Малая epsilon защищает от log(0) при нулевых ценах
        return np.diff(np.log(arr + 1e-10))

    def correlation(self, sym_a: str, sym_b: str) -> Optional[float]:
        """
        Pearson-корреляция log-returns двух монет.

        :return: Значение [-1, 1] или None если данных недостаточно.
        """
        r_a = self._returns(sym_a)
        r_b = self._returns(sym_b)
        if r_a is None or r_b is None:
            return None
        n = min(len(r_a), len(r_b))
        if n < 10:
            return None
        try:
            corr = float(np.corrcoef(r_a[-n:], r_b[-n:])[0, 1])
            return None if np.isnan(corr) else corr
        except Exception:
            return None

    def max_correlation(self, new_symbol: str, open_symbols: List[str]) -> float:
        """
        Максимальная |correlation| new_symbol с любой из открытых позиций.

        :return: 0.0 если нет данных или открытых позиций.
        """
        if not open_symbols:
            return 0.0
        corrs = []
        for sym in open_symbols:
            if sym == new_symbol:
                continue
            c = self.correlation(new_symbol, sym)
            if c is not None:
                corrs.append(abs(c))
        return max(corrs) if corrs else 0.0

    def is_allowed(self, new_symbol: str, open_symbols: List[str]) -> bool:
        """
        Проверяет, можно ли открыть позицию по new_symbol.

        Запрещает открытие если max |correlation| с любой открытой
        позицией превышает порог max_corr.

        :return: True — открыть можно, False — слишком высокая корреляция.
        """
        corr = self.max_correlation(new_symbol, open_symbols)
        if corr >= self._max_corr:
            logger.info(
                "CorrelationFilter: %s заблокирован (max |corr|=%.2f >= %.2f)",
                new_symbol,
                corr,
                self._max_corr,
            )
            return False
        return True

    def to_dict(self) -> dict:
        """Сериализует историю цен для сохранения в Redis."""
        return {sym: list(q) for sym, q in self._prices.items()}

    def from_dict(self, data: dict) -> None:
        """Восстанавливает историю цен из Redis."""
        for sym, prices in data.items():
            q = self._prices[sym]
            q.clear()
            q.extend(prices[-(self._window + 1):])
