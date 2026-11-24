"""Стратегии, перенесённые из Argent-Systems (github.com/deuteriumZzz/Argent-Systems).

В Argent каждая стратегия — функция signals(df), которая возвращает целевой вес
позиции на каждый бар (+1 лонг, 0 вне рынка, −1 шорт). Нашему боту нужен сигнал на
последнем баре: «открыть сейчас или нет». Поэтому здесь:

* вес считается так же, как в Argent (все операции причинные: вес бара зависит
  только от этого и прошлых баров);
* сигнал buy/sell выдаётся только в момент ВХОДА — когда вес на последнем баре
  изменился и не равен нулю. Пока позиция «держится», стратегия молчит;
  выходом занимаются стоп, тейк и защиты бота, а не вес;
* уверенность у всех портов одинаковая: у них нет честной самооценки, а в
  турнире локального отбора побеждает самая уверенная — завышать её нельзя.

Ни у одной из этих стратегий нет доказанного преимущества (по собственной проверке
Argent ни одна ценовая стратегия не значима по Deflated Sharpe). Это материал для
бэктеста и разнообразия, а не обещание прибыли.

Здесь — общий адаптер `PortedStrategy`, помощники, которыми пользуются
несколько портов, и `indicator_strategy` для индикаторных портов. Сами стратегии —
в experimental.py своих семейств.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Type

import numpy as np
import pandas as pd

from bitbot.domain.strategies.base import BaseStrategy

# Сколько последних свечей считаем вживую: хватает для самых длинных окон
# (200 + 14 + 90) с запасом на «прогрев» рекурсивных индикаторов.
_LIVE_TAIL = 1000


_CONFIDENCE = 0.70


class PortedStrategy(BaseStrategy):
    """База: вес позиции по Argent → сигнал входа на последнем баре."""

    confidence = _CONFIDENCE
    min_bars = 60
    experimental = True

    def __init__(self) -> None:
        # Вес, посчитанный заранее на всей истории (только бэктест, см. prepare()).
        self._cache: Optional[Tuple[pd.DataFrame, pd.Series]] = None

    def weights(self, df: pd.DataFrame) -> pd.Series:
        """Целевой вес позиции на каждый бар: +1, 0 или −1."""
        raise NotImplementedError

    def prepare(self, df: pd.DataFrame) -> None:
        """Бэктест: считает вес на всей истории один раз, а не на каждом шаге.

        Допустимо, потому что вес причинный: вес на префиксе истории равен
        префиксу веса на всей истории (это проверяется тестом).
        """
        self._cache = (df, self.weights(df))

    def _weights_for(self, data: pd.DataFrame) -> pd.Series:
        if self._cache is not None:
            full, w = self._cache
            n = len(data)
            if (
                n <= len(full)
                and data.index[0] == full.index[0]
                and data["close"].iloc[-1] == full["close"].iloc[n - 1]
                and data["volume"].iloc[-1] == full["volume"].iloc[n - 1]
            ):
                return w.iloc[:n]
        return self.weights(data.tail(_LIVE_TAIL))

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        if len(data) < self.min_bars:
            return {"action": "hold", "confidence": 0.50}
        w = self._weights_for(data)
        now, before = float(w.iloc[-1]), float(w.iloc[-2])
        if now != before and now != 0.0:
            return {
                "action": "buy" if now > 0 else "sell",
                "confidence": self.confidence,
                "price": float(data["close"].iloc[-1]),
            }
        return {"action": "hold", "confidence": 0.45}


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()],
        axis=1,
    ).max(axis=1)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _hold_state(
    events_long: pd.Series, events_short: pd.Series, flat: Optional[pd.Series] = None
) -> pd.Series:
    """Вес, который «залипает» между событиями (как ffill в Argent)."""
    w = pd.Series(np.nan, index=events_long.index)
    if flat is not None:
        w[flat] = 0.0
    w[events_short] = -1.0
    w[events_long] = 1.0
    return w.ffill().fillna(0.0)


def indicator_strategy(
    short: str,
    signals: Callable[..., pd.Series],
    *,
    description: str,
    min_bars: int,
    market_type: str,
    markets: str,
    risk: str,
) -> Type[BaseStrategy]:
    """Стратегия из индикаторной функции Argent `signals(df, allow_short)` → вес."""

    def weights(self: Any, df: pd.DataFrame) -> pd.Series:
        return signals(df, allow_short=True).astype(float).fillna(0.0)

    return type(
        "".join(p.title() for p in short.split("_")) + "Strategy",
        (PortedStrategy,),
        {
            "__module__": signals.__module__,  # иначе type() запишет «abc»
            "name": short,
            "description": description,
            "recommended_timeframes": ["15m", "1h"],
            "risk_level": risk,
            "market_type": market_type,
            "markets": markets,
            "min_bars": min_bars,
            "weights": weights,
        },
    )
