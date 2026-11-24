"""Основные стратегии для любого рынка: включены по умолчанию.
"""

from typing import Any, Dict

import pandas as pd

from bitbot.domain.strategies.base import BaseStrategy

# 17. CVD DIVERGENCE (ICT tier-1, приближённый — не тик-данные)


class CVDDivergenceStrategy(BaseStrategy):
    """
    Дивергенция кумулятивной дельты объёма (приближённый CVD).

    Настоящий order-flow CVD требует тик-данных с классификацией
    покупок/продаж на уровне сделок — в OHLCV-свечах их нет. Здесь дельта
    свечи приближается направлением тела (+volume если close>open, -volume
    если close<open) — грубее честного footprint-CVD, поэтому явно
    помечено как приближение, а не тик-точный расчёт. Ищет классическую
    дивергенцию: цена обновляет экстремум, а приближённый CVD — нет
    (накопление/распределение против текущего движения цены).
    """

    name = "cvd_divergence"
    description = (
        "Дивергенция цены и приближённой кумулятивной дельты объёма "
        "(по направлению тела свечи, не тик-данные). Разворотный сигнал."
    )
    recommended_timeframes = ["15m", "1h"]
    risk_level = "high"
    market_type = "any"

    _LOOKBACK = 30

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        if len(data) < self._LOOKBACK:
            return {"action": "hold", "confidence": 0.50}

        window = data.iloc[-self._LOOKBACK :]
        close = self._last(data, "close")
        rsi = self._last(data, "rsi", 50)

        direction = (window["close"] - window["open"]).apply(
            lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
        )
        cvd = (direction * window["volume"]).cumsum()

        half = self._LOOKBACK // 2
        price_first, price_second = (
            window["close"].iloc[:half],
            window["close"].iloc[half:],
        )
        cvd_first, cvd_second = cvd.iloc[:half], cvd.iloc[half:]
        avg_vol = float(window["volume"].mean()) * half + 1e-9

        price_new_low = price_second.min() < price_first.min()
        cvd_higher_low = cvd_second.min() > cvd_first.min()
        price_new_high = price_second.max() > price_first.max()
        cvd_lower_high = cvd_second.max() < cvd_first.max()

        if price_new_low and cvd_higher_low and rsi < 50:
            spread = abs(float(cvd_second.min() - cvd_first.min()))
            conf = round(min(0.60 + min(spread / avg_vol, 1.0) * 0.2, 0.82), 2)
            return {"action": "buy", "confidence": conf, "price": close}

        if price_new_high and cvd_lower_high and rsi > 50:
            spread = abs(float(cvd_first.max() - cvd_second.max()))
            conf = round(min(0.60 + min(spread / avg_vol, 1.0) * 0.2, 0.82), 2)
            return {"action": "sell", "confidence": conf, "price": close}

        return {"action": "hold", "confidence": 0.40}
