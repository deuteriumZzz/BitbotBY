"""
Параллельный сканер рынка.

За одну итерацию:
1. fetch_tickers() — один API-вызов, получаем все пары.
2. Фильтр: только /USDT, топ-N по объёму за 24ч.
3. asyncio.gather() — параллельная загрузка OHLCV + индикаторов.
4. build_snapshot() — снэпшот монеты для AI-анализатора.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config import JUNK_BASES, JUNK_PREFIXES, MEME_COIN_BASES, STABLECOIN_BASES, Config
from src.bybit_api import BybitAPI
from src.data_loader import DataLoader

# SMA-50 is the slowest indicator — need at least this many bars before
# calculate_technical_indicators() produces meaningful (non-zero) values.
_MIN_INDICATOR_ROWS = 50


class MarketScanner:
    """
    Параллельный сканер рынка.

    За одну итерацию:
    1. fetch_tickers() — один API-вызов, получаем все пары.
    2. Фильтр: только /USDT, топ-N по объёму за 24ч.
    3. asyncio.gather() — параллельная загрузка OHLCV + индикаторов.
    4. build_snapshot() — снэпшот монеты для AI-анализатора.
    """

    def __init__(self, api: BybitAPI, data_loader: DataLoader, rc: "Any | None" = None):
        self.api = api
        self.data_loader = data_loader
        self._rc = rc
        self.logger = logging.getLogger(__name__)

    async def get_top_symbols(
        self,
        n: int = 20,
        forced: set | None = None,
        excluded: set | None = None,
        bluechip_bases: frozenset | None = None,
        altcoin_exclude_bases: frozenset | None = None,
        buffer: int = 0,
    ) -> List[str]:
        """
        Возвращает топ-(N+buffer) кандидатов /USDT по объёму за 24ч.

        По умолчанию buffer=n (100% запас): для n=20 берём 40 кандидатов,
        для n=100 берём 200. Один сетевой вызов покрывает практически любой
        процент провалов при загрузке OHLCV.
        _scan_and_update_correlations обрезает итог до n в порядке объёма.

        :param n: Целевое количество монет после загрузки данных.
        :param buffer: Запас сверх n. 0 = использовать n (двойной пул).
        :param forced: Символы, всегда добавляемые в список (через /add).
        :param excluded: Символы, исключённые из сканирования (через /remove).
        :param bluechip_bases: Если задан — вернуть только монеты из этого набора.
        :param altcoin_exclude_bases: Если задан — исключить эти базы (altcoin сезон).
        :return: До (N+buffer) монет в формате ccxt ('BTC/USDT'), в порядке объёма.
        """
        forced = forced or set()
        excluded = excluded or set()
        try:
            params = {"category": "linear"} if Config.MARKET_TYPE == "linear" else {}
            tickers = await self.api.exchange.fetch_tickers(params=params)
            usdt: dict = {}
            for sym, t in tickers.items():
                if "/USDT" not in sym:
                    continue
                base_sym = sym.split(":")[0]
                base = base_sym.split("/")[0]
                if base in STABLECOIN_BASES:
                    continue
                if base in JUNK_BASES:
                    continue
                is_meme = (
                    self._rc is not None and self._rc.get_market_profile() == "meme"
                )
                if not is_meme and base in MEME_COIN_BASES:
                    continue
                if any(base.startswith(p) for p in JUNK_PREFIXES):
                    continue
                if base_sym in excluded:
                    continue
                # Сезонная фильтрация ДО обрезки по N — гарантирует нужное число монет
                if bluechip_bases is not None and base not in bluechip_bases:
                    continue
                if altcoin_exclude_bases is not None and base in altcoin_exclude_bases:
                    continue
                qv = t.get("quoteVolume") or 0
                if qv <= 0:
                    continue
                min_vol = (
                    self._rc.get_min_volume_usdt()
                    if self._rc is not None
                    else Config.MIN_VOLUME_USDT
                )
                max_vol = (
                    self._rc.get_max_volume_usdt()
                    if self._rc is not None
                    else Config.MAX_VOLUME_USDT
                )
                if qv < min_vol:
                    continue
                if max_vol > 0 and qv > max_vol and base not in MEME_COIN_BASES:
                    continue
                usdt[base_sym] = t

            ranked = sorted(
                usdt.items(),
                key=lambda x: x[1].get("quoteVolume", 0),
                reverse=True,
            )
            fetch_n = n + (buffer if buffer > 0 else n)
            symbols = [sym for sym, _ in ranked[:fetch_n]]

            # Forced символы добавляются поверх топа если их там нет
            for sym in forced:
                if sym not in symbols:
                    symbols.append(sym)

            self.logger.info(
                "Top %d by volume: %s...", len(symbols), ", ".join(symbols[:5])
            )
            return symbols
        except Exception as e:
            self.logger.error("fetch_tickers failed: %s", e)
            return Config.SYMBOLS[:n]

    async def _fetch_one(
        self,
        symbol: str,
        timeframe: str,
    ) -> Optional[Tuple[str, pd.DataFrame]]:
        """
        Загружает OHLCV + индикаторы для одного символа.

        :param symbol: Символ торговой пары.
        :param timeframe: Таймфрейм ccxt.
        :return: Кортеж (symbol, DataFrame) или None при ошибке/нехватке данных.
        """
        try:
            df = await self.data_loader.get_historical_data(symbol, timeframe, days=30)
            if df.empty:
                df = await self.data_loader.get_market_data(
                    symbol, timeframe, limit=100
                )
            if len(df) < _MIN_INDICATOR_ROWS:
                self.logger.debug(
                    "Skip %s: only %d bars, need >= %d for valid indicators",
                    symbol,
                    len(df),
                    _MIN_INDICATOR_ROWS,
                )
                return None
            df = self.data_loader.calculate_technical_indicators(df)
            return symbol, df
        except Exception as e:
            self.logger.debug("Skip %s: %s", symbol, e)
            return None

    async def scan_all(
        self,
        symbols: List[str],
        timeframe: str,
    ) -> Dict[str, pd.DataFrame]:
        """
        Параллельная загрузка данных для всех символов.

        :param symbols: Список торговых пар.
        :param timeframe: Таймфрейм ccxt ('15m', '1h', ...).
        :return: {symbol: DataFrame с индикаторами}
        """
        tasks = [self._fetch_one(s, timeframe) for s in symbols]
        results = await asyncio.gather(*tasks)

        data: Dict[str, pd.DataFrame] = {}
        for r in results:
            if r is not None:
                sym, df = r
                data[sym] = df

        self.logger.info("Scanned %d/%d symbols", len(data), len(symbols))
        return data

    @staticmethod
    def _safe(row: pd.Series, col: str, default: float = 0.0) -> float:
        """
        Безопасно извлекает числовое значение из строки DataFrame.

        :param row: Строка DataFrame.
        :param col: Название колонки.
        :param default: Значение по умолчанию если колонка отсутствует.
        :return: Числовое значение или default.
        """
        if col in row.index and pd.notna(row[col]):
            return float(row[col])
        return default

    def _pct(self, current: float, df: pd.DataFrame, bars_back: int) -> float:
        """
        Рассчитывает процентное изменение цены за N свечей назад.

        :param current: Текущая цена.
        :param df: DataFrame с колонкой 'close'.
        :param bars_back: Количество свечей назад.
        :return: Изменение в процентах, округлённое до 2 знаков.
        """
        if len(df) > bars_back:
            past = float(df.iloc[-(bars_back + 1)]["close"])
        else:
            past = float(df.iloc[0]["close"])
        return round((current - past) / past * 100, 2) if past else 0.0

    def build_snapshot(
        self,
        symbol: str,
        df: pd.DataFrame,
        news_sentiment: float,
        headlines: List[str],
    ) -> Dict[str, Any]:
        """
        Строит снэпшот монеты для AIAnalyzer.

        Содержит: цену, изменения за 1ч/24ч/7д, индикаторы (RSI,
        MACD, BB, тренд), уровни поддержки/сопротивления, ATR,
        объём, новостной сентимент и заголовки.

        :param symbol: Символ ('SOL/USDT').
        :param df: DataFrame с OHLCV + индикаторами.
        :param news_sentiment: Compound VADER (-1..1).
        :param headlines: Заголовки новостей.
        :return: Словарь-снэпшот.
        """
        if df.empty or len(df) < 2:
            return {}

        last = df.iloc[-1]
        close = float(last["close"])

        # Изменения (на 15m свечах: 4=1h, 96=24h, 672=7d)
        ch_1h = self._pct(close, df, 4)
        ch_24h = self._pct(close, df, 96)
        ch_7d = self._pct(close, df, 672)

        rsi = self._safe(last, "rsi", 50)
        macd = self._safe(last, "macd")
        macd_sig = self._safe(last, "macd_signal")
        bb_u = self._safe(last, "bb_upper", close * 1.02)
        bb_l = self._safe(last, "bb_lower", close * 0.98)
        bb_m = self._safe(last, "bb_middle", close)
        ema_s = self._safe(last, "ema_short", close)
        ema_l = self._safe(last, "ema_long", close)
        sma20 = self._safe(last, "sma_20", close)
        sma50 = self._safe(last, "sma_50", close)
        vol_ratio = self._safe(last, "volume_ratio", 1.0)
        atr = self._safe(last, "atr", close * 0.02)

        bb_width = (bb_u - bb_l) / bb_m if bb_m > 0 else 0

        if ema_s > ema_l and sma20 > sma50:
            trend = "uptrend"
        elif ema_s < ema_l and sma20 < sma50:
            trend = "downtrend"
        else:
            trend = "sideways"

        if close <= bb_l * 1.005:
            bb_pos = "near_lower"
        elif close >= bb_u * 0.995:
            bb_pos = "near_upper"
        else:
            bb_pos = "middle"

        prev20 = df.iloc[-21:-1] if len(df) >= 21 else df.iloc[:-1]
        resistance = float(prev20["high"].max())
        support = float(prev20["low"].min())

        return {
            "symbol": symbol,
            "price": round(close, 6),
            "atr": round(atr, 6),
            "changes": {
                "1h": f"{ch_1h:+.2f}%",
                "24h": f"{ch_24h:+.2f}%",
                "7d": f"{ch_7d:+.2f}%",
            },
            "volume_ratio": round(vol_ratio, 2),
            # Точные значения последней свечи для SAC-инференса
            "ohlcv": {
                "open": round(self._safe(last, "open", close), 6),
                "high": round(self._safe(last, "high", close), 6),
                "low": round(self._safe(last, "low", close), 6),
                "close": round(close, 6),
                "volume": round(self._safe(last, "volume", 0.0), 2),
                "macd": round(macd, 6),
                "macd_signal": round(macd_sig, 6),
            },
            "indicators": {
                "rsi": round(rsi, 1),
                "macd": "bullish" if macd > macd_sig else "bearish",
                "bb_position": bb_pos,
                "bb_width": round(bb_width, 4),
                "trend": trend,
            },
            "levels": {
                "resistance": round(resistance, 6),
                "support": round(support, 6),
            },
            "news_sentiment": round(news_sentiment, 3),
            "top_headlines": headlines[:3],
            "timestamp": datetime.now().isoformat(),
        }
