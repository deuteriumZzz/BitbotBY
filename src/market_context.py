"""
Рыночный контекст: funding rate, open interest и Fear & Greed Index.

Используется как contrarian-фильтр в SignalCombiner — когда рынок
перегрет в одну сторону, сигналы в ту же сторону ослабляются.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_BYBIT_BASE = "https://api.bybit.com"
_FNG_URL = "https://api.alternative.me/fng/?limit=1"

_CACHE_TTL = 300.0  # 5 минут
_TRENDS_TTL = 3600.0  # 60 минут
_PCR_TTL = 900.0  # 15 минут
_OB_TTL = 30.0  # 30 секунд
_GLASSNODE_TTL = 1800.0  # 30 минут
_ETF_TTL = 14400.0  # 4 часа
_REDDIT_TTL = 1800.0  # 30 минут
_STABLECOIN_TTL = 3600.0  # 60 минут

_DERIBIT_OPTION_SYMBOLS = {"BTC", "ETH"}

_COINGECKO_COIN_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "ADA": "cardano",
    "AVAX": "avalanche-2",
    "DOT": "polkadot",
    "MATIC": "matic-network",
    "LINK": "chainlink",
}


def _symbol_to_bybit(symbol: str) -> str:
    """Конвертирует 'BTC/USDT' → 'BTCUSDT'."""
    return symbol.replace("/", "")


def _symbol_to_base(symbol: str) -> str:
    """Конвертирует 'BTC/USDT' → 'BTC'."""
    return symbol.split("/")[0]


class MarketContext:
    """
    Получает и кэширует рыночный контекст из Bybit API и alternative.me.

    Все запросы кэшируются на 5 минут — бот вызывает это каждые 30 сек,
    но внешние API дёргаются не чаще одного раза в 5 мин на символ.
    """

    def __init__(self) -> None:
        # {symbol: (context_dict, timestamp)}
        self._cache: Dict[str, Tuple[dict, float]] = {}
        # FNG кэшируется отдельно — он глобальный, не per-symbol
        self._fng_cache: Tuple[Optional[int], float] = (None, 0.0)
        # Google Trends — обновляется раз в час
        self._trends_cache: Tuple[Tuple[int, str], float] = ((50, "neutral"), 0.0)
        # Deribit options (PCR + IV skew) — per-base (BTC/ETH), 15 мин
        self._deribit_cache: Dict[str, Tuple[Tuple[float, str, float, str], float]] = {}
        # Orderbook imbalance — per-symbol, 30 сек
        self._ob_cache: Dict[str, Tuple[Tuple[float, str], float]] = {}
        # Glassnode/CoinGecko macro — per-base, 30 мин
        self._glassnode_cache: Dict[str, Tuple[Tuple[float, str], float]] = {}
        # BTC ETF flows — global, 4 часа
        self._etf_cache: Tuple[Tuple[float, str], float] = ((0.0, "neutral"), 0.0)
        # Reddit sentiment — per-base, 30 мин
        self._reddit_cache: Dict[str, Tuple[Tuple[float, str], float]] = {}
        # Stablecoin supply — global, 60 мин
        self._stablecoin_cache: Tuple[Tuple[float, str], float] = (
            (0.0, "neutral"),
            0.0,
        )

    async def get_context(self, symbol: str, current_price: float) -> dict:
        """
        Возвращает рыночный контекст для символа.

        :param symbol: Символ ccxt ('BTC/USDT').
        :param current_price: Текущая цена (нужна для OI-сигнала).
        :return: Словарь с полями контекста.
        """
        now = time.monotonic()
        cached = self._cache.get(symbol)
        if cached and now - cached[1] < _CACHE_TTL:
            return cached[0]

        base = _symbol_to_base(symbol)
        is_btc = symbol == "BTC/USDT"

        funding_rate, funding_signal = await self._get_funding(symbol)
        oi_signal, liquidation_pressure = await self._get_oi_signal(
            symbol, current_price
        )
        fear_greed, fng_signal = await self._get_fear_greed()
        basis_pct, basis_signal = await self._get_basis(symbol, current_price)
        ob_imbalance, ob_signal = await self._get_orderbook_imbalance(symbol)

        if base == "BTC":
            google_trends, google_trends_signal = await self._get_google_trends(
                "buy bitcoin"
            )
        else:
            google_trends, google_trends_signal = 50, "neutral"

        if base in _DERIBIT_OPTION_SYMBOLS:
            pcr, pcr_signal, iv_skew, iv_signal = await self._get_deribit_options(base)
        else:
            pcr, pcr_signal, iv_skew, iv_signal = 1.0, "neutral", 0.0, "neutral"

        macro_val, macro_signal = await self._get_glassnode(base)

        if is_btc:
            etf_flow, etf_signal = await self._get_btc_etf_flows()
            stablecoin_val, stablecoin_signal = (
                await self._get_stablecoin_supply_change()
            )
        else:
            etf_flow, etf_signal = 0.0, "neutral"
            _, stablecoin_signal = 0.0, "neutral"

        reddit_val, reddit_signal = await self._get_reddit_sentiment(base)

        ctx = {
            "funding_rate": funding_rate,
            "funding_signal": funding_signal,
            "oi_signal": oi_signal,
            "liquidation_pressure": liquidation_pressure,
            "fear_greed": fear_greed,
            "fear_greed_signal": fng_signal,
            "basis_pct": basis_pct,
            "basis_signal": basis_signal,
            "google_trends": google_trends,
            "google_trends_signal": google_trends_signal,
            "pcr": pcr,
            "pcr_signal": pcr_signal,
            "ob_imbalance": ob_imbalance,
            "ob_signal": ob_signal,
            "iv_skew": iv_skew,
            "iv_signal": iv_signal,
            "macro_signal": macro_signal,
            "etf_flow": etf_flow,
            "etf_signal": etf_signal,
            "reddit_signal": reddit_signal,
            "stablecoin_signal": stablecoin_signal,
        }
        self._cache[symbol] = (ctx, now)
        logger.info(
            "MarketContext [%s]: funding=%.5f(%s) oi=%s liq=%s fng=%s(%s) "
            "basis=%.2f%%(%s) trends=%s(%s) pcr=%.3f(%s) ob=%.3f(%s) "
            "iv_skew=%.2f(%s) macro=%s etf=%.1f(%s) reddit=%s stable=%s",
            symbol,
            funding_rate,
            funding_signal,
            oi_signal,
            liquidation_pressure,
            fear_greed,
            fng_signal,
            basis_pct,
            basis_signal,
            google_trends,
            google_trends_signal,
            pcr,
            pcr_signal,
            ob_imbalance,
            ob_signal,
            iv_skew,
            iv_signal,
            macro_signal,
            etf_flow,
            etf_signal,
            reddit_signal,
            stablecoin_signal,
        )
        return ctx

    async def get_context_for_symbols(
        self, symbols: List[str], prices: Dict[str, float]
    ) -> Dict[str, dict]:
        """Возвращает {symbol: context} для всех символов параллельно."""
        tasks = {sym: self.get_context(sym, prices.get(sym, 0.0)) for sym in symbols}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        out: Dict[str, dict] = {}
        for sym, res in zip(tasks.keys(), results):
            out[sym] = res if isinstance(res, dict) else self._neutral_context()
        return out

    def _neutral_context(self) -> dict:
        return {
            "funding_rate": 0.0,
            "funding_signal": "neutral",
            "oi_signal": "oi_neutral",
            "liquidation_pressure": "neutral",
            "fear_greed": 50,
            "fear_greed_signal": "neutral",
            "basis_pct": 0.0,
            "basis_signal": "neutral",
            "google_trends": 50,
            "google_trends_signal": "neutral",
            "pcr": 1.0,
            "pcr_signal": "neutral",
            "ob_imbalance": 0.0,
            "ob_signal": "balanced",
            "iv_skew": 0.0,
            "iv_signal": "neutral",
            "macro_signal": "neutral",
            "etf_flow": 0.0,
            "etf_signal": "neutral",
            "reddit_signal": "neutral",
            "stablecoin_signal": "neutral",
        }

    # ── Funding rate ──────────────────────────────────────────────────────────

    async def _get_funding(self, symbol: str) -> Tuple[float, str]:
        """
        Получает последний funding rate для perpetual-символа.

        Возвращает (0.0, 'neutral') для спот-символов или при ошибке.
        """
        try:
            import aiohttp

            sym_bybit = _symbol_to_bybit(symbol)
            url = (
                f"{_BYBIT_BASE}/v5/market/funding/history"
                f"?category=linear&symbol={sym_bybit}&limit=1"
            )
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    data = await resp.json()

            result_list = (data.get("result") or {}).get("list") or []
            if not result_list:
                return 0.0, "neutral"

            rate = float(result_list[0].get("fundingRate", 0))
            if rate > 0.001:
                signal = "long_overheated"
            elif rate < -0.0005:
                signal = "short_overheated"
            else:
                signal = "neutral"
            return rate, signal

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("Funding rate fetch failed for %s: %s", symbol, e)
            return 0.0, "neutral"

    # ── Open Interest + Liquidation Pressure ──────────────────────────────────

    async def _get_oi_signal(
        self, symbol: str, current_price: float
    ) -> Tuple[str, str]:
        """
        Сравнивает последние 2 значения OI для определения давления.

        OI растёт + цена падает → "oi_bearish" (новые шорты открываются)
        OI падает + цена растёт → "oi_bullish" (шорты закрываются, шорт-сквиз)
        Также возвращает liquidation_pressure на основе резкого изменения OI.
        """
        try:
            import aiohttp

            sym_bybit = _symbol_to_bybit(symbol)
            url = (
                f"{_BYBIT_BASE}/v5/market/open-interest"
                f"?category=linear&symbol={sym_bybit}&intervalTime=15min&limit=2"
            )
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    data = await resp.json()

            result_list = (data.get("result") or {}).get("list") or []
            if len(result_list) < 2:
                return "oi_neutral", "neutral"

            # Bybit возвращает от новейшего к старейшему
            oi_now = float(result_list[0].get("openInterest", 0))
            oi_prev = float(result_list[1].get("openInterest", 0))
            if oi_prev <= 0:
                return "oi_neutral", "neutral"

            oi_growing = oi_now > oi_prev

            oi_drop_pct = (oi_prev - oi_now) / oi_prev if oi_prev > 0 else 0

            # Для определения направления цены сравниваем с предыдущим кэшем
            cached = self._cache.get(symbol)
            if cached:
                prev_price = cached[0].get("_prev_price", current_price)
            else:
                prev_price = current_price

            price_falling = current_price < prev_price * 0.9995

            if oi_drop_pct > 0.02 and price_falling:
                liquidation_pressure = (
                    "long_liquidation"  # OI упал + цена падает = лонги ликвидируются
                )
            elif oi_drop_pct > 0.02 and not price_falling:
                liquidation_pressure = (
                    "short_squeeze"  # OI упал + цена растёт = шорты закрываются
                )
            else:
                liquidation_pressure = "neutral"

            if oi_growing and price_falling:
                signal = "oi_bearish"
            elif not oi_growing and not price_falling:
                signal = "oi_bullish"
            else:
                signal = "oi_neutral"

            # Сохраняем текущую цену для следующего сравнения
            if cached:
                cached[0]["_prev_price"] = current_price

            return signal, liquidation_pressure

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("OI fetch failed for %s: %s", symbol, e)
            return "oi_neutral", "neutral"

    # ── Basis (futures - spot premium) ────────────────────────────────────────

    async def _get_basis(self, symbol: str, spot_price: float) -> Tuple[float, str]:
        """
        Basis = (futures_price - spot_price) / spot_price * 100
        > 1.5% → greed premium → bearish signal
        < -0.5% → backwardation → bullish
        """
        if spot_price <= 0:
            return 0.0, "neutral"
        try:
            import aiohttp

            sym_bybit = _symbol_to_bybit(symbol)
            url = (
                f"{_BYBIT_BASE}/v5/market/tickers"
                f"?category=linear&symbol={sym_bybit}"
            )
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    data = await resp.json()

            result_list = (data.get("result") or {}).get("list") or []
            if not result_list:
                return 0.0, "neutral"

            futures_price = float(result_list[0].get("lastPrice", 0))
            if futures_price <= 0:
                return 0.0, "neutral"

            basis_pct = (futures_price - spot_price) / spot_price * 100.0
            if basis_pct > 1.5:
                basis_signal = "greed_premium"
            elif basis_pct < -0.5:
                basis_signal = "backwardation"
            else:
                basis_signal = "neutral"

            return round(basis_pct, 4), basis_signal

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("Basis fetch failed for %s: %s", symbol, e)
            return 0.0, "neutral"

    # ── Fear & Greed ──────────────────────────────────────────────────────────

    async def _get_fear_greed(self) -> Tuple[int, str]:
        """
        Получает Fear & Greed Index с кэшем 5 минут.

        Возвращает (50, 'neutral') при ошибке — нейтральное значение.
        """
        now = time.monotonic()
        cached_val, cached_ts = self._fng_cache
        if cached_val is not None and now - cached_ts < _CACHE_TTL:
            return cached_val, self._fng_to_signal(cached_val)

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    _FNG_URL, timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    data = await resp.json(content_type=None)

            value = int((data.get("data") or [{}])[0].get("value", 50))
            self._fng_cache = (value, now)
            return value, self._fng_to_signal(value)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("Fear&Greed fetch failed: %s", e)
            return 50, "neutral"

    @staticmethod
    def _fng_to_signal(value: int) -> str:
        if value < 25:
            return "extreme_fear"
        if value > 75:
            return "extreme_greed"
        return "neutral"

    # ── Google Trends ─────────────────────────────────────────────────────────

    async def _get_google_trends(self, keyword: str = "buy bitcoin") -> Tuple[int, str]:
        """
        Возвращает (current_interest, signal).
        current_interest: 0-100 (относительный интерес за последнюю неделю)
        > 75 → розница активно ищет → contrarian SHORT signal
        < 20 → никто не интересуется → contrarian LONG (дно рынка)
        """
        now = time.monotonic()
        cached_val, cached_ts = self._trends_cache
        if cached_ts > 0 and now - cached_ts < _TRENDS_TTL:
            return cached_val

        loop = asyncio.get_running_loop()
        try:
            from pytrends.request import TrendReq

            def fetch() -> int:
                pt = TrendReq(hl="en-US", tz=0, timeout=(5, 10))
                pt.build_payload([keyword], timeframe="now 7-d")
                df = pt.interest_over_time()
                if df.empty:
                    return 50
                return int(df[keyword].iloc[-1])

            value = await loop.run_in_executor(None, fetch)
            signal = (
                "retail_fomo"
                if value > 75
                else ("retail_absent" if value < 20 else "neutral")
            )
            self._trends_cache = ((value, signal), now)
            return value, signal
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("Google Trends failed: %s", e)
            return 50, "neutral"

    # ── Deribit options (PCR + IV Skew) ──────────────────────────────────────

    async def _get_deribit_options(
        self, base: str = "BTC"
    ) -> Tuple[float, str, float, str]:
        """Возвращает (pcr, pcr_signal, iv_skew, iv_signal) за один запрос."""
        now = time.monotonic()
        cached = self._deribit_cache.get(base)
        if cached and now - cached[1] < _PCR_TTL:
            return cached[0]

        try:
            import aiohttp

            url = (
                f"https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
                f"?currency={base}&kind=option"
            )
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=8)
                ) as resp:
                    data = await resp.json()

            result = data.get("result", [])

            put_volume = sum(
                r.get("volume", 0)
                for r in result
                if r.get("instrument_name", "").endswith("-P")
            )
            call_volume = sum(
                r.get("volume", 0)
                for r in result
                if r.get("instrument_name", "").endswith("-C")
            )

            if call_volume == 0:
                result_val = (1.0, "neutral", 0.0, "neutral")
                self._deribit_cache[base] = (result_val, now)
                return result_val

            pcr = put_volume / call_volume
            if pcr > 1.5:
                pcr_signal = "fear_puts"
            elif pcr < 0.5:
                pcr_signal = "greed_calls"
            else:
                pcr_signal = "neutral"

            put_ivs = [
                r.get("mark_iv", 0.0)
                for r in result
                if r.get("instrument_name", "").endswith("-P")
                and r.get("volume", 0) > 0
                and r.get("mark_iv", 0.0) > 0
            ]
            call_ivs = [
                r.get("mark_iv", 0.0)
                for r in result
                if r.get("instrument_name", "").endswith("-C")
                and r.get("volume", 0) > 0
                and r.get("mark_iv", 0.0) > 0
            ]

            if put_ivs and call_ivs:
                avg_put_iv = sum(put_ivs) / len(put_ivs)
                avg_call_iv = sum(call_ivs) / len(call_ivs)
                iv_skew = avg_put_iv - avg_call_iv
            else:
                iv_skew = 0.0

            if iv_skew > 5.0:
                iv_signal = "put_skew"
            elif iv_skew < -3.0:
                iv_signal = "call_skew"
            else:
                iv_signal = "neutral"

            result_val = (round(pcr, 3), pcr_signal, round(iv_skew, 4), iv_signal)
            self._deribit_cache[base] = (result_val, now)
            return result_val

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("Deribit options fetch failed for %s: %s", base, e)
            return 1.0, "neutral", 0.0, "neutral"

    # ── Orderbook imbalance ───────────────────────────────────────────────────

    async def _get_orderbook_imbalance(self, symbol: str) -> Tuple[float, str]:
        """
        Imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        Диапазон: -1.0 (чисто продавцы) до +1.0 (чисто покупатели)
        > 0.3  → "bid_dominant"
        < -0.3 → "ask_dominant"
        иначе  → "balanced"
        """
        now = time.monotonic()
        cached = self._ob_cache.get(symbol)
        if cached and now - cached[1] < _OB_TTL:
            return cached[0]

        try:
            import aiohttp

            sym_bybit = _symbol_to_bybit(symbol)
            url = (
                f"{_BYBIT_BASE}/v5/market/orderbook"
                f"?category=linear&symbol={sym_bybit}&limit=25"
            )
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    data = await resp.json()

            result = data.get("result") or {}
            bids = result.get("b", [])
            asks = result.get("a", [])

            bid_volume = sum(float(b[1]) for b in bids if len(b) >= 2)
            ask_volume = sum(float(a[1]) for a in asks if len(a) >= 2)

            total = bid_volume + ask_volume
            if total <= 0:
                self._ob_cache[symbol] = ((0.0, "balanced"), now)
                return 0.0, "balanced"

            imbalance = (bid_volume - ask_volume) / total
            if imbalance > 0.3:
                ob_signal = "bid_dominant"
            elif imbalance < -0.3:
                ob_signal = "ask_dominant"
            else:
                ob_signal = "balanced"

            result_val = (round(imbalance, 4), ob_signal)
            self._ob_cache[symbol] = (result_val, now)
            return result_val

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("Orderbook imbalance fetch failed for %s: %s", symbol, e)
            return 0.0, "balanced"

    # ── Glassnode / CoinGecko macro ───────────────────────────────────────────

    async def _get_glassnode(self, base: str = "BTC") -> Tuple[float, str]:
        """
        Exchange Net Position Change через Glassnode (если есть ключ)
        или market_cap_change_24h через CoinGecko как прокси.
        """
        now = time.monotonic()
        cached = self._glassnode_cache.get(base)
        if cached and now - cached[1] < _GLASSNODE_TTL:
            return cached[0]

        try:
            import os

            import aiohttp

            api_key = os.getenv("GLASSNODE_API_KEY", "")

            if api_key:
                url = (
                    "https://api.glassnode.com/v1/metrics/distribution/exchange_net_position_change"  # noqa: E501
                    f"?a={base}&i=24h&api_key={api_key}"
                )
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=8)
                    ) as resp:
                        data = await resp.json(content_type=None)

                if isinstance(data, list) and data:
                    latest = data[-1]
                    value = float(latest.get("v", 0.0))
                    if value < 0:
                        signal = "macro_bullish"
                    elif value > 0:
                        signal = "macro_bearish"
                    else:
                        signal = "neutral"
                    result_val = (round(value, 4), signal)
                    self._glassnode_cache[base] = (result_val, now)
                    return result_val

            coin_id = _COINGECKO_COIN_IDS.get(base, "")
            if not coin_id:
                self._glassnode_cache[base] = ((0.0, "neutral"), now)
                return 0.0, "neutral"

            url = (
                f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                f"?localization=false&tickers=false&community_data=false"
            )
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=8)
                ) as resp:
                    data = await resp.json(content_type=None)

            market_data = data.get("market_data") or {}
            change_24h = float(market_data.get("market_cap_change_percentage_24h", 0.0))

            if change_24h < -5.0:
                signal = "macro_bearish"
            elif change_24h > 5.0:
                signal = "macro_bullish"
            else:
                signal = "neutral"

            result_val = (round(change_24h, 4), signal)
            self._glassnode_cache[base] = (result_val, now)
            return result_val

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("Glassnode/CoinGecko macro fetch failed for %s: %s", base, e)
            return 0.0, "neutral"

    # ── BTC ETF flows ─────────────────────────────────────────────────────────

    async def _get_btc_etf_flows(self) -> Tuple[float, str]:
        """
        Парсит последнюю строку таблицы с farside.co.uk/bitcoin-etf-flow-all-data-table.
        > +50M → "etf_inflow"
        < -50M → "etf_outflow"
        иначе  → "neutral"
        """
        now = time.monotonic()
        cached_val, cached_ts = self._etf_cache
        if cached_ts > 0 and now - cached_ts < _ETF_TTL:
            return cached_val

        try:
            import re

            import aiohttp

            url = "https://farside.co.uk/bitcoin-etf-flow-all-data-table"
            headers = {"User-Agent": "Mozilla/5.0 (compatible; BitbotBY/1.0)"}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=10), headers=headers
                ) as resp:
                    html = await resp.text()

            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(html, "html.parser")
                table = soup.find("table")
                if table:
                    rows = table.find_all("tr")
                    for row in reversed(rows):
                        cells = row.find_all(["td", "th"])
                        if len(cells) < 2:
                            continue
                        last_cell_text = cells[-1].get_text(strip=True)
                        last_cell_text = (
                            last_cell_text.replace(",", "")
                            .replace("(", "-")
                            .replace(")", "")
                        )
                        try:
                            total = float(last_cell_text)
                            if total != 0:
                                if total > 50:
                                    signal = "etf_inflow"
                                elif total < -50:
                                    signal = "etf_outflow"
                                else:
                                    signal = "neutral"
                                result_val = (round(total, 2), signal)
                                self._etf_cache = (result_val, now)
                                return result_val
                        except (ValueError, TypeError):
                            continue
            except ImportError:
                pattern = r"<tr[^>]*>.*?</tr>"
                rows = re.findall(pattern, html, re.DOTALL | re.IGNORECASE)
                for row in reversed(rows):
                    cells = re.findall(
                        r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL | re.IGNORECASE
                    )
                    if not cells:
                        continue
                    raw = re.sub(r"<[^>]+>", "", cells[-1]).strip()
                    raw = raw.replace(",", "").replace("(", "-").replace(")", "")
                    try:
                        total = float(raw)
                        if total != 0:
                            if total > 50:
                                signal = "etf_inflow"
                            elif total < -50:
                                signal = "etf_outflow"
                            else:
                                signal = "neutral"
                            result_val = (round(total, 2), signal)
                            self._etf_cache = (result_val, now)
                            return result_val
                    except (ValueError, TypeError):
                        continue

            self._etf_cache = ((0.0, "neutral"), now)
            return 0.0, "neutral"

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("ETF flows fetch failed: %s", e)
            return 0.0, "neutral"

    # ── Reddit sentiment ──────────────────────────────────────────────────────

    async def _get_reddit_sentiment(self, base: str) -> Tuple[float, str]:
        """
        Средний upvote_ratio последних 10 постов в r/CryptoCurrency по тикеру.
        upvote_ratio < 0.5  → "reddit_bearish"
        upvote_ratio > 0.75 → "reddit_bullish"
        иначе               → "neutral"
        """
        now = time.monotonic()
        cached = self._reddit_cache.get(base)
        if cached and now - cached[1] < _REDDIT_TTL:
            return cached[0]

        try:
            import os

            client_id = os.getenv("REDDIT_CLIENT_ID", "")
            client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
            user_agent = os.getenv("REDDIT_USER_AGENT", "BitbotBY/1.0")

            if not client_id or not client_secret:
                self._reddit_cache[base] = ((0.0, "neutral"), now)
                return 0.0, "neutral"

            import praw

            loop = asyncio.get_running_loop()

            def fetch() -> float:
                reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent,
                )
                subreddit = reddit.subreddit("CryptoCurrency")
                posts = list(subreddit.search(base, limit=10, time_filter="day"))
                if not posts:
                    return 0.5
                ratios = [p.upvote_ratio for p in posts if hasattr(p, "upvote_ratio")]
                return sum(ratios) / len(ratios) if ratios else 0.5

            avg_ratio = await loop.run_in_executor(None, fetch)

            if avg_ratio < 0.5:
                signal = "reddit_bearish"
            elif avg_ratio > 0.75:
                signal = "reddit_bullish"
            else:
                signal = "neutral"

            result_val = (round(avg_ratio, 4), signal)
            self._reddit_cache[base] = (result_val, now)
            return result_val

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("Reddit sentiment fetch failed for %s: %s", base, e)
            return 0.0, "neutral"

    # ── Stablecoin supply change ──────────────────────────────────────────────

    async def _get_stablecoin_supply_change(self) -> Tuple[float, str]:
        """
        Рост supply USDT = свежий кэш входит в рынок = потенциально бычий.
        > 0.5%  → "stablecoin_inflow"
        < -0.3% → "stablecoin_outflow"
        иначе   → "neutral"
        """
        now = time.monotonic()
        cached_val, cached_ts = self._stablecoin_cache
        if cached_ts > 0 and now - cached_ts < _STABLECOIN_TTL:
            return cached_val

        try:
            import aiohttp

            url = "https://api.coingecko.com/api/v3/coins/tether"
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=8)
                ) as resp:
                    data = await resp.json(content_type=None)

            market_data = data.get("market_data") or {}
            change_pct = (
                market_data.get("market_cap_change_percentage_24h_in_currency") or {}
            ).get("usd", 0.0)
            change_pct = float(change_pct)

            if change_pct > 0.5:
                signal = "stablecoin_inflow"
            elif change_pct < -0.3:
                signal = "stablecoin_outflow"
            else:
                signal = "neutral"

            result_val = (round(change_pct, 4), signal)
            self._stablecoin_cache = (result_val, now)
            return result_val

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("Stablecoin supply fetch failed: %s", e)
            return 0.0, "neutral"
