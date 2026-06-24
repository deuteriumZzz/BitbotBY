"""
Комбинатор сигналов SAC и AI для разных режимов работы бота.

Поддерживает режимы: local (пустой список), dqn (только SAC),
ai (только Claude API), hybrid (согласие обоих источников).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from config import Config
from src import chronos_analyzer
from src.ai_analyzer import AIAnalyzer
from src.dqn_signal import SACSignal

logger = logging.getLogger(__name__)


_REGIME_WEIGHTS: dict[str, tuple[float, float]] = {
    # (sac_weight, ai_weight) per detected market regime
    "trending_up": (0.5, 0.5),  # momentum favours SAC
    "trending_down": (0.3, 0.7),  # uncertainty → trust AI more
    "ranging": (0.4, 0.6),  # default balance
    "unknown": (0.4, 0.6),
}


class SignalCombiner:
    """
    Объединяет сигналы SAC и AI согласно MODE.

    MODE=local   → [] (trading_bot использует local fallback)
    MODE=dqn     → только SAC, без Claude API
    MODE=ai      → только Claude API, без SAC
    MODE=hybrid  → оба должны согласиться; расхождение → hold

    hybrid-логика:
    - Оба buy/sell → combined conf = w_sac*SAC + w_ai*AI (режим-зависимо)
    - Расходятся → пропуск
    - AI молчит, SAC conf >= 0.80 → доверяем SAC
    """

    _W_SAC = 0.4
    _W_AI = 0.6
    _SAC_SOLO_MIN = 0.80

    def __init__(self, ai: AIAnalyzer, rc: Any = None):
        self.ai = ai
        self.sac = SACSignal()
        self._rc = rc
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            "SignalCombiner mode=%s",
            self._rc.get_mode() if self._rc else Config.MODE,
        )

    async def combine(
        self,
        snapshots: List[Dict[str, Any]],
        balance: float,
        regime: str = "unknown",
        regimes: Dict[str, str] | None = None,
        market_context: Dict[str, Any] | None = None,
        sentiment: Dict[str, float] | None = None,
        market_data: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Возвращает рекомендации согласно MODE.

        :param snapshots: Снэпшоты из MarketScanner.
        :param balance: Баланс USDT.
        :param regime: Fallback-режим (используется если regimes=None).
        :param regimes: Per-symbol режимы {"BTC/USDT": "trending_up", ...}.
        :param market_context: Контекст из MarketContext — flat dict или
            per-symbol map {"BTC/USDT": {...}, "ETH/USDT": {...}}.
            Если None — фильтр не применяется.
        :param sentiment: Словарь {symbol: score -1..1} из NewsAnalyzer.
            Если None — фильтр не применяется.
        :return: Список рекомендаций.
        """
        mode = self._rc.get_mode() if self._rc else Config.MODE

        if mode == "local":
            return []

        if mode == "ai":
            recs = await self.ai.analyze(snapshots, balance)
        elif mode == "dqn":
            recs = self._sac_only(snapshots, balance)
        elif mode == "hybrid":
            recs = await self._hybrid(
                snapshots, balance, regime, regimes, market_data or {}
            )
        else:
            self.logger.warning("Unknown MODE='%s', fallback to 'ai'", mode)
            recs = await self.ai.analyze(snapshots, balance)

        if market_context and recs:
            recs = self._apply_market_context_filter(recs, market_context)
        if sentiment and recs:
            recs = self._apply_sentiment_filter(recs, sentiment)

        if market_context and snapshots:
            context_recs = self._generate_context_signals(snapshots, market_context)
            if context_recs:
                self.logger.info(
                    "Context signals generated: %d",
                    len(context_recs),
                )
                return self._merge_with_context(recs, context_recs)

        return recs

    @staticmethod
    def _sl_tp(price: float, atr: float, action: str) -> Tuple[float, float]:
        """
        Рассчитывает стоп-лосс и тейк-профит через ATR (1.5x / 3.0x).

        :param price: Текущая цена актива.
        :param atr: Средний истинный диапазон (ATR).
        :param action: Направление сделки ("buy" или "sell").
        :return: Кортеж (stop_loss, take_profit).
        """
        if action == "buy":
            sl = price - 1.5 * atr
            tp = price + 3.0 * atr
        else:
            sl = price + 1.5 * atr
            tp = price - 3.0 * atr
        return round(sl, 6), round(tp, 6)

    def _sac_only(
        self,
        snapshots: List[Dict[str, Any]],
        balance: float,
    ) -> List[Dict[str, Any]]:
        """
        Формирует рекомендации только на основе SAC-сигналов.

        :param snapshots: Снэпшоты из MarketScanner.
        :param balance: Баланс USDT.
        :return: Список рекомендаций с action, confidence, entry, sl/tp.
        """
        results: List[Dict[str, Any]] = []
        min_conf = Config.MIN_SIGNAL_CONFIDENCE
        for snap in snapshots:
            sig = self.sac.get_signal(snap, balance)
            if sig["action"] == "hold" or sig["confidence"] < min_conf:
                continue
            price = snap.get("price", 0)
            atr = snap.get("atr", price * 0.02)
            sl, tp = self._sl_tp(price, atr, sig["action"])
            results.append(
                {
                    "symbol": snap["symbol"],
                    "action": sig["action"],
                    "strategy": "sac",
                    "confidence": sig["confidence"],
                    "entry": price,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "reasoning": f"SAC conf {sig['confidence']:.0%}",
                }
            )
        return results

    async def _hybrid(
        self,
        snapshots: List[Dict],
        balance: float,
        regime: str = "unknown",
        regimes: Dict[str, str] | None = None,
        market_data: Dict[str, Any] | None = None,
    ) -> List[Dict]:
        """
        Гибридный режим: согласие SAC + AI с режим-зависимыми весами.

        Веса определяются per-symbol режимом из regimes (если передан),
        иначе используется общий fallback-режим.
        trending_up → SAC 50%/AI 50%, trending_down → SAC 30%/AI 70%,
        ranging / unknown → SAC 40%/AI 60%.
        """
        use_chronos = (
            not Config.PAPER_TRADING
            and self._rc is not None
            and self._rc.get_chronos_enabled()
        )
        if use_chronos:
            self.logger.info("Hybrid+Chronos: enhanced triple-confirmation active")

        ai_recs = await self.ai.analyze(snapshots, balance)
        ai_map: Dict[str, Dict] = {r["symbol"]: r for r in ai_recs}

        results = []
        for snap in snapshots:
            sym = snap["symbol"]
            sym_regime = (regimes or {}).get(sym, regime)
            w_sac, w_ai = _REGIME_WEIGHTS.get(sym_regime, (self._W_SAC, self._W_AI))

            sac_sig = self.sac.get_signal(snap, balance)
            ai = ai_map.get(sym)

            d_action = sac_sig["action"]
            d_conf = sac_sig["confidence"]

            if d_action == "hold":
                continue

            # AI молчит — принимаем SAC только при высоком conf
            if ai is None:
                if d_conf >= self._SAC_SOLO_MIN:
                    price = snap.get("price", 0)
                    atr = snap.get("atr", price * 0.02)
                    sl, tp = self._sl_tp(price, atr, d_action)
                    results.append(
                        {
                            "symbol": sym,
                            "action": d_action,
                            "strategy": "sac",
                            "confidence": d_conf,
                            "entry": price,
                            "stop_loss": sl,
                            "take_profit": tp,
                            "reasoning": f"SAC {d_conf:.0%}, AI silent",
                        }
                    )
                continue

            a_action = ai.get("action", "hold")

            if d_action == a_action:
                a_conf = ai.get("confidence", 0)
                combined = round(d_conf * w_sac + a_conf * w_ai, 3)
                if combined < Config.MIN_SIGNAL_CONFIDENCE:
                    continue

                # Chronos — третья точка зрения (только live + enhanced режим)
                if use_chronos and market_data:
                    df = market_data.get(sym)
                    if df is not None and not df.empty and "close" in df.columns:
                        chronos_dir = chronos_analyzer.predict_direction(
                            df["close"].tolist()
                        )
                        expected = "up" if d_action == "buy" else "down"
                        if chronos_dir not in (expected, "neutral"):
                            self.logger.info(
                                "%s: Chronos disagrees (expects %s, got %s) → hold",
                                sym,
                                expected,
                                chronos_dir,
                            )
                            continue

                rec = dict(ai)
                rec["confidence"] = combined
                base_strat = ai.get("strategy", "ai")
                enhanced = "+chronos" if use_chronos else ""
                rec["strategy"] = f"hybrid({base_strat}+sac{enhanced})"
                reason = ai.get("reasoning", "").strip()
                rec["reasoning"] = (
                    f"{reason} [SAC {d_conf:.0%}, regime={sym_regime}]"
                    if reason
                    else f"AI+SAC agree, SAC {d_conf:.0%}, regime={sym_regime}"
                )
                results.append(rec)
            else:
                self.logger.debug("%s: AI=%s vs SAC=%s → hold", sym, a_action, d_action)

        return results

    def _apply_market_context_filter(
        self, recs: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Корректирует confidence рекомендаций на основе рыночного контекста.

        Принимает либо flat dict (один контекст для всех), либо per-symbol map
        {"BTC/USDT": {...}, "ETH/USDT": {...}}.
        """
        min_conf = Config.MIN_SIGNAL_CONFIDENCE

        # Определяем, является ли context per-symbol map
        first_val = next(iter(context.values()), None) if context else None
        is_per_symbol = isinstance(first_val, dict)

        result = []
        for rec in recs:
            action = rec.get("action", "hold")
            conf = rec.get("confidence", 0.0)
            original_conf = conf
            symbol = rec.get("symbol", "?")

            if is_per_symbol:
                ctx = context.get(symbol) or {}
            else:
                ctx = context

            funding = ctx.get("funding_signal", "neutral")
            fng = ctx.get("fear_greed_signal", "neutral")
            oi = ctx.get("oi_signal", "oi_neutral")
            liquidation = ctx.get("liquidation_pressure", "neutral")
            basis = ctx.get("basis_signal", "neutral")
            google_trends = ctx.get("google_trends_signal", "neutral")
            pcr = ctx.get("pcr_signal", "neutral")
            ob = ctx.get("ob_signal", "balanced")
            iv = ctx.get("iv_signal", "neutral")
            macro = ctx.get("macro_signal", "neutral")
            etf = ctx.get("etf_signal", "neutral")
            reddit = ctx.get("reddit_signal", "neutral")
            stablecoin = ctx.get("stablecoin_signal", "neutral")

            if funding == "long_overheated":
                if action == "buy":
                    conf -= 0.15
                elif action == "sell":
                    conf = min(0.95, conf + 0.05)
            elif funding == "short_overheated":
                if action == "sell":
                    conf -= 0.15
                elif action == "buy":
                    conf = min(0.95, conf + 0.05)

            if fng == "extreme_greed" and action == "buy":
                conf -= 0.10
            elif fng == "extreme_fear" and action == "sell":
                conf -= 0.10

            if oi == "oi_bearish" and action == "buy":
                conf -= 0.08

            if liquidation == "long_liquidation" and action == "buy":
                conf -= 0.12
            elif liquidation == "short_squeeze" and action == "sell":
                conf -= 0.12

            if basis == "greed_premium" and action == "buy":
                conf -= 0.10
            elif basis == "backwardation" and action == "sell":
                conf -= 0.10

            if google_trends == "retail_fomo" and action == "buy":
                conf -= 0.08
            elif google_trends == "retail_absent" and action == "sell":
                conf -= 0.08

            if pcr == "greed_calls" and action == "buy":
                conf -= 0.10
            elif pcr == "fear_puts" and action == "sell":
                conf -= 0.10

            if ob == "ask_dominant" and action == "buy":
                conf -= 0.10
            elif ob == "bid_dominant" and action == "sell":
                conf -= 0.10

            if iv == "put_skew" and action == "buy":
                conf -= 0.08
            elif iv == "call_skew" and action == "sell":
                conf -= 0.08

            if macro == "macro_bearish" and action == "buy":
                conf -= 0.07

            if etf == "etf_outflow" and action == "buy":
                conf -= 0.08
            elif etf == "etf_inflow" and action == "sell":
                conf -= 0.08

            if reddit == "reddit_bearish" and action == "buy":
                conf -= 0.05

            if stablecoin == "stablecoin_outflow" and action == "buy":
                conf -= 0.06

            conf = round(conf, 3)
            if conf < min_conf:
                self.logger.info(
                    "Context filter dropped %s %s: conf %.3f→%.3f < %.2f "
                    "(funding=%s fng=%s oi=%s liq=%s basis=%s trends=%s pcr=%s "
                    "ob=%s iv=%s macro=%s etf=%s reddit=%s stable=%s)",
                    symbol,
                    action,
                    original_conf,
                    conf,
                    min_conf,
                    funding,
                    fng,
                    oi,
                    liquidation,
                    basis,
                    google_trends,
                    pcr,
                    ob,
                    iv,
                    macro,
                    etf,
                    reddit,
                    stablecoin,
                )
                continue

            if conf != original_conf:
                self.logger.debug(
                    "Context filter adjusted %s %s: conf %.3f→%.3f",
                    symbol,
                    action,
                    original_conf,
                    conf,
                )
            rec = dict(rec)
            rec["confidence"] = conf
            result.append(rec)

        return result

    def _apply_sentiment_filter(
        self, recs: List[Dict[str, Any]], sentiment: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Корректирует confidence на основе новостного сентимента.

        Очень негативный сентимент блокирует BUY-сигналы;
        очень позитивный немного усиливает их и ослабляет SELL.
        """
        min_conf = Config.MIN_SIGNAL_CONFIDENCE

        result = []
        for rec in recs:
            action = rec.get("action", "hold")
            conf = rec.get("confidence", 0.0)
            original_conf = conf
            symbol = rec.get("symbol", "?")

            score = sentiment.get(symbol, 0.0)

            if score < -0.6 and action == "buy":
                self.logger.info(
                    "Sentiment filter dropped %s buy: score=%.3f (very negative news)",
                    symbol,
                    score,
                )
                continue

            if score < -0.3 and action == "buy":
                conf -= 0.12
            elif score > 0.5 and action == "sell":
                conf -= 0.10

            if score > 0.7 and action == "buy":
                conf = min(0.95, conf + 0.05)

            conf = round(conf, 3)
            if conf < min_conf:
                self.logger.info(
                    "Sentiment filter dropped %s %s: conf %.3f→%.3f (score=%.3f)",
                    symbol,
                    action,
                    original_conf,
                    conf,
                    score,
                )
                continue

            if conf != original_conf:
                self.logger.debug(
                    "Sentiment filter adjusted %s %s: conf %.3f→%.3f (score=%.3f)",
                    symbol,
                    action,
                    original_conf,
                    conf,
                    score,
                )
            rec = dict(rec)
            rec["confidence"] = conf
            result.append(rec)

        return result

    def _generate_context_signals(
        self,
        snapshots: List[Dict[str, Any]],
        market_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Генерирует прямые BUY/SELL сигналы на основе экстремальных значений
        рыночного контекста, независимо от AI/стратегии.

        Принимает per-symbol map {"BTC/USDT": {...}, ...} или flat dict
        (применяется ко всем символам). Использует contrarian-логику для
        funding rate, PCR, Fear&Greed, basis, ETF flows и импульсную для
        ликвидаций и orderbook imbalance.
        """
        min_conf = Config.MIN_SIGNAL_CONFIDENCE

        # Строим lookup: symbol → snapshot для быстрого доступа
        snap_map: Dict[str, Dict[str, Any]] = {s["symbol"]: s for s in snapshots}

        # Определяем, является ли context per-symbol map
        first_val = (
            next(iter(market_context.values()), None) if market_context else None
        )
        is_per_symbol = isinstance(first_val, dict)

        results: List[Dict[str, Any]] = []

        for symbol, snap in snap_map.items():
            price = snap.get("price", 0.0)
            atr = snap.get("atr", price * 0.02)

            if not price or not atr:
                continue

            if is_per_symbol:
                ctx = market_context.get(symbol) or {}
            else:
                ctx = market_context

            if not ctx:
                continue

            funding_rate: float = ctx.get("funding_rate", 0.0)
            funding_signal: str = ctx.get("funding_signal", "neutral")
            liquidation_pressure: str = ctx.get("liquidation_pressure", "neutral")
            fear_greed: int = ctx.get("fear_greed", 50)
            basis_pct: float = ctx.get("basis_pct", 0.0)
            basis_signal: str = ctx.get("basis_signal", "neutral")
            ob_imbalance: float = ctx.get("ob_imbalance", 0.0)
            ob_signal: str = ctx.get("ob_signal", "balanced")
            pcr: float = ctx.get("pcr", 1.0)
            pcr_signal: str = ctx.get("pcr_signal", "neutral")
            etf_flow: float = ctx.get("etf_flow", 0.0)
            etf_signal: str = ctx.get("etf_signal", "neutral")

            # Список кандидатов: (action, confidence, reason)
            candidates: List[Tuple[str, float, str]] = []

            # ── Funding Rate (contrarian) ──────────────────────────────────
            if funding_signal == "long_overheated":
                conf = min(0.60 + abs(funding_rate) * 100, 0.88)
                reason = (
                    f"Funding overheated {funding_rate * 100:.3f}% → contrarian sell"
                )
                candidates.append(("sell", conf, reason))
            elif funding_signal == "short_overheated":
                conf = min(0.60 + abs(funding_rate) * 100, 0.85)
                reason = f"Funding negative {funding_rate * 100:.3f}% → contrarian buy"
                candidates.append(("buy", conf, reason))

            # ── Liquidation pressure (impulse) ────────────────────────────
            if liquidation_pressure == "long_liquidation":
                candidates.append(
                    (
                        "sell",
                        0.72,
                        "Long liquidation cascade detected → momentum sell",
                    )
                )
            elif liquidation_pressure == "short_squeeze":
                candidates.append(
                    (
                        "buy",
                        0.70,
                        "Short squeeze detected → momentum buy",
                    )
                )

            # ── Orderbook imbalance (short-term) ──────────────────────────
            if ob_signal == "bid_dominant" and ob_imbalance > 0.4:
                conf = 0.55 + ob_imbalance * 0.2
                reason = f"Orderbook bid dominant (imbalance={ob_imbalance:.2f})"
                candidates.append(("buy", conf, reason))
            elif ob_signal == "ask_dominant" and abs(ob_imbalance) > 0.4:
                conf = 0.55 + abs(ob_imbalance) * 0.2
                reason = f"Orderbook ask dominant (imbalance={ob_imbalance:.2f})"
                candidates.append(("sell", conf, reason))

            # ── Put/Call Ratio (institutional sentiment, contrarian) ───────
            if pcr_signal == "greed_calls":
                reason = f"Options PCR={pcr:.2f} (greed) → contrarian sell"
                candidates.append(("sell", 0.65, reason))
            elif pcr_signal == "fear_puts":
                reason = f"Options PCR={pcr:.2f} (fear) → contrarian buy"
                candidates.append(("buy", 0.63, reason))

            # ── Fear & Greed (only extreme panics/euphoria) ───────────────
            if fear_greed <= 10:
                reason = f"Fear&Greed={fear_greed} (extreme panic) → contrarian buy"
                candidates.append(("buy", 0.68, reason))
            elif fear_greed >= 92:
                reason = f"Fear&Greed={fear_greed} (extreme greed) → contrarian sell"
                candidates.append(("sell", 0.67, reason))

            # ── Basis premium (only strong premium > 2%) ──────────────────
            if basis_signal == "greed_premium" and basis_pct > 2.0:
                conf = 0.63 + min(basis_pct - 2.0, 1.0) * 0.05
                reason = (
                    f"Futures basis={basis_pct:.2f}% (greed premium) → contrarian sell"
                )
                candidates.append(("sell", conf, reason))

            # ── BTC ETF flows ──────────────────────────────────────────────
            base = symbol.split("/")[0]
            if base == "BTC":
                if etf_signal == "etf_outflow" and etf_flow < -100:
                    reason = f"BTC ETF outflow=${etf_flow:.0f}M → institutional selling"
                    candidates.append(("sell", 0.66, reason))
                elif etf_signal == "etf_inflow" and etf_flow > 150:
                    reason = f"BTC ETF inflow=${etf_flow:.0f}M → institutional buying"
                    candidates.append(("buy", 0.65, reason))

            # Генерируем сигналы из кандидатов
            for action, confidence, reason in candidates:
                confidence = round(confidence, 3)
                if confidence < min_conf:
                    continue
                sl, tp = self._sl_tp(price, atr, action)
                results.append(
                    {
                        "symbol": symbol,
                        "action": action,
                        "strategy": "context_signal",
                        "confidence": confidence,
                        "entry": price,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "reasoning": reason,
                    }
                )

        return results

    def _merge_with_context(
        self,
        existing: List[Dict[str, Any]],
        context_recs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Объединяет существующие сигналы с контекстными.

        Если в existing уже есть сигнал по тому же символу И тому же action —
        увеличивает confidence на 0.05 (не добавляет дубль).
        Иначе добавляет контекстный сигнал в список.
        Возвращает список отсортированный по confidence по убыванию.
        """
        # Строим индекс (symbol, action) → индекс в existing
        existing = [dict(r) for r in existing]
        index: Dict[Tuple[str, str], int] = {}
        for i, rec in enumerate(existing):
            key = (rec.get("symbol", ""), rec.get("action", ""))
            index[key] = i

        for ctx_rec in context_recs:
            key = (ctx_rec.get("symbol", ""), ctx_rec.get("action", ""))
            if key in index:
                # Усиливаем confidence существующего сигнала
                i = index[key]
                existing[i]["confidence"] = round(
                    min(0.95, existing[i]["confidence"] + 0.05), 3
                )
                self.logger.debug(
                    "Context boosted existing %s %s conf → %.3f",
                    key[0],
                    key[1],
                    existing[i]["confidence"],
                )
            else:
                existing.append(ctx_rec)
                index[key] = len(existing) - 1

        existing.sort(key=lambda r: r.get("confidence", 0.0), reverse=True)
        return existing
