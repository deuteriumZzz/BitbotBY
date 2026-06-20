"""
Вспомогательные функции одного торгового цикла.

CycleRunner объединяет: сбор снэпшотов с новостным сентиментом,
аннотацию аллокаций CVaR-оптимизатором, Telegram-уведомления
о новых сигналах и вывод отчёта в stdout.
Вынесен из TradingBot для независимого тестирования.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Dict, List

import pandas as pd

from config import Config

if TYPE_CHECKING:
    from src.market_scanner import MarketScanner
    from src.news_analyzer import NewsAnalyzer
    from src.portfolio_optimizer import PortfolioOptimizer
    from src.telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)


class CycleRunner:
    """
    Объединяет вспомогательные задачи одного торгового цикла.

    Отвечает за: сбор снэпшотов с параллельным получением новостного
    сентимента, аннотацию CVaR-аллокаций, дедуплицированные Telegram-
    уведомления и форматированный вывод рекомендаций в stdout.
    """

    def __init__(
        self,
        news: "NewsAnalyzer",
        scanner: "MarketScanner",
        portfolio_optimizer: "PortfolioOptimizer",
        telegram: "TelegramNotifier",
        get_current_regime: Callable[[], str],
    ) -> None:
        self._news = news
        self._scanner = scanner
        self._optimizer = portfolio_optimizer
        self._telegram = telegram
        self._get_current_regime = get_current_regime
        # symbol → last notified action; prevents re-sending identical signals
        self._last_signals: Dict[str, str] = {}

    # ── Snapshot collection ───────────────────────────────────────────────────

    async def collect_snapshots(
        self,
        symbols: List[str],
        market_data: Dict[str, pd.DataFrame],
    ) -> list:
        """
        Параллельно получает новостной сентимент и строит снэпшоты для AI.

        :param symbols: Символы с загруженными OHLCV данными в этом цикле.
        :param market_data: Словарь {symbol: DataFrame} из текущего скана.
        :return: Список снэпшотов готовых для AIAnalyzer.combine().
        """
        news_results = await asyncio.gather(
            *[self._news.get_sentiment(s) for s in symbols],
            return_exceptions=True,
        )
        snapshots = []
        for sym, result in zip(symbols, news_results):
            if isinstance(result, BaseException):
                logger.warning("News sentiment failed for %s: %s", sym, result)
                sent: float = 0.0
                headlines: List[str] = []
            else:
                sent, headlines = result
            df = market_data.get(sym)
            if df is None or df.empty:
                continue
            snap = self._scanner.build_snapshot(sym, df, sent, headlines)
            if snap:
                snapshots.append(snap)
        return snapshots

    # ── Allocation annotation ─────────────────────────────────────────────────

    def optimize_allocation(
        self,
        recs: list,
        market_data: Dict[str, pd.DataFrame],
    ) -> list:
        """
        Аннотирует buy-рекомендации CVaR-оптимальными долями аллокации.

        Когда ≥2 buy-сигналов, запускает PortfolioOptimizer.allocate()
        на матрице доходностей и добавляет "alloc_fraction" в каждую рек.
        Одиночные и sell-сигналы получают Config.RISK_PER_TRADE по умолчанию.

        :param recs: Список рекомендаций от SignalCombiner.
        :param market_data: Словарь {symbol: DataFrame} из текущего скана.
        :return: Рекомендации с добавленным полем "alloc_fraction".
        """
        buy_syms = [r["symbol"] for r in recs if r.get("action") == "buy"]

        if len(buy_syms) < 2:
            for r in recs:
                r.setdefault("alloc_fraction", Config.RISK_PER_TRADE)
            return recs

        returns_list = []
        valid: List[str] = []
        for sym in buy_syms:
            df = market_data.get(sym)
            if df is not None and len(df) >= 30:
                ret = df["close"].astype(float).pct_change().dropna().rename(sym)
                returns_list.append(ret)
                valid.append(sym)

        if len(valid) >= 2:
            returns_df = pd.concat(returns_list, axis=1).dropna()
            weights = self._optimizer.allocate(valid, returns_df)
        else:
            weights = {}

        for r in recs:
            sym = r.get("symbol", "")
            if r.get("action") == "buy" and sym in weights:
                # Cap at RISK_PER_TRADE * 3 so a single optimizer weight
                # can't dwarf the position-size guard in OrderExecutor.
                r["alloc_fraction"] = min(weights[sym], Config.RISK_PER_TRADE * 3)
            else:
                r.setdefault("alloc_fraction", Config.RISK_PER_TRADE)
        return recs

    # ── Telegram notifications ────────────────────────────────────────────────

    async def notify_new_signals(self, recs: list, balance: float, cycle: int) -> None:
        """
        Отправляет в Telegram только новые или изменившиеся buy/sell сигналы.

        Дедуплицирует: повторные сигналы того же действия по тому же символу
        не отправляются. Hold-сигналы никогда не отправляются. Символы,
        выбывшие из recs, удаляются из кэша дедупликации.

        :param recs: Список рекомендаций текущего цикла.
        :param balance: Текущий баланс USDT.
        :param cycle: Номер цикла для заголовка сообщения.
        """
        new_lines = []
        for r in recs:
            sym = r.get("symbol", "")
            action = r.get("action", "hold")
            if action not in ("buy", "sell"):
                continue
            if self._last_signals.get(sym) == action:
                continue

            icon = "🟢" if action == "buy" else "🔴"
            conf = r.get("confidence", 0) * 100
            strat = self.md_escape(str(r.get("strategy", "?")))
            entry = r.get("entry", 0)
            sl = r.get("stop_loss", 0)
            tp = r.get("take_profit", 0)
            regime = self.md_escape(str(self._get_current_regime()))
            reasoning = self.md_escape(str(r.get("reasoning", ""))[:120])

            line = f"{icon} *{sym}* — {action.upper()}\n"
            line += f"   Conf: {conf:.0f}% | {strat} | Режим: {regime}\n"
            if entry:
                line += f"   Entry: ${entry:.4f} | SL: ${sl:.4f} | TP: ${tp:.4f}\n"
            if reasoning:
                line += f"   {reasoning}"
            new_lines.append(line)
            self._last_signals[sym] = action

        # Evict hold symbols so the next buy/sell will be sent fresh
        signaled = {r.get("symbol") for r in recs if r.get("action") in ("buy", "sell")}
        for sym in list(self._last_signals):
            if sym not in signaled:
                self._last_signals.pop(sym, None)

        if not new_lines:
            return

        ts = datetime.now().strftime("%H:%M:%S")
        header = f"📊 Цикл \\#{cycle} | {ts} | Баланс: ${balance:.2f}\n\n"
        await self._telegram.notify(header + "\n\n".join(new_lines))

    # ── Console output ────────────────────────────────────────────────────────

    @staticmethod
    def print_recommendations(recs: list, balance: float, cycle: int) -> None:
        """
        Выводит форматированную таблицу рекомендаций в stdout.

        :param recs: Список рекомендаций.
        :param balance: Текущий баланс USDT.
        :param cycle: Номер цикла.
        """
        sep = "=" * 60
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n{sep}")
        print(f"  CYCLE #{cycle} | {ts} | Balance: ${balance:.2f} USDT")
        print(sep)
        if not recs:
            print("  No actionable signals this cycle.")
            print(sep)
            return
        for i, r in enumerate(recs, 1):
            action = r.get("action", "?").upper()
            sym = r.get("symbol", "?")
            conf = r.get("confidence", 0) * 100
            strat = r.get("strategy", "?")
            entry = r.get("entry", 0)
            sl = r.get("stop_loss", 0)
            tp = r.get("take_profit", 0)
            reasoning = r.get("reasoning", "")
            print(f"  [{i}] {action:4s} {sym:<12s} conf={conf:.0f}% strat={strat}")
            if entry:
                print(f"       entry={entry:.4f} SL={sl:.4f} TP={tp:.4f}")
            if reasoning:
                print(f"       {reasoning}")
        print(sep)

    @staticmethod
    def md_escape(text: str) -> str:
        """
        Экранирует спецсимволы Telegram Markdown v1 в произвольном тексте.

        :param text: Исходный текст.
        :return: Текст с экранированными символами _, *, `, [.
        """
        return (
            text.replace("_", "\\_")
            .replace("*", "\\*")
            .replace("`", "\\`")
            .replace("[", "\\[")
        )
