import asyncio
import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd

from config import Config
from src.ai_analyzer import AIAnalyzer
from src.bybit_api import BybitAPI
from src.signal_combiner import SignalCombiner
from src.data_loader import DataLoader
from src.market_scanner import MarketScanner
from src.news_analyzer import NewsAnalyzer
from src.portfolio_manager import PortfolioManager
from src.redis_client import RedisClient
from src.risk_management import RiskManager
from src.strategies import TradingStrategy

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s - %(name)s - "
        "%(levelname)s - %(message)s"
    ),
)
logger = logging.getLogger(__name__)


class TradingBot:
    """
    Гибридный бот: история + новости + AI.

    Цикл каждые TRADING_INTERVAL сек (по умолчанию 30):
    1. Топ-N монет (один fetch_tickers()).
    2. Параллельная загрузка OHLCV + индикаторов.
    3. Новостной сентимент (Redis-кэш 15 мин).
    4. Снэпшоты → Claude API (один batch-запрос).
    5. Fallback: локальная стратегия.
    6. Фильтр по балансу → вывод рекомендаций.
    7. Исполнение топ-1 при AUTO_EXECUTE=true.
    """

    def __init__(self):
        self.redis = RedisClient()
        self.api = BybitAPI()
        self.data_loader = DataLoader()
        self.portfolio_manager = PortfolioManager(
            Config.INITIAL_BALANCE
        )
        self.strategy = None
        self.risk_manager = RiskManager(
            Config.INITIAL_BALANCE,
            Config.RISK_PER_TRADE,
        )
        self.scanner = MarketScanner(
            self.api, self.data_loader
        )
        self.news = NewsAnalyzer()
        self.ai = AIAnalyzer()
        self.combiner = SignalCombiner(self.ai)
        self.is_running = False

    async def initialize(self):
        """
        Инициализирует API, стратегию, Redis-состояние.

        :raises Exception: Если инициализация не удалась.
        """
        try:
            await self.api.initialize(
                Config.BYBIT_API_KEY,
                Config.BYBIT_API_SECRET,
            )
            await self.data_loader.initialize(
                Config.BYBIT_API_KEY,
                Config.BYBIT_API_SECRET,
            )
            self.strategy = TradingStrategy(
                Config.DEFAULT_STRATEGY
            )
            await self.strategy.initialize()
            await self._restore_state()
            logger.info(
                "Trading bot initialized successfully"
            )
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise

    async def _restore_state(self):
        state = self.redis.load_trading_state(Config.SYMBOL)
        if state:
            logger.info(f"Restored state: {state}")
        portfolio = self.redis.load_trading_state(
            "portfolio_state"
        )
        if portfolio:
            self.portfolio_manager.current_balance = (
                portfolio.get("balance", Config.INITIAL_BALANCE)
            )
            self.portfolio_manager.positions = portfolio.get(
                "positions", {}
            )

    async def _get_balance_usdt(self) -> float:
        """Свободный баланс USDT (fallback — PortfolioManager)."""
        try:
            bal = await self.api.get_balance()
            if bal:
                return float(
                    bal.get("free", {}).get("USDT", 0)
                )
        except Exception as e:
            logger.warning(f"Balance fetch failed: {e}")
        return self.portfolio_manager.current_balance

    async def _collect_snapshots(
        self,
        symbols: List[str],
        market_data: dict,
    ) -> list:
        """
        Параллельно получает новости, строит снэпшоты.

        :param symbols: Символы с загруженными данными.
        :param market_data: {symbol: DataFrame}.
        :return: Список снэпшотов для AIAnalyzer.
        """
        news_results = await asyncio.gather(
            *[self.news.get_sentiment(s) for s in symbols]
        )
        snapshots = []
        for sym, (sent, headlines) in zip(
            symbols, news_results
        ):
            df = market_data.get(sym)
            if df is None or df.empty:
                continue
            snap = self.scanner.build_snapshot(
                sym, df, sent, headlines
            )
            if snap:
                snapshots.append(snap)
        return snapshots

    def _filter_by_balance(
        self, recs: list, balance: float
    ) -> list:
        """
        Оставляет рекомендации, доступные по балансу.

        Мин. лот = entry * 0.001.
        """
        result = []
        for r in recs:
            entry = r.get("entry", 0)
            if entry <= 0:
                result.append(r)
                continue
            if balance >= entry * 0.001:
                r["affordable"] = True
                result.append(r)
            else:
                logger.debug(
                    f"Skip {r['symbol']}: "
                    f"need ${entry * 0.001:.2f}, "
                    f"have ${balance:.2f}"
                )
        return result

    def _print_recommendations(
        self,
        recs: list,
        balance: float,
        cycle: int,
    ) -> None:
        sep = "=" * 60
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n{sep}")
        print(
            f"  CYCLE #{cycle} | {ts} | "
            f"Balance: ${balance:.2f} USDT"
        )
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
            print(
                f"  [{i}] {action:4s} {sym:<12s} "
                f"conf={conf:.0f}% strat={strat}"
            )
            if entry:
                print(
                    f"       entry={entry:.4f} "
                    f"SL={sl:.4f} TP={tp:.4f}"
                )
            if reasoning:
                print(f"       {reasoning}")
        print(sep)

    async def _execute_top_rec(
        self,
        recs: list,
        market_data: dict,
    ) -> None:
        """Исполняет топ-1 (только если AUTO_EXECUTE)."""
        if not recs or not Config.AUTO_EXECUTE:
            return
        top = recs[0]
        sym = top.get("symbol", Config.SYMBOL)
        df = market_data.get(sym, pd.DataFrame())
        signal = {
            "action": top.get("action", "hold"),
            "price": top.get("entry", 0),
            "confidence": top.get("confidence", 0),
            "strategy": top.get("strategy", ""),
        }
        await self._execute_trade(signal, df, sym)

    async def _execute_trade(
        self,
        signal: dict,
        market_data: pd.DataFrame,
        symbol: Optional[str] = None,
    ):
        """
        Проверяет сигнал и создаёт ордер через API.

        :param signal: {action, price, confidence}.
        :param market_data: DataFrame с OHLCV.
        :param symbol: Символ (по умолч. Config.SYMBOL).
        """
        if symbol is None:
            symbol = Config.SYMBOL
        try:
            ok = await self.risk_manager.validate_signal(
                signal, market_data
            )
            if not ok:
                logger.info("Signal validation failed")
                return

            entry = signal.get("price", 0)
            if entry <= 0 and not market_data.empty:
                entry = float(
                    market_data["close"].iloc[-1]
                )
            if entry <= 0:
                logger.warning(
                    f"Invalid entry_price for {symbol}"
                )
                return

            sl = await self.risk_manager.calculate_stop_loss(
                entry, signal
            )
            balance = await self.api.get_balance()
            if not balance:
                logger.error("Could not get balance")
                return

            cur_price = await self.api.get_current_price(
                symbol
            )
            if not cur_price:
                logger.error("Could not get price")
                return

            pos_size = (
                await self.risk_manager.calculate_position_size(
                    self.portfolio_manager.current_balance,
                    cur_price,
                    sl,
                )
            )
            if pos_size <= 0:
                logger.warning("Invalid position size")
                return

            side = (
                "buy" if signal["action"] == "buy" else "sell"
            )
            parts = (
                symbol.split("/")
                if "/" in symbol
                else [symbol[:-4], symbol[-4:]]
            )
            base = parts[0]
            quote = parts[1] if len(parts) > 1 else "USDT"
            free = balance.get("free", {})

            if side == "buy":
                cost = pos_size * entry
                if cost > free.get(quote, 0):
                    logger.warning(
                        f"Insufficient {quote}: "
                        f"need {cost:.2f}, "
                        f"have {free.get(quote, 0):.2f}"
                    )
                    return
            else:
                if free.get(base, 0) < pos_size:
                    logger.warning(
                        f"Insufficient {base} for sell"
                    )
                    return

            order = await self.api.create_order(
                symbol, "limit", side, pos_size, entry
            )
            if order:
                logger.info(f"Order executed: {order}")
                await self.portfolio_manager.update_portfolio(
                    symbol, side, pos_size, entry
                )
            else:
                logger.error("Failed to create order")

        except Exception as e:
            logger.error(f"Error executing trade: {e}")

    async def _update_performance_stats(self):
        try:
            prices = {}
            for sym in self.portfolio_manager.positions:
                p = await self.api.get_current_price(sym)
                if p:
                    prices[sym] = p
            pv = (
                await self.portfolio_manager
                .get_portfolio_value(prices)
            )
            pnl = (
                (pv - Config.INITIAL_BALANCE)
                / Config.INITIAL_BALANCE * 100
            )
            self.redis.update_performance_stats({
                "timestamp": datetime.now().isoformat(),
                "current_balance": (
                    self.portfolio_manager.current_balance
                ),
                "portfolio_value": pv,
                "profit_loss": pnl,
                "positions": (
                    self.portfolio_manager.get_positions()
                ),
            })
        except Exception as e:
            logger.error(
                f"Error updating performance stats: {e}"
            )

    async def trading_loop(self):
        """
        Основной гибридный цикл (30 сек).

        Точный sleep: учитывает время выполнения итерации.
        При ошибке ждёт 10 сек и продолжает.
        """
        self.is_running = True
        cycle = 0
        ai_status = "on" if self.ai.enabled else "off"
        logger.info(
            f"Starting hybrid loop "
            f"(interval={Config.TRADING_INTERVAL}s, "
            f"top={Config.SCAN_TOP_N}, ai={ai_status})"
        )

        while self.is_running:
            cycle += 1
            loop = asyncio.get_event_loop()
            t0 = loop.time()

            try:
                symbols = await self.scanner.get_top_symbols(
                    Config.SCAN_TOP_N
                )
                market_data = await self.scanner.scan_all(
                    symbols, Config.TIMEFRAME
                )
                scanned = list(market_data.keys())
                snapshots = await self._collect_snapshots(
                    scanned, market_data
                )

                balance = await self._get_balance_usdt()
                recs = await self.combiner.combine(
                    snapshots, balance
                )

                if not recs:
                    logger.info(
                        f"MODE={Config.MODE}, "
                        "no signals — local fallback"
                    )
                    for snap in snapshots:
                        strat, conf = (
                            self.combiner.ai
                            .recommend_strategy_local(snap)
                        )
                        if conf >= Config.MIN_SIGNAL_CONFIDENCE:
                            recs.append({
                                "symbol": snap["symbol"],
                                "action": "hold",
                                "strategy": strat,
                                "confidence": conf,
                                "reasoning": "Local analysis",
                            })

                filtered = self._filter_by_balance(
                    recs, balance
                )
                filtered.sort(
                    key=lambda x: x.get("confidence", 0),
                    reverse=True,
                )

                self._print_recommendations(
                    filtered, balance, cycle
                )
                await self._execute_top_rec(
                    filtered, market_data
                )
                await self._update_performance_stats()

                elapsed = loop.time() - t0
                sleep_for = max(
                    0, Config.TRADING_INTERVAL - elapsed
                )
                logger.info(
                    f"Cycle #{cycle}: {elapsed:.1f}s, "
                    f"sleep {sleep_for:.1f}s"
                )
                await asyncio.sleep(sleep_for)

            except Exception as e:
                logger.error(
                    f"Error in trading loop: {e}"
                )
                await asyncio.sleep(10)

    async def analyze_market(
        self, symbol: str, timeframe: str
    ):
        """Анализ одного символа (обратная совместимость)."""
        try:
            data = await self.data_loader.get_market_data(
                symbol, timeframe, limit=100
            )
            data = (
                self.data_loader
                .calculate_technical_indicators(data)
            )
            signal = await self.strategy.get_signal(data)
            logger.info(f"Signal for {symbol}: {signal}")
            return signal
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None

    async def stop(self):
        self.is_running = False
        await self.api.close()
        await self.data_loader.close()
        logger.info("Trading bot stopped")


async def main():
    """Запуск торгового бота."""
    bot = TradingBot()
    try:
        await bot.initialize()
        await bot.trading_loop()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
