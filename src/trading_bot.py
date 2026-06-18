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
from src.trade_history import TradeHistory, get_backtest_stats
from src.telegram_notifier import TelegramNotifier

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
        # symbol → {qty, stop_loss, take_profit, side, ...}
        self._monitored: dict = {}
        self._monitor_task: Optional[asyncio.Task] = None
        self.trade_history = TradeHistory()
        self.telegram = TelegramNotifier(
            Config.TELEGRAM_BOT_TOKEN,
            Config.TELEGRAM_CHAT_ID,
        )
        self._paper_balance: float = Config.INITIAL_BALANCE

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
            await self.telegram.start()
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
            strat_key = r.get("strategy")
            bt = get_backtest_stats(strat_key or "")
            live_wr = self.trade_history.get_win_rate(
                strat_key, lookback=50
            )
            live_ev = self.trade_history.get_expected_value(
                strat_key, lookback=50
            )
            live_n = self.trade_history.get_trade_count(
                strat_key, lookback=50
            )
            if bt["total_trades"] > 0:
                print(
                    f"       backtest : "
                    f"{bt['win_rate']:.0%} win  "
                    f"EV {bt['ev']*100:+.2f}%  "
                    f"({bt['total_trades']} сделок)"
                )
            else:
                print(
                    "       backtest : нет данных"
                    " (запустите backtest.py)"
                )
            if live_n > 0:
                print(
                    f"       live     : "
                    f"{live_wr:.0%} win  "
                    f"EV {live_ev*100:+.2f}%  "
                    f"({live_n} сделок)"
                )
            else:
                print(
                    "       live     : "
                    "-- (нет сделок пока)"
                )
        print(sep)

    async def _execute_top_rec(
        self,
        filtered: list,
        market_data: dict,
    ) -> None:
        """Исполняет топ-1 (только если AUTO_EXECUTE)."""
        if not filtered or not Config.AUTO_EXECUTE:
            return

        top = filtered[0]
        sym = top.get("symbol", Config.SYMBOL)
        action = top.get("action")

        if action not in ("buy", "sell"):
            return

        # Max positions check
        if len(self._monitored) >= Config.MAX_POSITIONS:
            logger.warning(
                f"Max positions ({Config.MAX_POSITIONS}) "
                "reached. Skipping."
            )
            return

        # Win rate: backtest (historical) + live (separate)
        strategy = top.get(
            "strategy", Config.DEFAULT_STRATEGY
        )
        bt = get_backtest_stats(strategy)
        live_wr = self.trade_history.get_win_rate(
            strategy, lookback=50
        )
        live_ev = self.trade_history.get_expected_value(
            strategy, lookback=50
        )
        live_n = self.trade_history.get_trade_count(
            strategy, lookback=50
        )

        # Telegram Variant B confirmation
        confirmed = await self.telegram.ask_confirm(
            top,
            live_win_rate=live_wr,
            live_trades=live_n,
            live_ev=live_ev,
            bt_win_rate=bt["win_rate"],
            bt_trades=bt["total_trades"],
            bt_ev=bt["ev"],
            timeout=Config.TELEGRAM_CONFIRM_TIMEOUT,
        )
        if not confirmed:
            logger.info(f"Trade rejected by user: {sym}")
            return

        entry = top.get("entry", 0.0)
        if not entry:
            return

        balance = await self._get_balance_usdt()
        quantity = (
            balance * Config.RISK_PER_TRADE / entry
        )
        if quantity <= 0:
            return

        commission = (
            quantity * entry * Config.COMMISSION_RATE
        )

        if Config.PAPER_TRADING:
            # Simulate: no real order
            logger.info(
                f"[PAPER] {action.upper()} "
                f"{quantity:.6f} {sym} @ {entry:.4f}"
            )
            if action == "buy":
                self._paper_balance -= (
                    quantity * entry + commission
                )
            else:
                self._paper_balance += (
                    quantity * entry - commission
                )
        else:
            order = await self.api.create_order(
                sym, "market", action, quantity
            )
            if not order:
                logger.error(
                    f"Order failed for {sym}"
                )
                return

        # Record in trade history
        trade_id = self.trade_history.record_open(
            symbol=sym,
            strategy=strategy,
            action=action,
            entry_price=entry,
            quantity=quantity,
            confidence=top.get("confidence", 0.0),
            commission=commission,
        )

        # Register for SL/TP monitoring
        self._monitored[sym] = {
            "trade_id": trade_id,
            "qty": quantity,
            "entry": entry,
            "stop_loss": top.get("stop_loss", 0.0),
            "take_profit": top.get("take_profit", 0.0),
            "side": action,
            "atr": top.get("atr", 0.0),
            "peak_price": entry,
        }

        await self.telegram.notify(
            f"Opened position *{sym}*\n"
            f"{action.upper()} {quantity:.6f} "
            f"@ ${entry:.4f}\n"
            f"SL: ${top.get('stop_loss', 0):.4f}  "
            f"TP: ${top.get('take_profit', 0):.4f}"
        )
        logger.info(
            f"Position opened: {sym} {action} "
            f"{quantity:.6f} @ {entry:.4f}"
        )

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

    async def _monitor_positions(self) -> None:
        """
        Background task: checks SL/TP every 5 seconds.
        Executes market close when price hits SL or TP.
        Applies trailing stop-loss based on ATR.
        """
        while self.is_running:
            for sym, pos in list(self._monitored.items()):
                try:
                    price = await self.api.get_current_price(
                        sym
                    )
                    if not price:
                        continue
                    sl = pos.get("stop_loss", 0)
                    tp = pos.get("take_profit", 0)
                    side = pos.get("side", "buy")
                    qty = pos.get("qty", 0)

                    # Trailing stop-loss: move SL toward price
                    atr = pos.get("atr", 0.0)
                    mult = Config.TRAILING_STOP_ATR_MULT
                    if atr and atr > 0 and mult > 0:
                        if side == "buy":
                            trail = price - atr * mult
                            if trail > pos.get(
                                "stop_loss", 0
                            ):
                                self._monitored[sym][
                                    "stop_loss"
                                ] = trail
                                pos["stop_loss"] = trail
                                sl = trail
                        else:
                            trail = price + atr * mult
                            if trail < pos.get(
                                "stop_loss", float("inf")
                            ):
                                self._monitored[sym][
                                    "stop_loss"
                                ] = trail
                                pos["stop_loss"] = trail
                                sl = trail

                    triggered = False
                    if side == "buy":
                        if sl and price <= sl:
                            logger.warning(
                                f"SL hit {sym}: "
                                f"price={price} <= sl={sl}"
                            )
                            triggered = True
                        elif tp and price >= tp:
                            logger.info(
                                f"TP hit {sym}: "
                                f"price={price} >= tp={tp}"
                            )
                            triggered = True
                    else:
                        if sl and price >= sl:
                            logger.warning(
                                f"SL hit {sym}: "
                                f"price={price} >= sl={sl}"
                            )
                            triggered = True
                        elif tp and price <= tp:
                            logger.info(
                                f"TP hit {sym}: "
                                f"price={price} <= tp={tp}"
                            )
                            triggered = True

                    if triggered:
                        trade_id = pos.get("trade_id")
                        close_side = (
                            "sell" if side == "buy" else "buy"
                        )
                        if not Config.PAPER_TRADING:
                            await self.api.create_order(
                                sym, "market",
                                close_side, qty
                            )
                        await (
                            self.portfolio_manager
                            .update_portfolio(
                                sym, close_side, qty, price
                            )
                        )
                        del self._monitored[sym]
                        logger.info(
                            f"Position closed: {sym} "
                            f"at {price}"
                        )
                        if trade_id:
                            self.trade_history.record_close(
                                trade_id=trade_id,
                                exit_price=price,
                                commission=(
                                    qty
                                    * price
                                    * Config.COMMISSION_RATE
                                ),
                            )
                        stats = (
                            self.trade_history.get_summary()
                        )
                        await self.telegram.notify(
                            f"Closed position *{sym}* "
                            f"@ ${price:.4f}\n"
                            f"Total trades: "
                            f"{stats['closed_trades']}  "
                            f"Win Rate: "
                            f"{stats['win_rate']:.0%}\n"
                            f"Total PnL: "
                            f"${stats['total_pnl']:+.2f}"
                        )
                except Exception as e:
                    logger.error(
                        f"Monitor error for {sym}: {e}"
                    )
            await asyncio.sleep(5)

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
        self._monitor_task = asyncio.create_task(
            self._monitor_positions()
        )
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
        if self._monitor_task:
            self._monitor_task.cancel()
        await self.telegram.stop()
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
