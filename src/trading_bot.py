import asyncio
import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd

from config import Config
from src.ai_analyzer import AIAnalyzer
from src.bybit_api import BybitAPI
from src.data_loader import DataLoader
from src.market_impact import estimate_from_df as _ac_impact
from src.market_scanner import MarketScanner
from src.news_analyzer import NewsAnalyzer
from src.portfolio_manager import PortfolioManager
from src.portfolio_optimizer import PortfolioOptimizer
from src.redis_client import RedisClient
from src.regime_detector import RegimeDetector
from src.risk_management import RiskManager
from src.signal_combiner import SignalCombiner
from src.strategies import TradingStrategy
from src.telegram_notifier import TelegramNotifier
from src.trade_history import TradeHistory

logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s - %(name)s - " "%(levelname)s - %(message)s"),
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
        self.portfolio_manager = PortfolioManager(Config.INITIAL_BALANCE)
        self.strategy = None
        self.risk_manager = RiskManager(
            Config.INITIAL_BALANCE,
            Config.RISK_PER_TRADE,
        )
        self.scanner = MarketScanner(self.api, self.data_loader)
        self.news = NewsAnalyzer()
        self.ai = AIAnalyzer()
        self.combiner = SignalCombiner(self.ai)
        self.regime_detector = RegimeDetector()
        self._current_regime: str = "unknown"
        self.portfolio_optimizer = PortfolioOptimizer()
        self.is_running = False
        # symbol → {qty, stop_loss, take_profit, side, ...}
        self._monitored: dict = {}
        self._monitored_lock = asyncio.Lock()
        self._monitor_task: Optional[asyncio.Task] = None
        self.trade_history = TradeHistory()
        self.telegram = TelegramNotifier(
            Config.TELEGRAM_BOT_TOKEN,
            Config.TELEGRAM_CHAT_ID,
        )
        self._paper_balance: float = Config.INITIAL_BALANCE
        # symbol → last notified action ("buy"/"sell"/"hold")
        self._last_signals: dict = {}
        if Config.PAPER_TRADING:
            logger.warning("*** PAPER TRADING MODE — no real orders will be placed ***")

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
            self.strategy = TradingStrategy(Config.DEFAULT_STRATEGY)
            await self.strategy.initialize()
            await self._restore_state()
            await self.telegram.start()
            await self._fit_regime_detector()
            logger.info("Trading bot initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise

    async def _restore_state(self):
        state = self.redis.load_trading_state(Config.SYMBOL)
        if state:
            logger.info(f"Restored state: {state}")
        portfolio = self.redis.load_trading_state("portfolio_state")
        if portfolio:
            self.portfolio_manager.current_balance = portfolio.get(
                "balance", Config.INITIAL_BALANCE
            )
            self.portfolio_manager.positions = portfolio.get("positions", {})

    async def _fit_regime_detector(self) -> None:
        """Загружает исторические данные и обучает RegimeDetector."""
        try:
            df = await self.data_loader.load_ohlcv(
                Config.SYMBOL, Config.TIMEFRAME, limit=2000
            )
            if df is not None and not df.empty:
                self.regime_detector.fit(df)
                logger.info("RegimeDetector fitted on %d candles", len(df))
        except Exception as exc:
            logger.warning("RegimeDetector fit skipped: %s", exc)

    def _optimize_allocation(
        self,
        recs: list,
        market_data: dict,
    ) -> list:
        """
        Annotate buy recommendations with CVaR-optimal allocation fractions.

        When ≥2 buy signals exist, runs PortfolioOptimizer.allocate() on the
        joint returns matrix and adds "alloc_fraction" to each rec.
        Single-signal or sell signals keep Config.RISK_PER_TRADE.

        :param recs: Recommendations from SignalCombiner.
        :param market_data: {symbol: DataFrame} from the current cycle scan.
        :return: recs with "alloc_fraction" field populated.
        """
        buy_syms = [r["symbol"] for r in recs if r.get("action") == "buy"]

        if len(buy_syms) < 2:
            for r in recs:
                r.setdefault("alloc_fraction", Config.RISK_PER_TRADE)
            return recs

        returns_list = []
        valid: list[str] = []
        for sym in buy_syms:
            df = market_data.get(sym)
            if df is not None and len(df) >= 30:
                ret = df["close"].astype(float).pct_change().dropna().rename(sym)
                returns_list.append(ret)
                valid.append(sym)

        if len(valid) >= 2:
            returns_df = pd.concat(returns_list, axis=1).dropna()
            weights = self.portfolio_optimizer.allocate(valid, returns_df)
        else:
            weights = {}

        for r in recs:
            sym = r.get("symbol", "")
            if r.get("action") == "buy" and sym in weights:
                # Weight is portfolio share; scale to max RISK_PER_TRADE * 3
                r["alloc_fraction"] = min(weights[sym], Config.RISK_PER_TRADE * 3)
            else:
                r.setdefault("alloc_fraction", Config.RISK_PER_TRADE)
        return recs

    async def _get_balance_usdt(self) -> float:
        """Свободный баланс USDT (fallback — PortfolioManager)."""
        try:
            bal = await self.api.get_balance()
            if bal:
                return float(bal.get("free", {}).get("USDT", 0))
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
            *[self.news.get_sentiment(s) for s in symbols],
            return_exceptions=True,
        )
        snapshots = []
        for sym, result in zip(symbols, news_results):
            if isinstance(result, Exception):
                logger.warning(f"News sentiment failed for {sym}: " f"{result}")
                sent, headlines = 0.0, []
            else:
                sent, headlines = result
            df = market_data.get(sym)
            if df is None or df.empty:
                continue
            snap = self.scanner.build_snapshot(sym, df, sent, headlines)
            if snap:
                snapshots.append(snap)
        return snapshots

    def _filter_by_balance(self, recs: list, balance: float) -> list:
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
        print(f"  CYCLE #{cycle} | {ts} | " f"Balance: ${balance:.2f} USDT")
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
            print(f"  [{i}] {action:4s} {sym:<12s} " f"conf={conf:.0f}% strat={strat}")
            if entry:
                print(f"       entry={entry:.4f} " f"SL={sl:.4f} TP={tp:.4f}")
            if reasoning:
                print(f"       {reasoning}")
        print(sep)

    async def _notify_new_signals(self, recs: list, balance: float, cycle: int) -> None:
        """
        Отправляет в Telegram только новые или изменившиеся buy/sell сигналы.
        Hold и повторы одного и того же действия по символу не отправляются.
        """
        new_lines = []
        for r in recs:
            sym = r.get("symbol", "")
            action = r.get("action", "hold")
            if action not in ("buy", "sell"):
                continue
            if self._last_signals.get(sym) == action:
                continue  # уже отправляли этот сигнал

            icon = "🟢" if action == "buy" else "🔴"
            conf = r.get("confidence", 0) * 100
            strat = r.get("strategy", "?")
            entry = r.get("entry", 0)
            sl = r.get("stop_loss", 0)
            tp = r.get("take_profit", 0)
            regime = self._current_regime
            reasoning = r.get("reasoning", "")

            line = f"{icon} *{sym}* — {action.upper()}\n"
            line += f"   Conf: {conf:.0f}% | {strat} | Режим: {regime}\n"
            if entry:
                line += f"   Entry: ${entry:.4f} | SL: ${sl:.4f} | TP: ${tp:.4f}\n"
            if reasoning:
                line += f"   _{reasoning[:120]}_"
            new_lines.append(line)
            self._last_signals[sym] = action

        # Сбрасываем hold-символы из кэша (чтобы при следующем buy/sell уведомить)
        signaled = {r.get("symbol") for r in recs if r.get("action") in ("buy", "sell")}
        for sym in list(self._last_signals):
            if sym not in signaled:
                self._last_signals.pop(sym, None)

        if not new_lines:
            return

        ts = datetime.now().strftime("%H:%M:%S")
        header = f"📊 Цикл \\#{cycle} | {ts} | Баланс: ${balance:.2f}\n\n"
        text = header + "\n\n".join(new_lines)
        await self.telegram.notify(text)

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

        # Max positions check + duplicate guard (race-safe)
        async with self._monitored_lock:
            if len(self._monitored) >= Config.MAX_POSITIONS:
                logger.warning(
                    f"Max positions " f"({Config.MAX_POSITIONS}) " "reached. Skipping."
                )
                return
            if sym in self._monitored:
                logger.info(f"{sym} already monitored, skip")
                return

        # EV and win rate for this strategy
        strategy = top.get("strategy", Config.DEFAULT_STRATEGY)
        from src.trade_history import get_backtest_stats

        bt = get_backtest_stats(strategy)
        live_wr = await self.trade_history.get_win_rate(strategy, lookback=50)
        live_n = await self.trade_history.get_trade_count(strategy, lookback=50)
        live_ev = await self.trade_history.get_expected_value(strategy, lookback=50)

        # Telegram confirmation with full stats
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

        # Portfolio optimizer provides the allocation fraction (CVaR-optimal)
        alloc_fraction = top.get("alloc_fraction", Config.RISK_PER_TRADE)
        portfolio_qty = balance * alloc_fraction / entry

        # Kelly criterion caps the portfolio allocation when enough trade history
        stop_loss = top.get("stop_loss", 0.0)
        take_profit = top.get("take_profit", 0.0)
        if stop_loss and take_profit and live_n >= 10:
            kelly_qty = self.risk_manager.calculate_kelly_size(
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                win_rate=live_wr,
                current_balance=balance,
            )
            quantity = min(portfolio_qty, kelly_qty) if kelly_qty > 0 else portfolio_qty
            logger.info(
                "Sizing: portfolio=%.6f kelly=%.6f final=%.6f "
                "(win_rate=%.0f%% alloc=%.1f%% regime=%s)",
                portfolio_qty,
                kelly_qty,
                quantity,
                live_wr * 100,
                alloc_fraction * 100,
                self._current_regime,
            )
        else:
            # No trade history yet: cap at standard RISK_PER_TRADE (conservative)
            conservative_qty = balance * Config.RISK_PER_TRADE / entry
            quantity = min(portfolio_qty, conservative_qty)
            logger.info(
                "Sizing (early trades %d/10): conservative cap=%.6f final=%.6f",
                live_n,
                conservative_qty,
                quantity,
            )

        # AC model: adjust entry price for estimated market impact
        sym = top.get("symbol", Config.SYMBOL)
        main_df = market_data.get(sym)
        if main_df is not None and not main_df.empty:
            impact = _ac_impact(main_df, quantity * entry, Config.TIMEFRAME)
            action = top.get("action")
            if action == "buy":
                entry = entry * (1.0 + impact)
            elif action == "sell":
                entry = entry * (1.0 - impact)

        if quantity <= 0:
            return

        commission = quantity * entry * Config.COMMISSION_RATE

        sl_price = top.get("stop_loss", 0.0)
        tp_price = top.get("take_profit", 0.0)
        close_side = "sell" if action == "buy" else "buy"
        exchange_sl_id: str | None = None
        exchange_tp_id: str | None = None

        if Config.PAPER_TRADING:
            # Simulate: no real order
            logger.info(
                f"[PAPER] {action.upper()} " f"{quantity:.6f} {sym} @ {entry:.4f}"
            )
            if action == "buy":
                self._paper_balance -= quantity * entry + commission
            else:
                self._paper_balance += quantity * entry - commission
        else:
            order = await self.api.create_order(sym, "market", action, quantity)
            if not order:
                logger.error(f"Order failed for {sym}")
                return
            # Place SL/TP on exchange — protects position if bot crashes
            exchange_sl_id, exchange_tp_id = await self.api.place_exchange_sl_tp(
                sym, close_side, quantity, sl_price, tp_price
            )

        # Record in trade history
        trade_id = await self.trade_history.record_open(
            symbol=sym,
            strategy=strategy,
            action=action,
            entry_price=entry,
            quantity=quantity,
            confidence=top.get("confidence", 0.0),
            commission=commission,
        )

        # Register for SL/TP monitoring (under lock)
        async with self._monitored_lock:
            self._monitored[sym] = {
                "trade_id": trade_id,
                "qty": quantity,
                "entry": entry,
                "stop_loss": sl_price,
                "take_profit": tp_price,
                "side": action,
                "atr": top.get("atr", 0.0),
                "peak_price": entry,
                "exchange_sl_id": exchange_sl_id,
                "exchange_tp_id": exchange_tp_id,
            }

        await self.telegram.notify(
            f"Opened position *{sym}*\n"
            f"{action.upper()} {quantity:.6f} "
            f"@ ${entry:.4f}\n"
            f"SL: ${top.get('stop_loss', 0):.4f}  "
            f"TP: ${top.get('take_profit', 0):.4f}"
        )
        logger.info(f"Position opened: {sym} {action} " f"{quantity:.6f} @ {entry:.4f}")

    async def _monitor_positions(self) -> None:
        """
        Background task: checks SL/TP every 5 seconds.
        Executes market close when price hits SL or TP.
        Applies trailing stop-loss based on ATR.
        """
        _error_counts: dict = {}
        while self.is_running:
            async with self._monitored_lock:
                snapshot = list(self._monitored.items())
            for sym, pos in snapshot:
                try:
                    price = await self.api.get_current_price(sym)
                    if not price:
                        continue
                    sl = pos.get("stop_loss", 0)
                    tp = pos.get("take_profit", 0)
                    side = pos.get("side", "buy")
                    qty = pos.get("qty", 0)

                    # Trailing stop-loss
                    atr = pos.get("atr", 0.0)
                    mult = Config.TRAILING_STOP_ATR_MULT
                    if atr and atr > 0 and mult > 0:
                        if side == "buy":
                            trail = price - atr * mult
                            if trail > pos.get("stop_loss", 0):
                                async with self._monitored_lock:
                                    if sym in (self._monitored):
                                        self._monitored[sym]["stop_loss"] = trail
                                        pos["stop_loss"] = trail
                                sl = trail
                        else:
                            trail = price + atr * mult
                            if trail < pos.get("stop_loss", float("inf")):
                                async with self._monitored_lock:
                                    if sym in (self._monitored):
                                        self._monitored[sym]["stop_loss"] = trail
                                        pos["stop_loss"] = trail
                                sl = trail

                    triggered = False
                    if side == "buy":
                        if sl and price <= sl:
                            logger.warning(
                                f"SL hit {sym}: " f"price={price} <= sl={sl}"
                            )
                            triggered = True
                        elif tp and price >= tp:
                            logger.info(f"TP hit {sym}: " f"price={price} >= tp={tp}")
                            triggered = True
                    else:
                        if sl and price >= sl:
                            logger.warning(
                                f"SL hit {sym}: " f"price={price} >= sl={sl}"
                            )
                            triggered = True
                        elif tp and price <= tp:
                            logger.info(f"TP hit {sym}: " f"price={price} <= tp={tp}")
                            triggered = True

                    if triggered:
                        trade_id = pos.get("trade_id")
                        close_side = "sell" if side == "buy" else "buy"
                        if not Config.PAPER_TRADING:
                            await self.api.create_order(sym, "market", close_side, qty)
                            # Cancel exchange SL/TP orders to avoid double-execution
                            for oid in (
                                pos.get("exchange_sl_id"),
                                pos.get("exchange_tp_id"),
                            ):
                                if oid:
                                    await self.api.cancel_order(oid, sym)
                        await self.portfolio_manager.update_portfolio(
                            sym, close_side, qty, price
                        )
                        async with self._monitored_lock:
                            self._monitored.pop(sym, None)
                        logger.info(f"Position closed: {sym} " f"at {price}")
                        if trade_id:
                            await self.trade_history.record_close(
                                trade_id=trade_id,
                                exit_price=price,
                                commission=(qty * price * Config.COMMISSION_RATE),
                            )
                        stats = await self.trade_history.get_summary()
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
                    _error_counts.pop(sym, None)
                except Exception as e:
                    _error_counts[sym] = _error_counts.get(sym, 0) + 1
                    count = _error_counts[sym]
                    logger.error(f"Monitor error {sym} " f"(attempt {count}): {e}")
                    if count >= 5:
                        logger.warning(
                            f"Removing {sym} from monitor" " after 5 consecutive errors"
                        )
                        async with self._monitored_lock:
                            self._monitored.pop(sym, None)
                        _error_counts.pop(sym, None)
            await asyncio.sleep(5)

    async def _update_performance_stats(self):
        try:
            prices = {}
            for sym in self.portfolio_manager.positions:
                p = await self.api.get_current_price(sym)
                if p:
                    prices[sym] = p
            pv = await self.portfolio_manager.get_portfolio_value(prices)
            pnl = (pv - Config.INITIAL_BALANCE) / Config.INITIAL_BALANCE * 100
            self.redis.update_performance_stats(
                {
                    "timestamp": datetime.now().isoformat(),
                    "current_balance": (self.portfolio_manager.current_balance),
                    "portfolio_value": pv,
                    "profit_loss": pnl,
                    "positions": (self.portfolio_manager.get_positions()),
                }
            )
        except Exception as e:
            logger.error(f"Error updating performance stats: {e}")

    async def trading_loop(self):
        """
        Основной гибридный цикл (30 сек).

        Точный sleep: учитывает время выполнения итерации.
        При ошибке ждёт 10 сек и продолжает.
        """
        self.is_running = True
        self._monitor_task = asyncio.create_task(self._monitor_positions())
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
                symbols = await self.scanner.get_top_symbols(Config.SCAN_TOP_N)
                market_data = await self.scanner.scan_all(symbols, Config.TIMEFRAME)
                scanned = list(market_data.keys())
                snapshots = await self._collect_snapshots(scanned, market_data)

                balance = await self._get_balance_usdt()

                # Enforce daily loss limit before doing anything this cycle
                if not self.risk_manager.check_daily_loss_limit(balance):
                    await self.telegram.notify(
                        "⛔ Дневной лимит потерь достигнут. "
                        f"Баланс: ${balance:.2f}. Торговля остановлена до завтра."
                    )
                    logger.warning("Daily loss limit hit — stopping bot")
                    self.is_running = False
                    break

                # Detect regime per symbol; fallback to main symbol regime
                regimes: dict = {}
                for sym, df in market_data.items():
                    if df is not None and not df.empty:
                        regimes[sym] = self.regime_detector.predict(df)
                self._current_regime = regimes.get(Config.SYMBOL, "unknown")
                logger.info(
                    "Regimes: %s",
                    {s: r for s, r in regimes.items() if r != "unknown"},
                )

                recs = await self.combiner.combine(
                    snapshots,
                    balance,
                    regime=self._current_regime,
                    regimes=regimes,
                )
                recs = self._optimize_allocation(recs, market_data)

                if not recs:
                    logger.info(f"MODE={Config.MODE}, " "no signals — local fallback")
                    for snap in snapshots:
                        strat, conf = self.combiner.ai.recommend_strategy_local(snap)
                        if conf >= Config.MIN_SIGNAL_CONFIDENCE:
                            recs.append(
                                {
                                    "symbol": snap["symbol"],
                                    "action": "hold",
                                    "strategy": strat,
                                    "confidence": conf,
                                    "reasoning": "Local analysis",
                                }
                            )

                filtered = self._filter_by_balance(recs, balance)
                filtered.sort(
                    key=lambda x: x.get("confidence", 0),
                    reverse=True,
                )

                self._print_recommendations(filtered, balance, cycle)
                await self._notify_new_signals(filtered, balance, cycle)
                await self._execute_top_rec(filtered, market_data)
                await self._update_performance_stats()

                elapsed = loop.time() - t0
                sleep_for = max(0, Config.TRADING_INTERVAL - elapsed)
                logger.info(
                    f"Cycle #{cycle}: {elapsed:.1f}s, " f"sleep {sleep_for:.1f}s"
                )
                await asyncio.sleep(sleep_for)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(10)

    async def analyze_market(self, symbol: str, timeframe: str):
        """Анализ одного символа (обратная совместимость)."""
        try:
            data = await self.data_loader.get_market_data(symbol, timeframe, limit=100)
            data = self.data_loader.calculate_technical_indicators(data)
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
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        await self.telegram.stop()
        await self.api.close()
        await self.data_loader.close()
        self.redis.close()
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
