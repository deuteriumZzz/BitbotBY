"""
Слой определения размера позиции и исполнения ордеров.

Вынесен из TradingBot._execute_top_rec чтобы отделить задачу «сколько
купить/продать и разместить ордер» от оркестрации.

Стек сайзинга (применяется по порядку):
  1. CVaR-оптимальная alloc_fraction от PortfolioOptimizer (задаётся upstream).
  2. Half-Kelly cap при ≥10 живых сделках.
  3. Консервативный RISK_PER_TRADE cap при <10 живых сделках.
  4. Поправка на рыночный импакт Almgren-Chriss.
  5. Округление до лота биржи через api.round_quantity.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Callable, Dict, Optional

import pandas as pd

from config import Config
from src.constants import KELLY_MIN_TRADES, TRADE_HISTORY_LOOKBACK
from src.market_impact import estimate_from_df as _ac_impact
from src.types import PositionRecord

if TYPE_CHECKING:
    from src.bybit_api import BybitAPI
    from src.correlation_filter import CorrelationFilter
    from src.portfolio_optimizer import PortfolioOptimizer
    from src.risk_management import RiskManager
    from src.telegram_notifier import TelegramNotifier
    from src.trade_history import TradeHistory

logger = logging.getLogger(__name__)


class OrderExecutor:
    """Определяет размер позиции и размещает рыночный ордер для топ-рекомендации."""

    def __init__(
        self,
        api: "BybitAPI",
        trade_history: "TradeHistory",
        telegram: "TelegramNotifier",
        risk_manager: "RiskManager",
        portfolio_optimizer: "PortfolioOptimizer",
        corr_filter: "CorrelationFilter",
        get_monitored: Callable[[], Dict[str, PositionRecord]],
        get_lock: Callable[[], asyncio.Lock],
        get_paper_balance: Callable[[], float],
        set_paper_balance: Callable[[float], None],
        set_last_trade_at: Callable[[float], None],
        get_current_regime: Callable[[], str],
    ) -> None:
        self._api = api
        self._trade_history = trade_history
        self._telegram = telegram
        self._risk_manager = risk_manager
        self._portfolio_optimizer = portfolio_optimizer
        self._corr_filter = corr_filter
        # Getters instead of direct references so that tests or runtime code that
        # replace bot._monitored (e.g. bot._monitored = {}) don't leave the executor
        # holding a stale reference to the old dict object.
        self._get_monitored = get_monitored
        self._get_lock = get_lock
        self._get_paper_balance = get_paper_balance
        self._set_paper_balance = set_paper_balance
        self._set_last_trade_at = set_last_trade_at
        self._get_current_regime = get_current_regime

    async def execute(
        self,
        top: dict,
        market_data: Dict[str, pd.DataFrame],
        balance: float,
    ) -> None:
        """
        Определяет размер позиции и размещает рыночный ордер для рекомендации.

        :param top: Рекомендация с ключами symbol, action, entry, stop_loss,
            take_profit.
        :param market_data: Словарь {symbol: DataFrame} для расчёта импакта.
        :param balance: Текущий баланс USDT.
        """
        # Resolve current dict/lock at call time so reassignments (e.g. in tests:
        # bot._monitored = {...}) are always visible.
        monitored = self._get_monitored()
        lock = self._get_lock()

        sym = top.get("symbol", Config.SYMBOL)
        action = top.get("action")
        if action not in ("buy", "sell"):
            return

        # Guard: max positions and duplicate — reserve the slot immediately under
        # lock so a second concurrent cycle can't open the same symbol while
        # Telegram confirmation is being awaited (up to TELEGRAM_CONFIRM_TIMEOUT s).
        async with lock:
            if len(monitored) >= Config.MAX_POSITIONS:
                logger.warning(
                    "Max positions (%d) reached, skipping %s",
                    Config.MAX_POSITIONS,
                    sym,
                )
                return
            if sym in monitored:
                logger.info("%s already monitored, skipping duplicate", sym)
                return
            # Placeholder: claim the slot before releasing lock.
            # Cleared in the finally-block if anything goes wrong before
            # the real PositionRecord is written.
            monitored[sym] = None  # type: ignore[assignment]
            open_syms = [s for s in monitored if s != sym]

        position_opened = False
        try:
            # Correlation guard: block positions too correlated with existing ones.
            if Config.MAX_CORRELATION > 0 and not self._corr_filter.is_allowed(
                sym, open_syms
            ):
                max_corr = self._corr_filter.max_correlation(sym, open_syms)
                logger.warning(
                    "Skipping %s: correlation %.2f >= MAX_CORRELATION %.2f",
                    sym,
                    max_corr,
                    Config.MAX_CORRELATION,
                )
                return

            strategy = top.get("strategy", Config.DEFAULT_STRATEGY)
            from src.trade_history import get_backtest_stats  # noqa: PLC0415

            bt = get_backtest_stats(strategy)
            live_wr = await self._trade_history.get_win_rate(
                strategy, lookback=TRADE_HISTORY_LOOKBACK
            )
            live_n = await self._trade_history.get_trade_count(
                strategy, lookback=TRADE_HISTORY_LOOKBACK
            )
            live_ev = await self._trade_history.get_expected_value(
                strategy, lookback=TRADE_HISTORY_LOOKBACK
            )

            confirmed = await self._telegram.ask_confirm(
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
                logger.info("Trade rejected via Telegram: %s", sym)
                return

            entry = top.get("entry", 0.0)
            if not entry:
                return

            sl_price = top.get("stop_loss", 0.0)
            tp_price = top.get("take_profit", 0.0)

            # Validate SL/TP direction to catch bad AI output before order.
            if sl_price and tp_price:
                if action == "buy" and (sl_price >= entry or tp_price <= entry):
                    logger.error(
                        "Invalid SL/TP for buy %s: entry=%.4f sl=%.4f tp=%.4f"
                        " — skipping",
                        sym, entry, sl_price, tp_price,
                    )
                    return
                if action == "sell" and (sl_price <= entry or tp_price >= entry):
                    logger.error(
                        "Invalid SL/TP for sell %s: entry=%.4f sl=%.4f tp=%.4f"
                        " — skipping",
                        sym, entry, sl_price, tp_price,
                    )
                    return

            quantity = self._size_position(top, balance, live_wr, live_n, entry)

            # Almgren-Chriss market impact: shift effective entry price for large
            # orders relative to ADV.
            df = market_data.get(sym)
            if df is None:
                logger.debug(
                    "No market data for %s — skipping Almgren-Chriss adjustment",
                    sym,
                )
            elif not df.empty:
                impact = _ac_impact(df, quantity * entry, Config.TIMEFRAME)
                entry = (
                    entry * (1.0 + impact)
                    if action == "buy"
                    else entry * (1.0 - impact)
                )

            quantity = self._api.round_quantity(sym, quantity)
            if quantity <= 0:
                return

            # Default commission — used in paper trading and as fallback when the
            # exchange response doesn't include fee info.
            commission = quantity * entry * Config.COMMISSION_RATE
            exchange_sl_id: Optional[str] = None
            exchange_tp_id: Optional[str] = None

            if Config.PAPER_TRADING:
                logger.info(
                    "[PAPER] %s %.6f %s @ %.4f", action.upper(), quantity, sym, entry
                )
                bal = self._get_paper_balance()
                if action == "buy":
                    self._set_paper_balance(bal - quantity * entry - commission)
                else:
                    self._set_paper_balance(bal + quantity * entry - commission)
            else:
                order = await self._api.create_order(sym, "market", action, quantity)
                if not order:
                    logger.error("Order creation failed for %s", sym)
                    return
                # Use actual exchange fee when available — varies by user tier
                # (VIP discounts) and order type (maker < taker).
                fee_cost = float((order.get("fee") or {}).get("cost") or 0)
                if fee_cost > 0:
                    commission = fee_cost
                close_side = "sell" if action == "buy" else "buy"
                exchange_sl_id, exchange_tp_id = await self._api.place_exchange_sl_tp(
                    sym, close_side, quantity, sl_price, tp_price
                )

            trade_id = await self._trade_history.record_open(
                symbol=sym,
                strategy=strategy,
                action=action,
                entry_price=entry,
                quantity=quantity,
                confidence=top.get("confidence", 0.0),
                commission=commission,
            )

            pos: PositionRecord = {
                "trade_id": trade_id,
                "qty": quantity,
                "entry": entry,
                "stop_loss": sl_price,
                "take_profit": tp_price,
                "side": action,
                "atr": top.get("atr", 0.0),
                "snap": top.get("_snap"),
                "balance_at_entry": balance,
                "peak_price": entry,
                "exchange_sl_id": exchange_sl_id,
                "exchange_tp_id": exchange_tp_id,
            }
            async with lock:
                monitored[sym] = pos
            position_opened = True

            self._set_last_trade_at(time.time())
            await self._telegram.notify(
                f"Opened *{sym}*\n"
                f"{action.upper()} {quantity:.6f} @ ${entry:.4f}\n"
                f"SL: ${sl_price:.4f}  TP: ${tp_price:.4f}"
            )
            logger.info(
                "Position opened: %s %s %.6f @ %.4f", sym, action, quantity, entry
            )

        finally:
            # If we didn't successfully open a position, release the placeholder
            # so the slot isn't permanently blocked.
            if not position_opened:
                async with lock:
                    if monitored.get(sym) is None:
                        monitored.pop(sym, None)

    # ── Sizing helpers ────────────────────────────────────────────────────────

    def _size_position(
        self,
        top: dict,
        balance: float,
        live_wr: float,
        live_n: int,
        entry: float,
    ) -> float:
        """
        Возвращает размер позиции в единицах базового актива.

        При ≥KELLY_MIN_TRADES использует Half-Kelly с ограничением portfolio_qty.
        При <KELLY_MIN_TRADES использует консервативный фиксированный фракционный кэп.

        :param top: Рекомендация с alloc_fraction, stop_loss, take_profit.
        :param balance: Текущий баланс USDT.
        :param live_wr: Win rate по живым сделкам.
        :param live_n: Количество живых сделок.
        :param entry: Цена входа.
        :return: Размер позиции в единицах базового актива.
        """
        alloc_fraction = top.get("alloc_fraction", Config.RISK_PER_TRADE)
        portfolio_qty = balance * alloc_fraction / entry

        stop_loss = top.get("stop_loss", 0.0)
        take_profit = top.get("take_profit", 0.0)

        if stop_loss and take_profit and live_n >= KELLY_MIN_TRADES:
            # Half-Kelly: full Kelly maximises geometric growth but is very
            # sensitive to win-rate estimation error; half-Kelly halves
            # drawdown variance at only ~25% expected-growth cost.
            kelly_qty = self._risk_manager.calculate_kelly_size(
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
                self._get_current_regime(),
            )
        else:
            # Fewer than 10 trades: Kelly estimate unreliable, fall back to
            # a conservative fixed fraction to limit exposure during warm-up.
            conservative_qty = balance * Config.RISK_PER_TRADE / entry
            quantity = min(portfolio_qty, conservative_qty)
            logger.info(
                "Sizing (early trades %d/10): conservative cap=%.6f final=%.6f",
                live_n,
                conservative_qty,
                quantity,
            )

        return quantity
