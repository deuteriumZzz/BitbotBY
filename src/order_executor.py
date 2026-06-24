"""
Слой определения размера позиции и исполнения ордеров.

Вынесен из TradingBot._execute_top_rec чтобы отделить задачу «сколько
купить/продать и разместить ордер» от оркестрации.

Стек сайзинга (применяется по порядку):
  1. CVaR-оптимальная alloc_fraction от PortfolioOptimizer (задаётся upstream).
  2. Half-Kelly cap при ≥10 живых сделках.
  3. Консервативный RISK_PER_TRADE cap при <10 живых сделках.
  4. Поправка на рыночный импакт Almgren-Chriss.
  5. Масштабирование при просадке от пика (УЛУЧШЕНИЕ 6).
  6. Округление до лота биржи через api.round_quantity.

УЛУЧШЕНИЕ 4: фильтр по времени суток (TRADING_HOURS UTC).
УЛУЧШЕНИЕ 5: фильтр по ликвидности (спред и 24h объём).
УЛУЧШЕНИЕ 6: уменьшение размера позиции при просадке от пика баланса.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

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


_REGIME_MULT = {"uptrend": 1.0, "sideways": 0.7, "downtrend": 0.5}


def _calc_dynamic_leverage(
    top: dict,
    entry: float,
    balance: float = 0.0,
    peak_balance: float = 0.0,
    runtime_config: "Any | None" = None,
) -> int:
    """
    Вычисляет плечо в зависимости от выбранного режима (LEVERAGE_MODE):

    fixed      — фиксированное Config.LEVERAGE на все монеты
    volatility — target_risk / (ATR / price)  [Вариант 2]
    full       — volatility × regime_mult × drawdown_mult  [Вариант 3]
    """
    try:
        lev_min = int(getattr(Config, "LEVERAGE_MIN", 1))
        lev_max = int(getattr(Config, "LEVERAGE_MAX", 5))
        fallback = max(lev_min, min(lev_max, int(getattr(Config, "LEVERAGE", 3))))

        # Режим: из RuntimeConfig (Telegram) или из .env
        mode = "volatility"
        target_risk = float(getattr(Config, "LEVERAGE_TARGET_RISK", 0.01))
        if runtime_config is not None:
            try:
                mode = runtime_config.get_leverage_mode()
                target_risk = runtime_config.get_leverage_target_risk()
            except Exception:
                pass
        else:
            mode = getattr(Config, "LEVERAGE_MODE", "volatility")

        if mode == "fixed":
            return fallback

        atr = float(top.get("atr", 0.0))
        price = entry if entry > 0 else float(top.get("price", 0.0))
        if atr <= 0 or price <= 0:
            return fallback

        atr_pct = atr / price
        base_lev = target_risk / atr_pct

        if mode == "full":
            # Режим рынка из индикаторов снэпшота
            trend = top.get("indicators", {}).get("trend", "sideways")
            regime_mult = _REGIME_MULT.get(trend, 0.7)

            # Просадка относительно пика баланса
            if peak_balance > 0 and balance > 0 and peak_balance > balance:
                dd_pct = (peak_balance - balance) / peak_balance
                try:
                    max_dd = (
                        runtime_config.get_max_drawdown_percent()
                        if runtime_config is not None
                        else float(getattr(Config, "MAX_DRAWDOWN_PERCENT", 0.15))
                    )
                except Exception:
                    max_dd = float(getattr(Config, "MAX_DRAWDOWN_PERCENT", 0.15))
                drawdown_mult = max(0.3, 1.0 - dd_pct / max(max_dd, 0.01))
            else:
                drawdown_mult = 1.0

            base_lev = base_lev * regime_mult * drawdown_mult

        lev = int(max(lev_min, min(lev_max, round(base_lev))))
        logger.debug(
            "Leverage [%s] %s: ATR=%.2f%% → %dx",
            mode,
            top.get("symbol", "?"),
            atr_pct * 100,
            lev,
        )
        return lev
    except Exception:
        return max(1, int(getattr(Config, "LEVERAGE", 3)))


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
        runtime_config: "Any | None" = None,
    ) -> None:
        self._api = api
        self._trade_history = trade_history
        self._telegram = telegram
        self._risk_manager = risk_manager
        self._portfolio_optimizer = portfolio_optimizer
        self._corr_filter = corr_filter
        self._runtime_config = runtime_config
        # Getters instead of direct references so that tests or runtime code that
        # replace bot._monitored (e.g. bot._monitored = {}) don't leave the executor
        # holding a stale reference to the old dict object.
        self._get_monitored = get_monitored
        self._get_lock = get_lock
        self._get_paper_balance = get_paper_balance
        self._set_paper_balance = set_paper_balance
        self._set_last_trade_at = set_last_trade_at
        self._get_current_regime = get_current_regime
        # УЛУЧШЕНИЕ 6: отслеживание пика баланса для масштабирования при просадке
        self._peak_balance: float = 0.0

    def configure_risk(
        self,
        max_positions: int | None = None,
        risk_per_trade: float | None = None,
        drawdown_scale_enabled: bool | None = None,
    ) -> None:
        """Обновляет параметры риска из RuntimeConfig во время работы."""
        if max_positions is not None:
            self._risk_manager.max_positions = max_positions
        if risk_per_trade is not None:
            self._risk_manager.risk_per_trade = risk_per_trade
        if drawdown_scale_enabled is not None:
            self._risk_manager.drawdown_scale_enabled = drawdown_scale_enabled

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

        # УЛУЧШЕНИЕ 4: фильтр по времени суток (UTC)
        _rc = getattr(self, "_runtime_config", None)
        _trading_hours = (
            _rc.get_trading_hours() if _rc is not None else Config.TRADING_HOURS
        )
        if _trading_hours:
            if _rc is not None:
                if not _rc.is_trading_time():
                    logger.info(
                        "Outside trading hours (%s UTC), skipping %s",
                        _trading_hours,
                        sym,
                    )
                    return
            else:
                hour = datetime.datetime.utcnow().hour
                try:
                    start_h, end_h = map(int, _trading_hours.split("-"))
                    in_hours = (
                        (start_h <= hour < end_h)
                        if start_h < end_h
                        else (hour >= start_h or hour < end_h)
                    )
                    if not in_hours:
                        logger.info(
                            "Outside trading hours (%s UTC), skipping %s",
                            _trading_hours,
                            sym,
                        )
                        return
                except ValueError:
                    pass

        # Guard: max positions and duplicate — reserve the slot immediately under
        # lock so a second concurrent cycle can't open the same symbol while
        # Telegram confirmation is being awaited (up to TELEGRAM_CONFIRM_TIMEOUT s).
        async with lock:
            _max = (
                self._runtime_config.get_max_positions()
                if self._runtime_config is not None
                else Config.MAX_POSITIONS
            )
            if len(monitored) >= _max:
                logger.warning(
                    "Max positions (%d) reached, skipping %s",
                    _max,
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

            _rc = getattr(self, "_runtime_config", None)
            _auto_exec = (
                _rc.get_auto_execute() if _rc is not None else Config.AUTO_EXECUTE
            )
            if _auto_exec:
                # AUTO_EXECUTE=true → без диалога, исполняем сразу
                confirmed = True
            else:
                _timeout = (
                    _rc.get_confirm_timeout()
                    if _rc is not None
                    else Config.TELEGRAM_CONFIRM_TIMEOUT
                )
                confirmed = await self._telegram.ask_confirm(
                    top,
                    live_win_rate=live_wr,
                    live_trades=live_n,
                    live_ev=live_ev,
                    bt_win_rate=bt["win_rate"],
                    bt_trades=bt["total_trades"],
                    bt_ev=bt["ev"],
                    timeout=_timeout,
                )
            if not confirmed:
                logger.info("Trade rejected via Telegram: %s", sym)
                return

            entry = top.get("entry", 0.0)
            if not entry:
                return

            # УЛУЧШЕНИЕ 5: фильтр по ликвидности (спред + объём)
            if not await self._check_liquidity(sym, entry):
                return

            sl_price = top.get("stop_loss", 0.0)
            tp_price = top.get("take_profit", 0.0)

            # Валидируем каждый уровень независимо — инвертированный SL опасен
            # даже при отсутствии TP, и наоборот.
            if action == "buy":
                if sl_price and sl_price >= entry:
                    logger.error(
                        "Invalid SL for buy %s: entry=%.4f sl=%.4f — skipping",
                        sym,
                        entry,
                        sl_price,
                    )
                    return
                if tp_price and tp_price <= entry:
                    logger.error(
                        "Invalid TP for buy %s: entry=%.4f tp=%.4f — skipping",
                        sym,
                        entry,
                        tp_price,
                    )
                    return
            elif action == "sell":
                if sl_price and sl_price <= entry:
                    logger.error(
                        "Invalid SL for sell %s: entry=%.4f sl=%.4f — skipping",
                        sym,
                        entry,
                        sl_price,
                    )
                    return
                if tp_price and tp_price >= entry:
                    logger.error(
                        "Invalid TP for sell %s: entry=%.4f tp=%.4f — skipping",
                        sym,
                        entry,
                        tp_price,
                    )
                    return

            quantity = self._size_position(top, balance, live_wr, live_n, entry)

            # УЛУЧШЕНИЕ 6: масштабирование при просадке от пика баланса
            scale = self._drawdown_scale(balance)
            if scale < 1.0:
                quantity *= scale

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

            # После Almgren-Chriss entry мог сдвинуться достаточно, чтобы
            # инвалидировать SL/TP которые прошли валидацию до поправки.
            if action == "buy":
                if sl_price and sl_price >= entry:
                    logger.error(
                        "Post-impact SL invalid for buy %s: adj_entry=%.4f sl=%.4f"
                        " — skipping",
                        sym,
                        entry,
                        sl_price,
                    )
                    return
                if tp_price and tp_price <= entry:
                    logger.error(
                        "Post-impact TP invalid for buy %s: adj_entry=%.4f tp=%.4f"
                        " — skipping",
                        sym,
                        entry,
                        tp_price,
                    )
                    return
            elif action == "sell":
                if sl_price and sl_price <= entry:
                    logger.error(
                        "Post-impact SL invalid for sell %s: adj_entry=%.4f sl=%.4f"
                        " — skipping",
                        sym,
                        entry,
                        sl_price,
                    )
                    return
                if tp_price and tp_price >= entry:
                    logger.error(
                        "Post-impact TP invalid for sell %s: adj_entry=%.4f tp=%.4f"
                        " — skipping",
                        sym,
                        entry,
                        tp_price,
                    )
                    return

            quantity = self._api.round_quantity(sym, quantity)
            if quantity <= 0:
                return

            # Default commission — used in paper trading and as fallback when the
            # exchange response doesn't include fee info.
            commission = quantity * entry * Config.COMMISSION_RATE
            exchange_sl_id: Optional[str] = None
            exchange_tp_id: Optional[str] = None

            # Volatility-targeted leverage: target_risk / (ATR / price)
            # Falls back to Config.LEVERAGE when ATR is unavailable
            dynamic_lev = _calc_dynamic_leverage(
                top,
                entry,
                balance=balance,
                peak_balance=self._peak_balance,
                runtime_config=self._runtime_config,
            )

            if Config.PAPER_TRADING:
                logger.info(
                    "[PAPER] %s %.6f %s @ %.4f", action.upper(), quantity, sym, entry
                )
                bal = self._get_paper_balance()
                if action == "buy":
                    # Открытие LONG: тратим деньги
                    self._set_paper_balance(bal - quantity * entry - commission)
                else:
                    if Config.MARKET_TYPE == "spot":
                        # SPOT продажа: получаем деньги (закрытие лонга)
                        self._set_paper_balance(bal + quantity * entry - commission)
                    else:
                        # LINEAR SHORT: резервируем маржу (entry / leverage)
                        margin = quantity * entry / dynamic_lev
                        self._set_paper_balance(bal - margin - commission)
            else:
                if Config.MARKET_TYPE != "spot":
                    try:
                        await self._api.set_leverage(sym, dynamic_lev)
                    except Exception as _lev_err:
                        logger.warning("set_leverage failed for %s: %s", sym, _lev_err)
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
                "partial_tp_triggered": False,
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

    # ── Liquidity filter (УЛУЧШЕНИЕ 5) ───────────────────────────────────────

    async def _check_liquidity(self, sym: str, price: float) -> bool:
        """Возвращает True если ликвидность достаточная для входа.

        Проверяет спред (ask-bid)/mid и 24h volume в USDT.
        При ошибке получения данных возвращает True (не блокируем торговлю).
        """
        try:
            ticker = await self._api.exchange.fetch_ticker(sym)
            bid = ticker.get("bid") or price
            ask = ticker.get("ask") or price
            volume_usdt = ticker.get("quoteVolume") or 0.0

            mid = (bid + ask) / 2
            spread_pct = (ask - bid) / mid * 100 if mid > 0 else 999.0

            if spread_pct > Config.MAX_SPREAD_PCT:
                logger.info(
                    "Liquidity filter: %s spread=%.3f%% > %.3f%% — skip",
                    sym,
                    spread_pct,
                    Config.MAX_SPREAD_PCT,
                )
                return False
            if volume_usdt < Config.MIN_VOLUME_USDT:
                logger.info(
                    "Liquidity filter: %s volume=$%.0f < $%.0f — skip",
                    sym,
                    volume_usdt,
                    Config.MIN_VOLUME_USDT,
                )
                return False
            return True
        except Exception as e:
            logger.debug("Liquidity check failed for %s: %s — allowing", sym, e)
            return True  # при ошибке — не блокируем

    # ── Drawdown scaling (УЛУЧШЕНИЕ 6) ───────────────────────────────────────

    def _drawdown_scale(self, balance: float) -> float:
        """Возвращает множитель размера позиции [DRAWDOWN_SCALE_FACTOR, 1.0].

        При просадке >= DRAWDOWN_SCALE_THRESHOLD от пика возвращает
        DRAWDOWN_SCALE_FACTOR (по умолчанию 0.5). Иначе — 1.0.
        Пик обновляется каждый раз когда баланс растёт.
        """
        try:
            enabled = bool(Config.DRAWDOWN_SCALE_ENABLED)
            threshold = float(Config.DRAWDOWN_SCALE_THRESHOLD)
            factor = float(Config.DRAWDOWN_SCALE_FACTOR)
        except (TypeError, ValueError):
            return 1.0
        if not enabled:
            return 1.0
        if balance > self._peak_balance:
            self._peak_balance = balance
        if self._peak_balance <= 0:
            return 1.0
        drawdown = (self._peak_balance - balance) / self._peak_balance
        if drawdown >= threshold:
            logger.warning(
                "Drawdown %.1f%% from peak $%.2f — scaling position size to %.0f%%",
                drawdown * 100,
                self._peak_balance,
                factor * 100,
            )
            return factor
        return 1.0

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
