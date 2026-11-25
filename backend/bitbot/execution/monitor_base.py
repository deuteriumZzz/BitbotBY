"""Монитор позиций, основа: состояние, события для приложения, dynamic exit, circuit breaker.

Монитор разложен по ролям цепочкой классов (каждый видит методы нижних):
MonitorBase → BookingMixin (учёт закрытия) → ClosingMixin (закрытие на бирже)
→ LevelsMixin (SL/TP, трейлинг, частичное закрытие) → PositionMonitor (цикл).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set, Tuple

from bitbot.config.settings import Config
from bitbot.domain.types import PositionRecord, is_trusted

if TYPE_CHECKING:
    from bitbot.adapters.exchange.bybit import BybitAPI
    from bitbot.adapters.telegram.notifier import TelegramNotifier
    from bitbot.execution.portfolio import PortfolioManager
    from bitbot.ml.online_learner import OnlineLearner
    from bitbot.storage.redis_client import RedisClient
    from bitbot.storage.trade_history import TradeHistory

logger = logging.getLogger(__name__)


class MonitorBase:
    ledger: Any = None  # журнал решений; подключает TradingBot после создания

    def __init__(
        self,
        api: "BybitAPI",
        trade_history: "TradeHistory",
        telegram: "TelegramNotifier",
        portfolio_manager: "PortfolioManager",
        set_running: Callable[[bool], None],
        redis: "Optional[RedisClient]" = None,
        online_learner: "Optional[OnlineLearner]" = None,
        # (символ, {"reason": tp|sl|trail|manual_app|…, "price": цена выхода})
        on_position_closed: "Optional[Callable[[str, Dict[str, Any]], None]]" = None,
        emit: "Optional[Callable[[str, Dict[str, Any]], None]]" = None,
    ) -> None:
        self._api = api
        self._trade_history = trade_history
        self._telegram = telegram
        self._portfolio_manager = portfolio_manager
        # Callback instead of a direct TradingBot reference so the circuit
        # breaker can halt trading without a circular import.
        self._set_running = set_running
        self._redis = redis
        self._online_learner = online_learner
        self._on_position_closed = on_position_closed
        self._emit = emit
        # Load persisted counter so circuit breaker survives bot restarts.
        self._consecutive_losses: int = self._load_cb_state()
        # State for dynamic exits — updated each cycle via update_market_state().
        self._current_signals: Dict[str, Any] = {}
        self._current_market_ctx: Dict[str, Any] = {}
        self._current_regime: str = "unknown"
        # Счётчик подтверждений для regime_flip/funding: {symbol: count}
        # При N=3 подряд — escalate to force_close
        self._exit_confirm_counts: Dict[str, int] = {}
        self._EXIT_CONFIRM_THRESHOLD = 3
        # Keeps strong references to fire-and-forget tasks so GC can't destroy them.
        self._background_tasks: "Set[asyncio.Task[Any]]" = set()

    def _publish(self, type_: str, data: Dict[str, Any]) -> None:
        if self._emit is not None:
            try:
                self._emit(type_, data)
            except Exception as e:
                logger.debug("event %s not published: %s", type_, e)

    def _publish_tick(self, sym: str, pos: PositionRecord, price: float) -> None:
        """Цена и текущие уровни позиции — для живых карточек в приложении."""
        self._publish(
            "tick",
            {
                "symbol": sym,
                "trade_id": pos.get("trade_id"),
                "price": price,
                "side": "long" if pos.get("side", "buy") == "buy" else "short",
                "entry": pos.get("entry"),
                "qty": pos.get("qty"),
                "sl": pos.get("stop_loss"),
                "tp": pos.get("take_profit"),
                "trailing": pos.get("trailing", True),
            },
        )

    # ── Dynamic exit state ────────────────────────────────────────────────────

    def update_market_state(
        self,
        signals: Dict[str, Any],
        market_ctx: Dict[str, Any],
        regime: str,
    ) -> None:
        """Обновляет рыночное состояние для динамических выходов.

        Вызывается trading_bot каждый цикл после генерации рекомендаций.
        """
        self._current_signals = signals
        self._current_market_ctx = market_ctx
        self._current_regime = regime

    def _check_dynamic_exit(
        self, sym: str, record: PositionRecord
    ) -> Tuple[Optional[str], str]:
        """Проверяет условия досрочного выхода из позиции.

        :return: (action: str | None, reason: str)
            action="force_close" — закрыть по рынку немедленно (сильный сигнал)
            action="tighten_sl"  — подтянуть SL ближе (дать шанс на отскок)
            action=None          — ничего не делать
        """
        if not self._current_signals and not self._current_market_ctx:
            return None, ""
        if is_trusted(record):
            # Вход по посту доверенного канала: стратегии, режим и контекст
            # рынка его не закрывают — только SL, TP и трейлинг.
            return None, ""

        side = record.get("side", "buy")
        ctx = self._current_market_ctx.get(sym, {})
        signal = self._current_signals.get(sym, {})

        # 1. Разворот сигнала с высокой уверенностью → закрыть сразу
        sig_action = signal.get("action", "")
        sig_conf = float(signal.get("confidence", 0.0))
        if sig_action and sig_conf >= 0.80:
            if side == "buy" and sig_action == "sell":
                return "force_close", f"signal_reversal SELL conf={sig_conf:.2f}"
            if side == "sell" and sig_action == "buy":
                return "force_close", f"signal_reversal BUY conf={sig_conf:.2f}"

        # 2. Смена режима против позиции → подтянуть SL (цена может отскочить)
        regime = self._current_regime
        if side == "buy" and regime == "trending_down":
            return "tighten_sl", "regime_flip trending_down vs LONG"
        if side == "sell" and regime == "trending_up":
            return "tighten_sl", "regime_flip trending_up vs SHORT"

        # 3. Funding развернулся против позиции → подтянуть SL
        funding_signal = ctx.get("funding_signal", "neutral")
        if side == "sell" and funding_signal == "short_overheated":
            return "tighten_sl", "funding short_overheated — short squeeze risk"
        if side == "buy" and funding_signal == "long_overheated":
            return "tighten_sl", "funding long_overheated — long liquidation risk"

        # 4. Экстремальный Fear&Greed → закрыть сразу (фиксируем прибыль)
        fear_greed = int(ctx.get("fear_greed", 50))
        if side == "sell" and fear_greed <= 10:
            return "force_close", f"fear_greed={fear_greed} extreme_fear"
        if side == "buy" and fear_greed >= 92:
            return "force_close", f"fear_greed={fear_greed} extreme_greed"

        # 5. Ликвидационный каскад → закрыть сразу
        liq = ctx.get("liquidation_pressure", "neutral")
        if side == "buy" and liq == "long_liquidation":
            return "force_close", "long_liquidation cascade — exit LONG"
        if side == "sell" and liq == "short_squeeze":
            return "force_close", "short_squeeze cascade — exit SHORT"

        return None, ""

    # ── Circuit-breaker persistence ───────────────────────────────────────────

    def _load_cb_state(self) -> int:
        """Загружает счётчик подряд идущих убытков из Redis (0 если недоступен)."""
        if self._redis is None:
            return 0
        state = self._redis.load_trading_state("circuit_breaker")
        return int((state or {}).get("consecutive_losses", 0))

    def _save_cb_state(self) -> None:
        """Сохраняет счётчик в Redis чтобы пережить рестарт бота."""
        if self._redis is not None:
            self._redis.save_trading_state(
                "circuit_breaker",
                {"consecutive_losses": self._consecutive_losses},
            )

    def _update_circuit_breaker(
        self,
        side: str,
        exit_price: float,
        entry_price: float,
    ) -> None:
        """
        Останавливает торговлю после N последовательных убыточных сделок.

        Серия убытков (например, flash crash) сигнализирует о сломанном
        рыночном режиме. Случайные убытки вперемешку с прибылями — норма.

        :param side: Направление закрытой сделки ("buy" или "sell").
        :param exit_price: Цена закрытия.
        :param entry_price: Цена открытия.
        """
        cb = Config.CIRCUIT_BREAKER_LOSSES
        if cb <= 0:
            return

        is_loss = (side == "buy" and exit_price < entry_price) or (
            side == "sell" and exit_price > entry_price
        )
        if is_loss:
            self._consecutive_losses += 1
            logger.warning(
                "Circuit breaker: %d/%d consecutive losses",
                self._consecutive_losses,
                cb,
            )
            self._save_cb_state()
            if self._consecutive_losses >= cb:
                if Config.PAPER_TRADING:
                    msg = (
                        f"⚠️ Circuit breaker: {self._consecutive_losses}"
                        " убытков подряд. В paper режиме торговля продолжается."
                    )
                else:
                    msg = (
                        f"⛔ Circuit breaker: {self._consecutive_losses}"
                        " losses in a row. Trading halted automatically."
                    )
                    self._publish(
                        "guard",
                        {
                            "kind": "circuit_breaker",
                            "active": True,
                            "detail": (
                                f"{self._consecutive_losses} убыточных сделок подряд"
                            ),
                        },
                    )
                    self._set_running(False)
                logger.critical(msg)
                _t = asyncio.create_task(self._telegram.notify(msg))
                self._background_tasks.add(_t)
                _t.add_done_callback(self._background_tasks.discard)
        else:
            self._consecutive_losses = 0
            self._save_cb_state()
