"""Главный модуль торгового бота BitbotBY.

Содержит класс TradingBot — оркестратор всего цикла:
сканирование рынка, генерация сигналов, исполнение ордеров,
мониторинг позиций и управление рисками.
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import ccxt
import pandas as pd

from config import Config
from src.ai_analyzer import AIAnalyzer
from src.bybit_api import BybitAPI
from src.constants import REDIS_TTL_MARKET_DATA, SILENT_DEATH_ALERT_COOLDOWN
from src.correlation_filter import CorrelationFilter
from src.cycle import CycleRunner
from src.data_loader import DataLoader
from src.health_server import cycles_counter
from src.logger import _SecretFilter, setup_logging
from src.macro_calendar import MacroCalendar
from src.market_context import MarketContext
from src.market_scanner import MarketScanner
from src.news_analyzer import NewsAnalyzer
from src.online_learner import OnlineLearner
from src.order_executor import OrderExecutor
from src.portfolio_manager import PortfolioManager
from src.portfolio_optimizer import PortfolioOptimizer
from src.position_monitor import PositionMonitor
from src.redis_client import RedisClient
from src.regime_detector import RegimeDetector
from src.risk_management import RiskManager
from src.runtime_config import BLUECHIP_BASES, RuntimeConfig
from src.season_detector import SeasonDetector
from src.signal_combiner import SignalCombiner
from src.strategies import TradingStrategy, create_strategy
from src.telegram_commander import (
    TelegramCommander,
    _kb_main,
    _kb_sac_train,
    _kb_tune_sac,
)
from src.telegram_notifier import TelegramNotifier
from src.trade_history import TradeHistory
from src.types import PositionRecord

_SECONDS_PER_HOUR: int = 3_600

setup_logging()
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

    def __init__(self) -> None:
        self.redis = RedisClient()
        self._runtime_config = RuntimeConfig(redis_client=self.redis)
        # Восстанавливаем runtime-переключение Paper/Live после рестарта
        override = self._runtime_config.get_paper_trading_override()
        if override is not None:
            Config.PAPER_TRADING = override
            os.environ["PAPER_TRADING"] = "true" if override else "false"
        self.api = BybitAPI()
        self.data_loader = DataLoader()
        self.portfolio_manager = PortfolioManager(
            Config.INITIAL_BALANCE, redis_client=self.redis
        )
        self.strategy: Optional[TradingStrategy] = None
        self.risk_manager = RiskManager(
            Config.INITIAL_BALANCE,
            Config.RISK_PER_TRADE,
        )
        self.scanner = MarketScanner(
            self.api, self.data_loader, rc=self._runtime_config
        )
        self.news = NewsAnalyzer()
        self.ai = AIAnalyzer(runtime_config=self._runtime_config)
        self.combiner = SignalCombiner(self.ai, rc=self._runtime_config)
        self.regime_detector = RegimeDetector()
        self._current_regime: str = "unknown"
        self.portfolio_optimizer = PortfolioOptimizer()
        self.corr_filter = CorrelationFilter(
            window=Config.CORRELATION_WINDOW,
            max_corr=(
                Config.MAX_CORRELATION_PAPER
                if Config.PAPER_TRADING
                else Config.MAX_CORRELATION
            ),
        )
        self.is_running: bool = False
        # Явная схема через TypedDict — заменяет Dict[str, Any], чтобы ловить
        # отсутствующие поля при создании (mypy), а не как KeyError в мониторе.
        self._monitored: Dict[str, PositionRecord] = {}
        self._monitored_lock: asyncio.Lock = asyncio.Lock()
        self._monitor_task: Optional[asyncio.Task[None]] = None
        self.trade_history = TradeHistory()
        self.telegram = TelegramNotifier(
            Config.TELEGRAM_BOT_TOKEN,
            Config.TELEGRAM_CHAT_ID,
        )
        self._paper_balance: float = Config.INITIAL_BALANCE
        self._live_balance_cache: float = Config.INITIAL_BALANCE
        # Equity (баланс + unrealized PnL) для drawdown halt
        self._live_equity_cache: float = Config.INITIAL_BALANCE
        # Пик equity для hard drawdown halt (обновляется в main loop)
        self._peak_balance: float = 0.0
        # Счётчик подтверждения просадки: halt только после N циклов подряд
        self._drawdown_consec: int = 0
        # кэш режимов: символ → (режим, timestamp) — обновляется каждые 5 мин
        self._regime_cache: Dict[str, Tuple[str, float]] = {}
        self._regime_ttl: float = float(
            os.getenv("REGIME_CACHE_TTL", str(REDIS_TTL_MARKET_DATA))
        )
        # детектор «тихой смерти»: фиксируем время последней сделки и последнего алерта
        self._last_trade_at: Optional[float] = None
        self._silent_death_alerted_at: float = 0.0
        self._daily_limit_notified_at: float = 0.0
        self._silent_death_hours: float = float(os.getenv("SILENT_DEATH_HOURS", "6"))
        if Config.PAPER_TRADING:
            logger.warning("*** PAPER TRADING MODE — no real orders will be placed ***")
        # Регистрируем секреты для маскировки в логах
        _SecretFilter.register(
            Config.BYBIT_API_KEY,
            Config.BYBIT_API_SECRET,
            Config.ANTHROPIC_API_KEY,
            Config.DEEPSEEK_API_KEY,
            Config.TELEGRAM_BOT_TOKEN,
            getattr(Config, "OPENAI_API_KEY", ""),
            getattr(Config, "NEWS_API_KEY", ""),
        )

        # Мониторинг и исполнение — отдельные объекты: каждый отвечает за своё
        # (счётчик circuit-breaker, логика размера позиции) и тестируется
        # независимо от всего TradingBot.
        self._online_learner = OnlineLearner(redis_client=self.redis)
        self._position_monitor = PositionMonitor(
            api=self.api,
            trade_history=self.trade_history,
            telegram=self.telegram,
            portfolio_manager=self.portfolio_manager,
            set_running=lambda v: setattr(self, "is_running", v),
            redis=self.redis,
            online_learner=self._online_learner,
        )
        self._executor = OrderExecutor(
            api=self.api,
            trade_history=self.trade_history,
            telegram=self.telegram,
            risk_manager=self.risk_manager,
            portfolio_optimizer=self.portfolio_optimizer,
            corr_filter=self.corr_filter,
            # Лямбды откладывают поиск dict/lock до момента вызова, чтобы код,
            # заменяющий bot._monitored (в том числе в тестах), всегда был виден.
            get_monitored=lambda: self._monitored,
            get_lock=lambda: self._monitored_lock,
            get_paper_balance=lambda: self._paper_balance,
            set_paper_balance=self._set_paper_balance,
            set_last_trade_at=lambda v: setattr(self, "_last_trade_at", v),
            get_current_regime=lambda: self._current_regime,
            runtime_config=self._runtime_config,
        )
        self._commander = TelegramCommander(
            notifier=self.telegram,
            runtime_config=self._runtime_config,
            get_state=self._get_bot_state,
            on_mode_switch=self._handle_mode_switch,
            on_restart=self._handle_restart,
        )
        self._cycle = CycleRunner(
            news=self.news,
            scanner=self.scanner,
            portfolio_optimizer=self.portfolio_optimizer,
            telegram=self.telegram,
            get_current_regime=lambda: self._current_regime,
        )
        self._market_context = MarketContext()
        # УЛУЧШЕНИЕ 7: макро-событийный blackout фильтр
        self._macro_calendar = MacroCalendar()
        self._season_detector = SeasonDetector()

    async def initialize(self) -> None:
        """
        Инициализирует API, стратегию, Redis-состояние.

        :raises Exception: Если инициализация не удалась.
        """
        Config().validate()
        try:
            await self.api.initialize(
                Config.BYBIT_API_KEY,
                Config.BYBIT_API_SECRET,
                testnet=Config.TESTNET,
            )
        except ccxt.AuthenticationError as e:
            if Config.PAPER_TRADING:
                logger.warning(
                    "API auth failed — paper trading mode, public data only: %s",
                    e,
                )
            else:
                logger.error("Failed to initialize Bybit API: %s", e)
                raise
        try:
            await self.data_loader.initialize(
                Config.BYBIT_API_KEY,
                Config.BYBIT_API_SECRET,
            )
            if not Config.PAPER_TRADING:
                try:
                    bal = await self.api.get_balance()
                    real_balance = (
                        float(bal.get("free", {}).get("USDT", 0)) if bal else 0
                    )
                    if real_balance > 0:
                        self.risk_manager.initial_balance = real_balance
                        self.risk_manager._day_start_balance = real_balance
                        self.portfolio_manager.current_balance = real_balance
                        self._live_balance_cache = real_balance
                        logger.info(
                            "Live mode: real balance synced — $%.2f", real_balance
                        )
                    else:
                        logger.warning(
                            "Live mode: real balance is 0 — check Bybit account"
                        )
                except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                    logger.warning(
                        "Live mode: balance sync failed, using INITIAL_BALANCE: %s", e
                    )
            self.strategy = TradingStrategy(Config.DEFAULT_STRATEGY)
            await self.strategy.initialize()
            await self._restore_state()
            await self._reconcile_positions()
            await self.telegram.start()
            self._commander.register()
            await self._fit_regime_detector()
            self._load_corr_filter()
            logger.info("Trading bot initialized successfully")
        except Exception as e:
            logger.error("Bot initialization failed: %s", e)
            raise

    async def _check_tune_reminder(self) -> None:
        """Напоминает о тюнинге если прошло 60+ дней и tune ещё не запускался."""
        if self._runtime_config.is_tune_reminded():
            return
        if os.path.exists(os.path.join("models", "best_hyperparams.json")):
            return
        days = self._runtime_config.days_since_first_start()
        if days < 60:
            return
        self._runtime_config.set_tune_reminded()
        await self.telegram.notify(
            f"📅 *Прошло {days} дней с запуска бота*\n\n"
            "Рекомендуем запустить тюнинг SAC — модель будет обучена\n"
            "с параметрами оптимальными для твоего рынка.\n\n"
            "⏱ Займёт ~3 часа, бот продолжает торговать.",
            reply_markup=_kb_tune_sac(),
        )

    async def _check_sac_model(self) -> None:
        """Предупреждает об отсутствии SAC модели для текущего профиля."""
        profile = self._runtime_config.get_market_profile()
        model_path = self._runtime_config.get_sac_model_path()

        if os.path.exists(model_path):
            return
        if self._runtime_config.is_sac_prompted(profile):
            return
        self._runtime_config.set_sac_prompted(profile)

        profile_labels = {"bluechip": "🔵 Блючипы", "altcoin": "🟡 Альткоины"}
        profile_str = profile_labels.get(profile, "")
        profile_note = f" для профиля *{profile_str}*" if profile_str else ""

        await self.telegram.notify(
            f"⚠️ *SAC модель{profile_note} не найдена*\n\n"
            "Без неё бот работает только на AI + локальных стратегиях.\n"
            "В `hybrid` режиме SAC даёт дополнительный сигнал.\n\n"
            f"📁 Ожидается: `{model_path}`\n"
            "⏱ Обучение займёт ~60 мин на CPU.\n"
            "_После достаточного числа сделок модель переобучается автоматически._\n\n"
            "Хотите обучить SAC сейчас?",
            reply_markup=_kb_sac_train(),
        )

    async def _reconcile_positions(self) -> None:
        """
        Сверяет _monitored с реальными позициями биржи после перезапуска.

        - Удаляет из _monitored позиции, которых больше нет на бирже.
        - Добавляет позиции, которые есть на бирже, но потеряны при крэше бота.

        Пропускается в PAPER_TRADING режиме (нечего сверять).
        """
        if Config.PAPER_TRADING:
            return
        try:
            exchange_positions = await self.api.fetch_positions()
        except Exception as exc:
            logger.warning("Reconciliation skipped: %s", exc)
            return

        # ccxt symbol может быть 'BTC/USDT:USDT' — нормализуем к 'BTC/USDT'
        active: dict = {}
        for p in exchange_positions:
            sym = p.get("symbol", "")
            if ":" in sym:
                sym = sym.split(":")[0]
            active[sym] = p

        async with self._monitored_lock:
            stale = [s for s in self._monitored if s not in active]
            for sym in stale:
                logger.warning(
                    "Reconcile: удалена устаревшая позиция %s (закрыта на бирже)",
                    sym,
                )
                self._monitored.pop(sym, None)

            added = []
            for sym, pos in active.items():
                if sym not in self._monitored:
                    side = pos.get("side", "buy")
                    qty = float(pos.get("contracts") or 0)
                    if qty <= 0:
                        # частичное исполнение / позиция с нулевым размером — пропускаем
                        logger.debug("Reconcile: skip zero-qty position %s", sym)
                        continue
                    entry = float(pos.get("entryPrice") or 0)

                    # Биржа возвращает SL/TP для фьючерсных позиций
                    # (поля stopLoss/takeProfit). На спотовом рынке эти поля
                    # отсутствуют — пытаемся достать из info-словаря.
                    sl = float(
                        pos.get("stopLoss")
                        or (pos.get("info") or {}).get("stopLoss")
                        or 0
                    )
                    tp = float(
                        pos.get("takeProfit")
                        or (pos.get("info") or {}).get("takeProfit")
                        or 0
                    )

                    # Запасной SL по процентному отступу от входа,
                    # если биржа не вернула. Без SL позиция не будет
                    # отслеживаться (_check_and_close игнорирует sl=0).
                    if not sl and entry > 0:
                        sl_pct = getattr(Config, "STOP_LOSS_PERCENT", 0.02)
                        sl = (
                            entry * (1.0 - sl_pct)
                            if side == "buy"
                            else entry * (1.0 + sl_pct)
                        )
                        logger.warning(
                            "Reconcile %s: SL не найден на бирже — запасной %.4f"
                            " (%.0f%% от входа)",
                            sym,
                            sl,
                            sl_pct * 100,
                        )

                    self._monitored[sym] = {
                        "trade_id": None,
                        "qty": qty,
                        "entry": entry,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "side": side,
                        "atr": 0.0,
                        "peak_price": entry,
                        "exchange_sl_id": None,
                        "exchange_tp_id": None,
                    }
                    added.append(sym)
                    logger.warning(
                        "Reconcile: восстановлена потерянная позиция %s %s %.6f @ %.4f"
                        " sl=%.4f tp=%.4f",
                        sym,
                        side,
                        qty,
                        entry,
                        sl,
                        tp,
                    )

        logger.info(
            "Reconciliation: удалено=%d восстановлено=%d активных=%d",
            len(stale),
            len(added),
            len(active),
        )

    async def _restore_state(self) -> None:
        state = self.redis.load_trading_state(Config.SYMBOL)
        if state:
            logger.info("Restored state: %s", state)
        _mode = "paper" if Config.PAPER_TRADING else "live"
        portfolio = self.redis.load_trading_state(f"portfolio_state_{_mode}")
        if portfolio:
            restored_balance = portfolio.get("balance", Config.INITIAL_BALANCE)
            self.portfolio_manager.current_balance = restored_balance
            self.portfolio_manager.positions = portfolio.get("positions", {})
            self._paper_balance = restored_balance
            self._live_balance_cache = restored_balance
        monitored_state = self.redis.load_trading_state(f"monitored_positions_{_mode}")
        if monitored_state and isinstance(monitored_state, dict):
            async with self._monitored_lock:
                for sym, pos in monitored_state.items():
                    if pos and sym not in self._monitored:
                        self._monitored[sym] = pos
            logger.info(
                "Restored %d monitored positions from Redis", len(monitored_state)
            )

    def _save_monitored_state(self) -> None:
        _mode = "paper" if Config.PAPER_TRADING else "live"
        try:
            self.redis.save_trading_state(
                f"monitored_positions_{_mode}", dict(self._monitored)
            )
        except Exception as e:
            logger.warning("Failed to save monitored state: %s", e)

    _CORR_REDIS_KEY = "corr_filter_prices"

    def _load_corr_filter(self) -> None:
        """Восстанавливает историю цен CorrelationFilter из Redis."""
        try:
            data = self.redis.load_trading_state(self._CORR_REDIS_KEY)
            if data:
                self.corr_filter.from_dict(data)
                logger.info(
                    "CorrelationFilter: восстановлено %d символов из Redis", len(data)
                )
        except Exception as exc:
            logger.warning("CorrelationFilter load failed: %s", exc)

    def _save_corr_filter(self) -> None:
        """Сохраняет историю цен CorrelationFilter в Redis."""
        try:
            self.redis.save_trading_state(
                self._CORR_REDIS_KEY, self.corr_filter.to_dict()
            )
        except Exception as exc:
            logger.warning("CorrelationFilter save failed: %s", exc)

    async def _fit_regime_detector(self) -> None:
        """Загружает исторические данные и обучает RegimeDetector."""
        try:
            df = await self.data_loader.get_market_data(
                Config.SYMBOL, self._runtime_config.get_timeframe(), limit=2000
            )
            if df is not None and not df.empty:
                self.regime_detector.fit(df)
                logger.info("RegimeDetector fitted on %d candles", len(df))
        except Exception as exc:
            logger.warning("RegimeDetector fit skipped: %s", exc)

    def _get_bot_state(self) -> Dict[str, Any]:
        """Снэпшот состояния бота для TelegramCommander."""
        positions = [
            {
                "symbol": sym,
                "side": pos.get("side", "buy"),
                "qty": pos.get("qty", 0),
                "entry_price": pos.get("entry", 0),
                "stop_loss": pos.get("stop_loss", 0),
                "take_profit": pos.get("take_profit", 0),
                "pnl_pct": pos.get("pnl_pct", 0),
            }
            for sym, pos in self._monitored.items()
            if pos is not None
        ]
        balance = (
            self.portfolio_manager.current_balance
            if Config.PAPER_TRADING
            else self._live_balance_cache
        )
        from src.strategies import get_all_strategies  # noqa: PLC0415

        disabled = self._runtime_config.get_disabled_strategies()
        strategies = [
            {**s, "enabled": s["name"] not in disabled} for s in get_all_strategies()
        ]
        return {
            "balance": balance,
            "initial_balance": Config.INITIAL_BALANCE,
            "positions": positions,
            "paper_trading": Config.PAPER_TRADING,
            "strategies": strategies,
        }

    def _set_paper_balance(self, value: float) -> None:
        """Синхронизирует оба регистра бумажного баланса.

        OrderExecutor уменьшает баланс при открытии позиции через этот метод.
        PositionMonitor восстанавливает его через portfolio_manager.update_portfolio.
        Оба должны смотреть в один источник истины — portfolio_manager.current_balance.
        """
        self._paper_balance = value
        self.portfolio_manager.current_balance = value

    async def _get_balance_usdt(self) -> float:
        """Свободный баланс в USDT; использует PortfolioManager как запасной
        источник при недоступности биржи."""
        if Config.PAPER_TRADING:
            return self.portfolio_manager.current_balance
        try:
            bal = await self.api.get_balance()
            if bal:
                # Bybit UNIFIED: ccxt не маппит free/total — читаем raw поля
                info_result = (bal.get("info") or {}).get("result") or {}
                account = (info_result.get("list") or [{}])[0]
                available = account.get("totalAvailableBalance")
                equity = account.get("totalEquity")
                if available is not None:
                    value = float(available)
                    self._live_equity_cache = float(equity) if equity else value
                else:
                    # fallback: стандартный ccxt путь (не-UNIFIED аккаунты)
                    value = float((bal.get("free") or {}).get("USDT") or 0)
                    self._live_equity_cache = value
                self._live_balance_cache = value
                return value
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logger.warning("Balance fetch failed, using cached value: %s", e)
        return self._live_balance_cache

    def _filter_by_balance(self, recs: list, balance: float) -> list:
        """Оставляет только рекомендации, минимальный лот которых укладывается в баланс.

        Минимальный лот = entry * 0.001 (наименьший торгуемый номинал на Bybit).
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
                    "Skip %s: need $%.2f, have $%.2f",
                    r["symbol"],
                    entry * 0.001,
                    balance,
                )
        return result

    async def _execute_top_rec(
        self,
        filtered: list,
        market_data: dict,
    ) -> None:
        """Исполняет лучшую рекомендацию — делегирует OrderExecutor."""
        if not filtered:
            return
        # УЛУЧШЕНИЕ 7: пропускаем исполнение во время макро-событийного blackout
        if Config.MACRO_BLACKOUT_ENABLED:
            if await self._macro_calendar.is_blackout():
                logger.warning("Macro blackout active — skipping execution")
                return
        balance = await self._get_balance_usdt()
        max_pos = self._runtime_config.get_max_positions()
        for rec in filtered[:max_pos]:
            await self._executor.execute(rec, market_data, balance)

    async def _close_all_open_positions(self, reason: str) -> None:
        """Принудительно закрывает все открытые позиции — делегирует PositionMonitor."""
        await self._position_monitor.close_all_positions(
            self._monitored, self._monitored_lock, reason=reason
        )

    async def _handle_restart(self) -> None:
        """Останавливает бота — Docker с restart: unless-stopped поднимет заново.

        Не вызываем self.stop() изнутри Telegram-хендлера: это дедлок
        (stop() ждёт завершения хендлеров, хендлер ждёт stop()).
        Вместо этого ждём 2 с чтобы Telegram успел отправить сообщение,
        затем SIGTERM — Docker перезапустит контейнер.
        """
        import os as _os
        import signal as _signal

        logger.info("Restart requested via Telegram — sending SIGTERM in 2s")
        await asyncio.sleep(2)
        _os.kill(_os.getpid(), _signal.SIGTERM)

    async def _handle_mode_switch(self) -> None:
        """Вызывается при переключении paper ↔ live через Telegram.

        Очищает _monitored от старых позиций и синхронизирует с биржей
        чтобы paper-позиции не отображались в live режиме и наоборот.
        """
        async with self._monitored_lock:
            self._monitored.clear()
        self._peak_balance = 0.0
        self._drawdown_consec = 0
        logger.info("Mode switch: _monitored cleared, peak balance reset")
        await self._reconcile_positions()

    def _is_atr_spike(self, market_data: Dict) -> bool:
        """Возвращает True если ATR вырос в ATR_SPIKE_MULT раз от медианы.

        Проверяет до 3 символов (BTC, основной символ, первый доступный).
        """
        lookback = Config.ATR_SPIKE_LOOKBACK
        mult = Config.ATR_SPIKE_MULT
        candidates: list = []
        for sym in ["BTCUSDT", Config.SYMBOL]:
            if sym in market_data and sym not in candidates:
                candidates.append(sym)
        for sym in market_data:
            if sym not in candidates:
                candidates.append(sym)
            if len(candidates) >= 3:
                break
        for sym in candidates:
            df = market_data.get(sym)
            if df is None or "atr" not in df.columns or len(df) < lookback + 2:
                continue
            baseline = df["atr"].iloc[-(lookback + 1) : -1].median()
            current = df["atr"].iloc[-1]
            if baseline > 0 and current >= baseline * mult:
                logger.warning(
                    "ATR spike on %s: current=%.4f baseline=%.4f"
                    " (%.1fx >= %.1fx threshold)",
                    sym,
                    current,
                    baseline,
                    current / baseline,
                    mult,
                )
                return True
        return False

    async def _monitor_positions(self) -> None:
        """Фоновый цикл SL/TP/трейлинг/circuit-breaker —
        делегирует PositionMonitor."""
        await self._position_monitor.run(
            is_running=lambda: self.is_running,
            monitored=self._monitored,
            lock=self._monitored_lock,
        )

    async def _season_check_loop(self) -> None:
        """Фоновый цикл детектора сезона — проверяет CoinGecko каждые 4 часа."""

        import os as _os

        try:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        except ImportError:
            InlineKeyboardButton = None  # type: ignore[assignment,misc]
            InlineKeyboardMarkup = None  # type: ignore[assignment,misc]

        check_interval_h = float(_os.getenv("SEASON_CHECK_INTERVAL_H", "4"))
        check_interval_s = int(check_interval_h * 3600)

        # Первая проверка сразу — чтобы сезон был виден в статусе с момента старта.
        # Алерт не сработает: гистерезис требует 2 проверки подряд.
        await asyncio.sleep(30)

        while self.is_running:
            try:
                # Fear & Greed — один запрос к alternative.me
                try:
                    import aiohttp as _aiohttp

                    async with _aiohttp.ClientSession(
                        timeout=_aiohttp.ClientTimeout(total=10)
                    ) as _s:
                        async with _s.get(
                            "https://api.alternative.me/fng/?limit=1"
                        ) as _r:
                            if _r.status == 200:
                                _fng = await _r.json()
                                _entry = _fng.get("data", [{}])[0]
                                _fng_val = int(_entry.get("value", 0))
                                _fng_label = _entry.get("value_classification", "")
                                self._runtime_config.set_fear_greed(
                                    _fng_val, _fng_label
                                )
                except Exception as _exc:
                    logger.debug("FNG fetch failed: %s", _exc)

                data = await self._season_detector.fetch_data()
                if data:
                    index = self._season_detector.compute_index(data)
                    if index:
                        current_profile = self._runtime_config.get_market_profile()
                        signal = self._season_detector.classify(index)
                        self._runtime_config.set_season_index(signal, index)
                        now = time.time()

                        needs_alert = self._season_detector.should_alert(
                            signal, current_profile, now
                        )
                        if needs_alert:
                            msg = self._season_detector.format_message(signal, index)

                            auto_switch = (
                                self._runtime_config.get_season_switch_mode() == "auto"
                            )
                            if auto_switch:
                                self._runtime_config.apply_market_profile(signal)
                                profile_labels = {
                                    "bluechip": "🔵 Блючипы",
                                    "altcoin": "🟡 Альткоины",
                                }
                                label = profile_labels.get(signal, signal)
                                suffix = f"\n\n✅ *Профиль переключён на {label}.*"
                                await self.telegram.notify(msg + suffix)
                            else:
                                btn_label = (
                                    "✅ Переключить на Альты"
                                    if signal == "altcoin"
                                    else "✅ Переключить на Блючипы"
                                )
                                kb = InlineKeyboardMarkup(
                                    [
                                        [
                                            InlineKeyboardButton(
                                                btn_label,
                                                callback_data=(
                                                    f"market_profile:{signal}"
                                                ),
                                            ),
                                            InlineKeyboardButton(
                                                "❌ Оставить как есть",
                                                callback_data="season_dismiss",
                                            ),
                                        ]
                                    ]
                                )
                                await self.telegram.notify(msg, reply_markup=kb)

                        logger.info(
                            "Season check: index=%.0f dom=%.1f%% signal=%s profile=%s",
                            index["altcoin_index"],
                            index["btc_dominance"],
                            signal,
                            current_profile,
                        )

            except Exception as exc:
                logger.warning("Season check loop error: %s", exc)
                await asyncio.sleep(300)  # быстрый ретрай при ошибке
                continue

            await asyncio.sleep(check_interval_s)

    async def _update_performance_stats(self) -> None:
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
        except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
            logger.error("Error updating performance stats: %s", e)

    async def _scan_and_update_correlations(
        self, symbols: list
    ) -> Dict[str, pd.DataFrame]:
        """Сканирует рыночные данные, обновляет корреляционный фильтр
        и сохраняет в Redis.

        :param symbols: Список символов для сканирования.
        :return: Словарь {символ: DataFrame с OHLCV+индикаторами}.
        """
        market_data = await self.scanner.scan_all(
            symbols, self._runtime_config.get_timeframe()
        )
        for sym, df in market_data.items():
            self.corr_filter.update_from_df(sym, df)
        self._save_corr_filter()
        return market_data

    async def _detect_regimes(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, str]:
        """Определяет рыночный режим для каждого символа с кэшем TTL.

        :param market_data: Словарь {символ: DataFrame с OHLCV+индикаторами}.
        :return: Словарь {символ: режим} ('trending_up', 'trending_down', 'ranging').
        """
        now = time.monotonic()
        regimes: Dict[str, str] = {}
        for sym, df in market_data.items():
            cached_regime, cached_ts = self._regime_cache.get(
                sym, ("unknown", float("-inf"))
            )
            if now - cached_ts < self._regime_ttl:
                regimes[sym] = cached_regime
            elif df is not None and not df.empty:
                regime = self.regime_detector.predict(df)
                regimes[sym] = regime
                self._regime_cache[sym] = (regime, now)
        self._current_regime = regimes.get(Config.SYMBOL, "unknown")
        logger.info(
            "Regimes: %s",
            {s: r for s, r in regimes.items() if r != "unknown"},
        )
        return regimes

    async def _generate_signals(
        self,
        snapshots: list,
        balance: float,
        regimes: Dict[str, str],
        market_data: Dict[str, pd.DataFrame],
    ) -> list:
        """Генерирует торговые сигналы через combiner с локальным fallback.

        :param snapshots: Список снэпшотов по символам.
        :param balance: Текущий баланс в USDT.
        :param regimes: Словарь {символ: режим}.
        :param market_data: Словарь {символ: DataFrame} для fallback-стратегии.
        :return: Список рекомендаций.
        """
        recs = await self.combiner.combine(
            snapshots,
            balance,
            regime=self._current_regime,
            regimes=regimes,
            market_data=market_data,
        )
        recs = self._cycle.optimize_allocation(recs, market_data)

        if not recs:
            logger.info("MODE=%s, no signals — local fallback", Config.MODE)
            for snap in snapshots:
                sym = snap["symbol"]
                strat, conf = self.combiner.ai.recommend_strategy_local(snap)
                _min_conf = self._runtime_config.get_signal_confidence(
                    Config.PAPER_TRADING
                )
                if conf >= _min_conf:
                    df = market_data.get(sym)
                    sig = {}
                    if df is not None and not df.empty:
                        try:
                            _local_strat = create_strategy(strat)
                            sig = _local_strat.generate_signal(df)
                        except (ValueError, KeyError):
                            pass
                    action = sig.get("action", "hold")
                    _price = float(snap.get("price") or 0.0)
                    _sl_pct = Config.STOP_LOSS_PERCENT
                    _sl = (
                        _price * (1 - _sl_pct)
                        if action == "buy"
                        else _price * (1 + _sl_pct)
                    ) if _price else 0.0
                    _tp = (
                        _price * (1 + 2 * _sl_pct)
                        if action == "buy"
                        else _price * (1 - 2 * _sl_pct)
                    ) if _price else 0.0
                    recs.append(
                        {
                            "symbol": sym,
                            "action": action,
                            "strategy": strat,
                            "confidence": conf,
                            "entry": _price,
                            "stop_loss": _sl,
                            "take_profit": _tp,
                            "reasoning": "Local analysis",
                        }
                    )
        return recs

    async def _check_silent_death(self) -> None:
        """Отправляет Telegram-алерт если бот не совершал сделок
        дольше порогового времени.

        Пропускается в режиме 'local' и когда сделок ещё не было.
        """
        if Config.MODE == "local" or self._last_trade_at is None:
            return
        hours_idle = (time.time() - self._last_trade_at) / _SECONDS_PER_HOUR
        if (
            hours_idle >= self._silent_death_hours
            and time.time() - self._silent_death_alerted_at
            > SILENT_DEATH_ALERT_COOLDOWN
        ):
            self._silent_death_alerted_at = time.time()
            await self.telegram.notify(
                f"⚠️ Бот работает, но нет сделок уже "
                f"{hours_idle:.0f}ч. Проверьте сигналы и баланс AI."
            )

    async def trading_loop(self) -> None:
        """
        Основной гибридный цикл (30 сек).

        Точный sleep: учитывает время выполнения итерации.
        При ошибке ждёт 10 сек и продолжает.
        """
        self.is_running = True
        self._monitor_task = asyncio.create_task(self._monitor_positions())
        asyncio.create_task(self._season_check_loop())
        cycle = 0
        ai_status = "on" if self.ai.enabled else "off"
        scan_n = self._runtime_config.get_scan_top_n()
        logger.info(
            "Starting hybrid loop (interval=%ds, top=%d, ai=%s)",
            Config.TRADING_INTERVAL,
            scan_n,
            ai_status,
        )
        mode = "paper" if Config.PAPER_TRADING else "live"
        self._runtime_config.ensure_first_start_date()
        start_balance = await self._get_balance_usdt()
        await self.telegram.notify(
            f"🤖 *BitbotBY запущен* [{mode}]\n"
            f"Стратегия: `{Config.DEFAULT_STRATEGY}`\n"
            f"Баланс: `${start_balance:,.2f}`\n"
            f"AI: {ai_status} | Символов: {scan_n}\n\n"
            f"Используй кнопки ниже или набери /help",
            reply_markup=_kb_main(),
        )
        await self._check_sac_model()
        await self._check_tune_reminder()

        while self.is_running:
            cycle += 1
            loop = asyncio.get_running_loop()
            t0 = loop.time()

            if self._runtime_config.is_paused():
                await asyncio.sleep(Config.TRADING_INTERVAL)
                continue

            try:
                _profile = self._runtime_config.get_market_profile()
                _scan_n = self._runtime_config.get_scan_top_n()
                if _profile == "bluechip":
                    symbols = await self.scanner.get_top_symbols(
                        n=_scan_n,
                        bluechip_bases=BLUECHIP_BASES,
                    )
                elif _profile == "altcoin":
                    symbols = await self.scanner.get_top_symbols(
                        n=_scan_n,
                        altcoin_exclude_bases=BLUECHIP_BASES,
                    )
                else:
                    symbols = await self.scanner.get_top_symbols(n=_scan_n)
                market_data = await self._scan_and_update_correlations(symbols)
                snapshots = await self._cycle.collect_snapshots(
                    list(market_data.keys()), market_data
                )
                balance = await self._get_balance_usdt()

                # ── Обновляем пик equity ───────────────────────────────────
                equity = (
                    self._live_equity_cache if not Config.PAPER_TRADING else balance
                )
                if equity > self._peak_balance:
                    self._peak_balance = equity

                # ── [П.2] Hard drawdown halt (с подтверждением) ───────────
                # Считаем по equity (включает unrealized PnL) — точнее чем
                # по свободному балансу. Подтверждение N циклов подряд.
                # В paper режиме: уведомляет но не останавливает торговлю.
                if (
                    self._peak_balance > 0
                    and (self._peak_balance - equity) / self._peak_balance
                    >= Config.MAX_DRAWDOWN_PERCENT
                ):
                    self._drawdown_consec += 1
                    dd_pct = (self._peak_balance - equity) / self._peak_balance * 100
                    confirm = self._runtime_config.get_drawdown_confirm_cycles()
                    logger.warning(
                        "Drawdown %.1f%% from peak $%.2f — confirm %d/%d",
                        dd_pct,
                        self._peak_balance,
                        self._drawdown_consec,
                        confirm,
                    )
                    if self._drawdown_consec >= confirm:
                        if Config.PAPER_TRADING:
                            logger.warning(
                                "[PAPER] Hard drawdown halt would trigger:"
                                " %.1f%% drawdown confirmed %d cycles",
                                dd_pct,
                                confirm,
                            )
                            await self.telegram.notify(
                                f"⚠️ *[PAPER] Hard drawdown halt*: просадка"
                                f" {dd_pct:.1f}% от пика"
                                f" ${self._peak_balance:.2f}"
                                f" подтверждена {confirm} цикла подряд.\n"
                                f"В live режиме: позиции закрыты,"
                                f" пауза {Config.DRAWDOWN_HALT_HOURS:.0f}ч."
                            )
                            self._drawdown_consec = 0
                        else:
                            halt_secs = Config.DRAWDOWN_HALT_HOURS * 3600
                            logger.critical(
                                "Hard drawdown halt: %.1f%% confirmed %d cycles"
                                " — closing losers, tightening winners, pausing %.0fh",
                                dd_pct,
                                confirm,
                                Config.DRAWDOWN_HALT_HOURS,
                            )
                            _dd_prices = {
                                snap["symbol"]: float(snap["price"])
                                for snap in snapshots
                                if snap
                            }
                            await self._position_monitor.close_losers_tighten_winners(
                                self._monitored,
                                self._monitored_lock,
                                _dd_prices,
                                reason="hard_drawdown",
                            )
                            await self.telegram.notify(
                                f"🚨 *Hard drawdown halt*: просадка {dd_pct:.1f}%"
                                f" от пика ${self._peak_balance:.2f}"
                                f" подтверждена {confirm} цикла подряд.\n"
                                f"Убыточные позиции закрыты."
                                f" Прибыльные: SL перенесён на breakeven.\n"
                                f"Новые входы заблокированы на"
                                f" {Config.DRAWDOWN_HALT_HOURS:.0f}ч."
                            )
                            self._drawdown_consec = 0
                            await asyncio.sleep(halt_secs)
                            self._peak_balance = await self._get_balance_usdt()
                            continue
                    else:
                        continue
                else:
                    self._drawdown_consec = 0

                # ── [П.3] Дневной лимит потерь ────────────────────────────
                # Считаем по equity (кэш + нереализованный P&L), а не по
                # свободному балансу — иначе открытие позиций само по себе
                # выглядит как «убыток» и лимит срабатывает мгновенно.
                # При срабатывании: блокируем новые входы, открытые позиции
                # держим до их SL/TP (PositionMonitor управляет ими независимо).
                # В paper режиме: уведомляет но не останавливает торговлю.
                if Config.PAPER_TRADING:
                    _paper_prices = {
                        snap["symbol"]: float(snap["price"])
                        for snap in snapshots
                        if snap
                    }
                    _equity_for_limit = (
                        await self.portfolio_manager.get_portfolio_value(_paper_prices)
                    )
                else:
                    _equity_for_limit = self._live_equity_cache
                if not self.risk_manager.check_daily_loss_limit(_equity_for_limit):
                    now_utc = datetime.utcnow()
                    midnight = (now_utc + timedelta(days=1)).replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                    sleep_secs = (midnight - now_utc).total_seconds()
                    if Config.PAPER_TRADING:
                        logger.warning(
                            "[PAPER] Daily loss limit would trigger:"
                            " equity $%.2f, would pause %.0f s",
                            _equity_for_limit,
                            sleep_secs,
                        )
                        _now = time.time()
                        if _now - self._daily_limit_notified_at >= 3600:
                            self._daily_limit_notified_at = _now
                            await self.telegram.notify(
                                f"⚠️ *[PAPER] Дневной лимит потерь*:"
                                f" equity ${_equity_for_limit:.2f}.\n"
                                f"В live режиме: новые входы заблокированы до 00:00 UTC"
                                f" ({int(sleep_secs // 3600)}ч"
                                f" {int((sleep_secs % 3600) // 60)}м)."
                            )
                    else:
                        await self.telegram.notify(
                            "⛔ Дневной лимит потерь достигнут. "
                            f"Equity: ${_equity_for_limit:.2f}.\n"
                            f"Новые входы заблокированы.\n"
                            f"Торговля возобновится автоматически при восстановлении"
                            f" equity или в 00:00 UTC"
                            f" (через {int(sleep_secs // 3600)}ч"
                            f" {int((sleep_secs % 3600) // 60)}м).\n"
                            f"Открытые позиции работают по своим SL/TP."
                        )
                        logger.warning(
                            "Daily loss limit hit — blocking new entries,"
                            " checking recovery every 60s until midnight UTC",
                        )
                        _check_interval = 60
                        _elapsed = 0.0
                        while _elapsed < sleep_secs:
                            await asyncio.sleep(_check_interval)
                            _elapsed += _check_interval
                            _recovery_eq = await self._get_balance_usdt()
                            _recovery_eq = self._live_equity_cache
                            if self.risk_manager.check_daily_loss_limit(_recovery_eq):
                                logger.info(
                                    "Daily loss limit recovered:"
                                    " equity $%.2f — resuming trading",
                                    _recovery_eq,
                                )
                                await self.telegram.notify(
                                    f"✅ Equity восстановилась до"
                                    f" ${_recovery_eq:.2f}.\n"
                                    f"Торговля возобновлена досрочно."
                                )
                                break
                        continue

                # ── [П.1] ATR spike / volatility circuit breaker ──────────
                # Если ATR вырос в ATR_SPIKE_MULT раз от нормы → закрыть
                # позиции и пропустить этот цикл (следующий через 30 с).
                if self._is_atr_spike(market_data):
                    logger.warning(
                        "Volatility circuit breaker triggered — skipping cycle"
                    )
                    await self._close_all_open_positions("atr_spike")
                    await self.telegram.notify(
                        "⚡ *Volatility circuit breaker*: скачок ATR в "
                        f"{Config.ATR_SPIKE_MULT:.0f}x от нормы.\n"
                        "Позиции закрыты, цикл пропущен."
                    )
                    continue

                regimes = await self._detect_regimes(market_data)
                recs = await self._generate_signals(
                    snapshots, balance, regimes, market_data
                )

                # БАГ 1: строим signals_map из реальных рекомендаций (после combine),
                # а не из снэпшотов, где top_signal/market_context не существуют.
                _signals_map = {r["symbol"]: r for r in recs if r.get("symbol")}
                _prices = {
                    sym: df["close"].iloc[-1]
                    for sym, df in market_data.items()
                    if df is not None and not df.empty
                }
                try:
                    _market_ctx_map = (
                        await self._market_context.get_context_for_symbols(
                            list(market_data.keys()), _prices
                        )
                    )
                except Exception as _mc_exc:
                    logger.warning("MarketContext fetch failed: %s", _mc_exc)
                    _market_ctx_map = {}
                self._position_monitor.update_market_state(
                    signals=_signals_map,
                    market_ctx=_market_ctx_map,
                    regime=self._current_regime,
                )

                snap_map = {s["symbol"]: s for s in snapshots}
                for rec in recs:
                    rec.setdefault("_snap", snap_map.get(rec.get("symbol")))

                filtered = self._filter_by_balance(recs, balance)
                filtered.sort(key=lambda x: x.get("confidence", 0), reverse=True)

                self._cycle.print_recommendations(filtered, balance, cycle)
                await self._cycle.notify_new_signals(filtered, balance, cycle)
                await self._execute_top_rec(filtered, market_data)
                await self._update_performance_stats()

                self.combiner.sac.reload_if_updated()
                await self._check_silent_death()
                self._save_monitored_state()

                cycles_counter.inc()
                elapsed = loop.time() - t0
                sleep_for = max(0, Config.TRADING_INTERVAL - elapsed)
                logger.info(
                    "Cycle #%d: %.1fs elapsed, sleeping %.1fs",
                    cycle,
                    elapsed,
                    sleep_for,
                )
                await asyncio.sleep(sleep_for)

            except Exception as e:
                # Перехват верхнего уровня: удерживает цикл живым при
                # временных сбоях (обрывы сети, техобслуживание биржи).
                # Специфические ошибки перехватываются ближе к источнику;
                # здесь обрабатывается всё остальное непредвиденное.
                logger.error("Error in trading loop: %s", e)
                await asyncio.sleep(10)

    async def stop(self) -> None:
        self.is_running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        try:
            stop_balance = await self._get_balance_usdt()
            await self.telegram.notify(
                f"⛔ *BitbotBY остановлен*\n" f"Баланс: `${stop_balance:,.2f}`"
            )
        except Exception:
            pass
        await self.telegram.stop()
        await self.api.close()
        await self.data_loader.close()
        logger.info("Trading bot stopped")
