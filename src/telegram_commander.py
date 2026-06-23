"""
Telegram-команды и inline-панель управления BitbotBY.

Все настройки доступны прямо из Telegram — пользователю не нужно
смотреть в репозиторий или перезапускать бота.

Команды:
  /start, /help  — главная панель
  /status        — текущий статус
  /settings      — все настройки (режим, символы, авто-сделки, пауза)
  /pause         — приостановить торговлю
  /resume        — возобновить торговлю
  /mode MODE     — сменить режим ai / local / hybrid / dqn
  /scan N        — кол-во сканируемых символов
  /trainn N      — кол-во символов для обучения SAC (1–100, default 20)
  /lev           — режим плеча: fixed | volatility | full
  /lev fixed     — фиксированное плечо из LEVERAGE
  /lev volatility — ATR-таргетинг (авто по волатильности)
  /lev full      — ATR + режим рынка + просадка
  /lev target N    — целевой риск на ATR (0.005–0.1, default 0.01)
  /lev drawdown N  — порог просадки для режима full (0.05–0.5, default 0.15)
  /provider      — показать текущий AI провайдер
  /provider groq — переключить на Groq (auto|anthropic|openai|deepseek|groq)
  /pnl           — P&L за день и всего
  /pos           — открытые позиции
  /add SYM       — добавить символ (напр. /add SOL)
  /remove SYM    — исключить символ из сканирования
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.ext import CallbackQueryHandler, CommandHandler, ContextTypes

    _TG_AVAILABLE = True
except ImportError:
    _TG_AVAILABLE = False


# ── Клавиатуры ────────────────────────────────────────────────────────────────


def _kb_main() -> "InlineKeyboardMarkup":
    """Главная панель — появляется после /start и /help."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("📊 Статус", callback_data="status"),
                InlineKeyboardButton("📈 P&L", callback_data="pnl"),
                InlineKeyboardButton("📋 Позиции", callback_data="pos"),
            ],
            [
                InlineKeyboardButton("⚙️ Настройки", callback_data="settings"),
                InlineKeyboardButton("❓ Справка", callback_data="help_menu"),
            ],
        ]
    )


def _kb_status(paused: bool) -> "InlineKeyboardMarkup":
    toggle = (
        InlineKeyboardButton("▶️ Возобновить", callback_data="resume")
        if paused
        else InlineKeyboardButton("⏸ Пауза", callback_data="pause")
    )
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("📈 P&L", callback_data="pnl"),
                InlineKeyboardButton("📋 Позиции", callback_data="pos"),
                InlineKeyboardButton("🔄 Обновить", callback_data="status"),
            ],
            [
                toggle,
                InlineKeyboardButton("⚙️ Настройки", callback_data="settings"),
            ],
            [InlineKeyboardButton("« Главная", callback_data="main")],
        ]
    )


def _kb_settings(paused: bool, auto_exec: bool) -> "InlineKeyboardMarkup":
    pause_btn = (
        InlineKeyboardButton("▶️ Возобновить торговлю", callback_data="resume")
        if paused
        else InlineKeyboardButton("⏸ Поставить на паузу", callback_data="pause")
    )
    exec_label = (
        "✅ Авто-сделки: ВКЛ  →  выкл" if auto_exec else "❌ Авто-сделки: ВЫКЛ  →  вкл"
    )
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(exec_label, callback_data="toggle_auto_exec")],
            [pause_btn],
            [
                InlineKeyboardButton("🤖 Режим торговли", callback_data="mode_menu"),
                InlineKeyboardButton("🔢 Кол-во символов", callback_data="scan_menu"),
            ],
            [InlineKeyboardButton("📐 Стратегии", callback_data="strategies")],
            [InlineKeyboardButton("⚖️ Риск-профиль", callback_data="risk_menu")],
            [InlineKeyboardButton("🕐 Часы торговли", callback_data="hours_info")],
            [InlineKeyboardButton("🤖 AI-провайдер", callback_data="provider_menu")],
            [InlineKeyboardButton("🔬 Тюнинг SAC (~2ч) + обучение", callback_data="tune_sac_menu")],
            [InlineKeyboardButton(
                "⏱ Таймаут подтверждения", callback_data="timeout_menu"
            )],
            [InlineKeyboardButton("🔄 Сброс настроек", callback_data="reset_defaults")],
            [InlineKeyboardButton("« Главная", callback_data="main")],
        ]
    )


def _kb_mode_menu() -> "InlineKeyboardMarkup":
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("🤖 AI", callback_data="mode:ai"),
                InlineKeyboardButton("📐 Local", callback_data="mode:local"),
            ],
            [
                InlineKeyboardButton("🔀 Hybrid", callback_data="mode:hybrid"),
                InlineKeyboardButton("🧠 SAC", callback_data="mode:dqn"),
            ],
            [InlineKeyboardButton("« Настройки", callback_data="settings")],
        ]
    )


def _kb_scan_menu() -> "InlineKeyboardMarkup":
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("10", callback_data="scan:10"),
                InlineKeyboardButton("20", callback_data="scan:20"),
                InlineKeyboardButton("30", callback_data="scan:30"),
                InlineKeyboardButton("50", callback_data="scan:50"),
            ],
            [
                InlineKeyboardButton("75", callback_data="scan:75"),
                InlineKeyboardButton("100", callback_data="scan:100"),
            ],
            [InlineKeyboardButton("« Настройки", callback_data="settings")],
        ]
    )


def _kb_help_menu() -> "InlineKeyboardMarkup":
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("📖 О боте", callback_data="help:about"),
                InlineKeyboardButton("💡 Режимы", callback_data="help:modes"),
            ],
            [
                InlineKeyboardButton("⚙️ Настройки", callback_data="help:settings"),
                InlineKeyboardButton("📋 Команды", callback_data="help:commands"),
            ],
            [
                InlineKeyboardButton("💰 Символы", callback_data="help:symbols"),
                InlineKeyboardButton("⚖️ Риски", callback_data="help:risk"),
            ],
            [InlineKeyboardButton("🔒 Безопасность", callback_data="help:security")],
            [InlineKeyboardButton("« Главная", callback_data="main")],
        ]
    )


def _kb_help_back() -> "InlineKeyboardMarkup":
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("« Справка", callback_data="help_menu"),
                InlineKeyboardButton("🏠 Главная", callback_data="main"),
            ],
        ]
    )


def _kb_timeout_menu(current: int) -> "InlineKeyboardMarkup":
    def _btn(sec: int) -> "InlineKeyboardButton":
        label = f"{'✅ ' if sec == current else ''}{sec}с"
        return InlineKeyboardButton(label, callback_data=f"timeout:{sec}")

    return InlineKeyboardMarkup(
        [
            [_btn(15), _btn(30), _btn(60)],
            [_btn(120), _btn(180), _btn(300)],
            [InlineKeyboardButton("« Настройки", callback_data="settings")],
        ]
    )


def _kb_tune_sac() -> "InlineKeyboardMarkup":
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ Запустить тюнинг (~3 ч)", callback_data="tune_sac_now"
                ),
                InlineKeyboardButton("⏭ Отмена", callback_data="tune_sac_skip"),
            ]
        ]
    )


def _kb_sac_train() -> "InlineKeyboardMarkup":
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ Обучить сейчас (~60 мин)", callback_data="train_sac_now"
                ),
                InlineKeyboardButton("⏭ Пропустить", callback_data="train_sac_skip"),
            ]
        ]
    )


def _kb_pnl() -> "InlineKeyboardMarkup":
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("📊 Статус", callback_data="status"),
                InlineKeyboardButton("📋 Позиции", callback_data="pos"),
                InlineKeyboardButton("🔄 Обновить", callback_data="pnl"),
            ],
            [InlineKeyboardButton("« Главная", callback_data="main")],
        ]
    )


def _kb_pos() -> "InlineKeyboardMarkup":
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("📊 Статус", callback_data="status"),
                InlineKeyboardButton("📈 P&L", callback_data="pnl"),
                InlineKeyboardButton("🔄 Обновить", callback_data="pos"),
            ],
            [InlineKeyboardButton("« Главная", callback_data="main")],
        ]
    )


def _kb_after_action() -> "InlineKeyboardMarkup":
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("📊 Статус", callback_data="status"),
                InlineKeyboardButton("⚙️ Настройки", callback_data="settings"),
            ]
        ]
    )


def _kb_risk_menu(drawdown_on: bool) -> "InlineKeyboardMarkup":
    dd_label = (
        "✅ Защита просадки: ВКЛ  →  выкл"
        if drawdown_on
        else "❌ Защита просадки: ВЫКЛ  →  вкл"
    )
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "🟢 Консерватив.", callback_data="risk:conservative"
                ),
                InlineKeyboardButton("🟡 Умеренный", callback_data="risk:moderate"),
                InlineKeyboardButton("🔴 Агрессивный", callback_data="risk:aggressive"),
            ],
            [InlineKeyboardButton(dd_label, callback_data="toggle_drawdown")],
            [InlineKeyboardButton("📊 Управление плечом →", callback_data="lev_menu")],
            [InlineKeyboardButton("« Настройки", callback_data="settings")],
        ]
    )


def _kb_hours_info() -> "InlineKeyboardMarkup":
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("🕐 1-10 (Азия)", callback_data="hours:1-10"),
                InlineKeyboardButton("🕐 8-20 (Европа)", callback_data="hours:8-20"),
            ],
            [
                InlineKeyboardButton("🕐 14-22 (США)", callback_data="hours:14-22"),
                InlineKeyboardButton("🕐 22-6 (ночь)", callback_data="hours:22-6"),
            ],
            [InlineKeyboardButton("🔄 24/7 (сброс)", callback_data="hours:")],
            [InlineKeyboardButton("« Настройки", callback_data="settings")],
        ]
    )


def _kb_lev_menu(mode: str, target: float, drawdown: float) -> "InlineKeyboardMarkup":
    def _m(label: str, val: str) -> "InlineKeyboardButton":
        tick = "✅ " if val == mode else ""
        return InlineKeyboardButton(f"{tick}{label}", callback_data=f"lev:mode:{val}")

    rows = [
        [_m("fixed", "fixed"), _m("volatility", "volatility"), _m("full", "full")],
        [
            InlineKeyboardButton(
                f"── Риск на ATR: {target*100:.1f}% ──", callback_data="noop"
            )
        ],
        [
            InlineKeyboardButton("0.5%", callback_data="lev:target:0.005"),
            InlineKeyboardButton("1%", callback_data="lev:target:0.01"),
            InlineKeyboardButton("1.5%", callback_data="lev:target:0.015"),
            InlineKeyboardButton("2%", callback_data="lev:target:0.02"),
            InlineKeyboardButton("3%", callback_data="lev:target:0.03"),
        ],
    ]
    if mode == "full":

        def _d(label: str, val: str) -> "InlineKeyboardButton":
            tick = "✅ " if abs(drawdown - float(val)) < 0.001 else ""
            return InlineKeyboardButton(
                f"{tick}{label}", callback_data=f"lev:drawdown:{val}"
            )

        rows += [
            [
                InlineKeyboardButton(
                    f"── Порог просадки: {drawdown*100:.0f}% ──", callback_data="noop"
                )
            ],
            [_d("10%", "0.1"), _d("15%", "0.15"), _d("20%", "0.2"), _d("25%", "0.25")],
        ]
    rows += [
        [InlineKeyboardButton("✏️ Ввести вручную", callback_data="lev:manual")],
        [InlineKeyboardButton("« Риск-профиль", callback_data="risk_menu")],
    ]
    return InlineKeyboardMarkup(rows)


def _kb_provider_menu(current: str) -> "InlineKeyboardMarkup":
    providers = [
        ("🆓 Groq", "groq"),
        ("Claude", "anthropic"),
        ("DeepSeek", "deepseek"),
        ("OpenAI", "openai"),
    ]

    def _p(label: str, val: str) -> "InlineKeyboardButton":
        tick = "✅ " if val == current else ""
        return InlineKeyboardButton(f"{tick}{label}", callback_data=f"provider:{val}")

    auto_tick = "✅ " if current == "auto" else ""
    return InlineKeyboardMarkup(
        [
            [_p(label, val) for label, val in providers],
            [
                InlineKeyboardButton(
                    f"{auto_tick}🔄 auto (цепочка)", callback_data="provider:auto"
                )
            ],
            [InlineKeyboardButton("« Настройки", callback_data="settings")],
        ]
    )


_RISK_ICON = {"low": "🟢", "medium": "🟡", "high": "🔴"}
_STRAT_SHORT = {
    "ema_crossover": "EMA↕",
    "rsi_momentum": "RSI",
    "macd_crossover": "MACD↕",
    "bollinger_bands": "BB",
    "scalping": "Scalp🔴",
    "swing_trading": "Swing",
    "breakout": "Break🔴",
    "mean_reversion": "Mean↩",
    "trend_following": "Trend🟢",
}


def _kb_strategies(strategies: list) -> "InlineKeyboardMarkup":
    """Кнопка на каждую стратегию — нажатие переключает вкл/выкл."""
    buttons = []
    row: list = []
    for i, s in enumerate(strategies):
        icon = "✅" if s["enabled"] else "❌"
        label = f"{icon} {_STRAT_SHORT.get(s['name'], s['name'])}"
        row.append(InlineKeyboardButton(label, callback_data=f"strat:{s['name']}"))
        if len(row) == 3 or i == len(strategies) - 1:
            buttons.append(row)
            row = []
    buttons.append(
        [
            InlineKeyboardButton("🔁 Все ВКЛ", callback_data="strat_reset"),
            InlineKeyboardButton("« Настройки", callback_data="settings"),
        ]
    )
    return InlineKeyboardMarkup(buttons)


# ── Тексты справки ────────────────────────────────────────────────────────────

_HELP_ABOUT = (
    "📖 *О боте BitbotBY*\n\n"
    "Гибридный крипто-трейдинг бот для Bybit Futures.\n\n"
    "*Как работает:*\n"
    "1️⃣ Сканирует топ-N монет по объёму каждые 30 сек\n"
    "2️⃣ Загружает OHLCV + технические индикаторы\n"
    "3️⃣ Анализирует через AI (Claude / DeepSeek / Groq)\n"
    "4️⃣ Комбинирует AI-сигнал с локальными стратегиями\n"
    "5️⃣ При AUTO_EXECUTE — спрашивает подтверждение в Telegram\n"
    "6️⃣ Управляет позицией: SL/TP, трейлинг-стоп, circuit breaker\n\n"
    "*Paper trading:* виртуальные сделки, реальные данные биржи\n"
    "*Live trading:* реальные ордера через Bybit API"
)

_HELP_MODES = (
    "💡 *Режимы торговли*\n\n"
    "🤖 *AI* — всё решает Claude / DeepSeek / Groq\n"
    "   Лучший режим при наличии API ключа\n\n"
    "📐 *Local* — локальные стратегии без AI\n"
    "   RSI, EMA, Bollinger, Breakout и др.\n"
    "   Работает без интернет-зависимостей\n\n"
    "🔀 *Hybrid* — AI + локальные стратегии совместно\n"
    "   Сигналы комбинируются и взвешиваются\n\n"
    "🧠 *SAC* — нейросеть (Soft Actor-Critic), обученная на истории\n"
    "   Требует предварительного обучения тренером\n\n"
    "Менять режим: кнопка *Режим торговли* в Настройках\n"
    "или команда `/mode ai`"
)

_HELP_SETTINGS = (
    "⚙️ *Настройки бота*\n\n"
    "🔄 *Авто-сделки* — исполнение ордеров\n"
    "   ВКЛ = бот торгует сам (с подтверждением в Telegram)\n"
    "   ВЫКЛ = только сигналы, без сделок\n\n"
    "⏸ *Пауза* — остановить новые сделки без перезапуска\n"
    "   Открытые позиции продолжают мониториться!\n\n"
    "🤖 *Режим* — AI / Local / Hybrid / SAC\n\n"
    "🔢 *Кол-во символов* — сколько монет сканировать\n\n"
    "📐 *Стратегии* — включить/выключить отдельные стратегии\n"
    "   Актуально для режимов Local и Hybrid\n\n"
    "⚖️ *Риск-профиль* — управление рисками:\n"
    "   Макс. позиций / Риск на сделку / Защита просадки\n"
    "   Пресеты: 🟢 Консервативный / 🟡 Умеренный / 🔴 Агрессивный\n\n"
    "🕐 *Часы торговли* — диапазон UTC когда открывать сделки\n"
    "   Пусто = 24/7. Формат: `9-17`, поддерживается `22-6`\n\n"
    "🔄 *Сброс настроек* — вернуть все параметры к .env\n\n"
    "Все настройки сохраняются в Redis и переживают перезапуск"
)

_HELP_COMMANDS = (
    "📋 *Команды бота*\n\n"
    "*Информация:*\n"
    "`/status`      — баланс, режим, позиции\n"
    "`/pnl`         — P&L за день и с начала\n"
    "`/pos`         — открытые позиции\n\n"
    "*Управление торговлей:*\n"
    "`/pause`       — поставить на паузу\n"
    "`/resume`      — возобновить торговлю\n"
    "`/mode ai`     — сменить режим (ai/local/hybrid/dqn)\n"
    "`/scan 50`     — сканировать 50 символов\n\n"
    "*Символы:*\n"
    "`/add SOL`     — всегда включать SOL в сканирование\n"
    "`/remove HYPE` — исключить HYPE из сканирования\n\n"
    "*Стратегии и риски:*\n"
    "`/strategies`  — меню включения/выключения стратегий\n"
    "`/risk`        — риск-профиль (позиции, % риска, просадка)\n"
    "`/hours 9-17`  — торговля только с 9 до 17 UTC\n"
    "`/hours`       — сброс на 24/7\n\n"
    "*Прочее:*\n"
    "`/settings`    — панель всех настроек\n"
    "`/help`        — эта справка\n\n"
    "💡 Все команды доступны также через кнопки в меню"
)

_HELP_SYMBOLS = (
    "💰 *Управление символами*\n\n"
    "По умолчанию бот сам выбирает топ-N монет по объёму за 24ч.\n\n"
    "*Добавить символ поверх топа:*\n"
    "`/add SOL` или `/add SOL/USDT`\n"
    "→ SOL сканируется всегда, даже если вылетел из топа\n\n"
    "*Исключить символ:*\n"
    "`/remove PEPE` или `/remove PEPE/USDT`\n"
    "→ PEPE пропускается при сканировании\n\n"
    "*Изменить кол-во сканируемых:*\n"
    "`/scan 50` или кнопка *Кол-во символов* в Настройках\n"
    "Диапазон: 1–200 символов\n\n"
    "*Часы торговли:*\n"
    "`/hours 9-17` — торговать только с 9:00 до 17:00 UTC\n"
    "`/hours 22-6` — поддерживается перенос через полночь\n"
    "`/hours` — сброс, торговать 24/7\n\n"
    "⚡ Все настройки сохраняются в Redis и переживают перезапуск бота"
)

_HELP_RISK = (
    "⚖️ *Стратегии и риск-менеджмент*\n\n"
    "*Стратегии* (`/strategies`):\n"
    "9 встроенных стратегий — каждую можно вкл/выкл:\n"
    "EMA↕ · RSI · MACD↕ · BB · Scalp🔴 · Swing · Break🔴 · Mean↩ · Trend🟢\n"
    "🟢 низкий риск · 🟡 средний · 🔴 высокий\n"
    "В режиме AI — передаются в промпт\n"
    "В режиме Local — выбираются автоматически по индикаторам\n\n"
    "*Риск-профиль* (`/risk`):\n"
    "🟢 Консервативный — 2 поз, 1%/сделку, защита просадки ВКЛ\n"
    "🟡 Умеренный       — 3 поз, 2%/сделку, защита просадки ВКЛ\n"
    "🔴 Агрессивный     — 5 поз, 4%/сделку, защита просадки ВЫКЛ\n\n"
    "*Защита просадки:*\n"
    "При просадке ≥10% от пика баланс позиция уменьшается вдвое\n\n"
    "⚡ Всё меняется на лету без перезапуска"
)

_HELP_SECURITY = (
    "🔒 *Безопасность*\n\n"
    "✅ *Что можно делать через Telegram:*\n"
    "   Менять режим, паузу, кол-во символов\n"
    "   Подтверждать/отклонять сделки\n"
    "   Смотреть баланс, позиции, P&L\n\n"
    "❌ *Что НЕ делается через Telegram:*\n"
    "   API ключи Bybit — только в .env файле\n"
    "   Ключи AI провайдеров — только в .env файле\n\n"
    "📍 *Почему?*\n"
    "Telegram не шифрует историю сообщений.\n"
    "Ключ биржи в чате = риск потерять деньги.\n"
    "Секреты живут только в .env на сервере.\n\n"
    "🛡 Только ты (твой chat_id) можешь управлять ботом"
)


class TelegramCommander:
    """
    Telegram-панель управления BitbotBY.

    Регистрирует команды и inline-кнопки на Application из TelegramNotifier.
    Состояние бота получает через callback get_state(), не держит ссылку на TradingBot.
    """

    def __init__(
        self,
        notifier: Any,
        runtime_config: Any,
        get_state: Callable[[], Dict[str, Any]],
    ) -> None:
        self._notifier = notifier
        self._rc = runtime_config
        self._get_state = get_state
        self._sac_training = False
        self._background_tasks: set = set()

    def register(self) -> None:
        if not _TG_AVAILABLE:
            logger.warning("TelegramCommander: python-telegram-bot не установлен")
            return
        app = getattr(self._notifier, "_app", None)
        if app is None:
            logger.warning("TelegramCommander: notifier._app не инициализирован")
            return

        commands = [
            ("start", self._cmd_help),
            ("help", self._cmd_help),
            ("settings", self._cmd_settings),
            ("status", self._cmd_status),
            ("pause", self._cmd_pause),
            ("resume", self._cmd_resume),
            ("mode", self._cmd_mode),
            ("scan", self._cmd_scan),
            ("pnl", self._cmd_pnl),
            ("pos", self._cmd_pos),
            ("add", self._cmd_add),
            ("remove", self._cmd_remove),
            ("strategies", self._cmd_strategies),
            ("hours", self._cmd_hours),
            ("risk", self._cmd_risk),
            ("trainn", self._cmd_trainn),
            ("lev", self._cmd_lev),
            ("provider", self._cmd_provider),
        ]
        for name, handler in commands:
            app.add_handler(CommandHandler(name, handler))
        app.add_handler(CallbackQueryHandler(self._handle_callback))

        bot_commands = [
            ("start", "Главная панель"),
            ("status", "Текущий статус бота"),
            ("settings", "Все настройки"),
            ("pnl", "Прибыль и убытки"),
            ("pos", "Открытые позиции"),
            ("pause", "Поставить на паузу"),
            ("resume", "Возобновить торговлю"),
            ("mode", "Сменить режим (ai/local/hybrid)"),
            ("lev", "Управление плечом"),
            ("provider", "AI провайдер"),
            ("risk", "Риск-профиль"),
            ("hours", "Часы торговли"),
            ("scan", "Топ монет по объёму"),
            ("add", "Добавить символ"),
            ("remove", "Удалить символ"),
            ("strategies", "Стратегии"),
            ("trainn", "Переобучить SAC"),
            ("help", "Справка"),
        ]
        try:
            import asyncio as _asyncio
            loop = _asyncio.get_event_loop()
            if loop.is_running():
                _asyncio.ensure_future(app.bot.set_my_commands(bot_commands))
            else:
                loop.run_until_complete(app.bot.set_my_commands(bot_commands))
        except Exception as e:
            logger.warning("set_my_commands failed: %s", e)
        logger.info("TelegramCommander: %d команд зарегистрировано", len(commands))

    # ── Auth & helpers ────────────────────────────────────────────────────────

    def _authorized(self, update: Any) -> bool:
        chat_id = str(getattr(update.effective_chat, "id", ""))
        return chat_id == str(self._notifier._chat_id)

    async def _reply(self, update: Any, text: str, kb: Any = None) -> None:
        if kb is None:
            kb = _kb_main()
        if update and update.message:
            await update.message.reply_text(
                text, parse_mode="Markdown", reply_markup=kb
            )

    async def _edit(self, query: Any, text: str, kb: Any = None) -> None:
        if kb is None:
            kb = _kb_main()
        try:
            await query.edit_message_text(text, parse_mode="Markdown", reply_markup=kb)
        except Exception as _e:
            if "message is not modified" not in str(_e).lower():
                raise

    # ── Builders ──────────────────────────────────────────────────────────────

    def _build_status(self) -> tuple[str, Any]:
        s = self._get_state()
        mode = self._rc.get_mode()
        n = self._rc.get_scan_top_n()
        paused = self._rc.is_paused()
        auto_exec = self._rc.get_auto_execute()
        balance = s.get("balance", 0.0)
        initial = s.get("initial_balance", balance)
        pnl_pct = (balance - initial) / initial * 100 if initial else 0.0
        positions = s.get("positions", [])
        paper = "paper" if s.get("paper_trading") else "live"
        forced = sorted(self._rc.get_forced_symbols())
        lev_mode = self._rc.get_leverage_mode()
        lev_target = self._rc.get_leverage_target_risk()
        provider = self._rc.get_ai_provider()
        hours = self._rc.get_trading_hours()

        state_icon = "⏸" if paused else "🟢"
        exec_icon = "✅" if auto_exec else "❌"
        hours_str = f"`{hours}`" if hours else "`24/7`"
        text = (
            f"{state_icon} *BitbotBY [{paper}]*\n\n"
            f"💰 Баланс: `${balance:,.2f}` ({pnl_pct:+.2f}%)\n"
            f"🤖 Режим: `{mode}` | AI: `{provider}`\n"
            f"🔢 Символов: `{n}` | Позиций: `{len(positions)}`\n"
            f"📊 Плечо: `{lev_mode}` ({lev_target*100:.1f}% ATR)\n"
            f"🕐 Часы: {hours_str}\n"
            f"{exec_icon} Авто-сделки: `{'ВКЛ' if auto_exec else 'ВЫКЛ'}`\n"
        )
        if forced:
            text += f"➕ Доп. символы: `{', '.join(forced)}`\n"
        if paused:
            text += "\n⏸ *Торговля приостановлена*"
        return text, _kb_status(paused)

    def _build_settings(self) -> tuple[str, Any]:
        paused = self._rc.is_paused()
        auto_exec = self._rc.get_auto_execute()
        mode = self._rc.get_mode()
        n = self._rc.get_scan_top_n()
        excluded = sorted(self._rc.get_excluded_symbols())
        lev_mode = self._rc.get_leverage_mode()
        lev_target = self._rc.get_leverage_target_risk()
        lev_dd = self._rc.get_max_drawdown_percent()
        provider = self._rc.get_ai_provider()
        hours = self._rc.get_trading_hours()
        max_pos = self._rc.get_max_positions()
        risk_pt = self._rc.get_risk_per_trade()
        dd_scale = self._rc.get_drawdown_scale_enabled()
        train_n = self._rc.get_train_top_n()
        disabled_strats = self._rc.get_disabled_strategies()

        state_str = "⏸ на паузе" if paused else "🟢 торгует"
        exec_str = "✅" if auto_exec else "❌"
        dd_str = "✅" if dd_scale else "❌"
        hours_str = hours if hours else "24/7"
        lev_str = f"`{lev_mode}` ({lev_target*100:.1f}% ATR)"
        if lev_mode == "full":
            lev_str += f", просадка `{lev_dd*100:.0f}%`"
        strats_str = (
            f"❌ выкл: `{', '.join(sorted(disabled_strats))}`"
            if disabled_strats
            else "✅ все включены"
        )

        text = (
            f"⚙️ *Настройки бота*\n\n"
            f"*Статус:* {state_str}\n"
            f"*Режим торговли:* `{mode}`\n"
            f"*AI-провайдер:* `{provider}`\n\n"
            f"*Риск-менеджмент:*\n"
            f"  Макс. позиций: `{max_pos}`\n"
            f"  Риск на сделку: `{risk_pt*100:.1f}%`\n"
            f"  Защита просадки: {dd_str}\n"
            f"  Плечо: {lev_str}\n\n"
            f"*Сканирование:*\n"
            f"  Символов (торговля): `{n}`\n"
            f"  Символов (обучение SAC): `{train_n}`\n\n"
            f"*Стратегии:* {strats_str}\n"
            f"*Часы торговли:* `{hours_str}`\n"
            f"*Авто-сделки:* {exec_str}\n"
        )
        if excluded:
            text += f"*Исключены из скан.:* `{', '.join(excluded)}`\n"
        text += "\n_Нажми кнопку для изменения:_"
        return text, _kb_settings(paused, auto_exec)

    def _build_risk(self) -> tuple[str, Any]:
        r = self._rc.get_risk_summary()
        max_pos = r["max_positions"]
        rpt = r["risk_per_trade"]
        dd = r["drawdown_scale_enabled"]
        dd_icon = "✅" if dd else "❌"
        text = (
            f"⚖️ *Риск-профиль*\n\n"
            f"Макс. позиций: `{max_pos}`\n"
            f"Риск на сделку: `{rpt * 100:.1f}%`\n"
            f"Защита просадки: {dd_icon}\n\n"
            f"_Пресеты:_\n"
            f"🟢 Конс. — 2 поз, 1%, защита ВКЛ\n"
            f"🟡 Умер. — 3 поз, 2%, защита ВКЛ\n"
            f"🔴 Агр. — 5 поз, 4%, защита ВЫКЛ\n"
        )
        return text, _kb_risk_menu(dd)

    def _build_hours_text(self) -> str:
        hours = self._rc.get_trading_hours()
        if hours:
            active = f"`{hours}` UTC"
            status = "🕐 Торговля ограничена по времени"
        else:
            active = "24/7"
            status = "🟢 Торговля без ограничений"
        return (
            f"🕐 *Часы торговли*\n\n"
            f"{status}\n"
            f"Текущий диапазон: {active}\n\n"
            f"Формат: `ЧЧ-ЧЧ` UTC (пример: `9-17`)\n"
            f"Поддерживается перенос через полночь: `22-6`\n\n"
            f"_Команда: `/hours 9-17` или `/hours` для сброса_"
        )

    def _build_strategies(self) -> tuple[str, Any]:
        strats = self._get_state().get("strategies", [])
        total = len(strats)
        enabled_count = sum(1 for s in strats if s["enabled"])
        mode = self._rc.get_mode()
        note = (
            "\n_В режиме AI стратегии передаются в промпт._"
            if mode != "local"
            else "\n_В режиме Local стратегия выбирается автоматически._"
        )
        lines = [f"📐 *Стратегии* — включено {enabled_count}/{total}\n"]
        for s in strats:
            icon = "✅" if s["enabled"] else "❌"
            risk = _RISK_ICON.get(s.get("risk_level", "medium"), "🟡")
            mtype = s.get("market_type", "any")
            lines.append(f"{icon} `{s['name']}` {risk} _{mtype}_")
        text = "\n".join(lines) + note
        return text, _kb_strategies(strats)

    def _build_pnl(self) -> str:
        s = self._get_state()
        balance = s.get("balance", 0.0)
        initial = s.get("initial_balance", balance)
        total = balance - initial
        total_p = (total / initial * 100) if initial else 0.0
        daily_p = s.get("daily_pnl_pct", 0.0)
        icon = "📈" if total >= 0 else "📉"
        return (
            f"{icon} *P&L отчёт*\n\n"
            f"Баланс: `${balance:,.2f}`\n"
            f"Старт:  `${initial:,.2f}`\n"
            f"Итого:  `{total:+.2f}$` ({total_p:+.2f}%)\n"
            f"За день: `{daily_p:+.2f}%`\n"
        )

    def _build_pos(self) -> str:
        positions = self._get_state().get("positions", [])
        if not positions:
            return "📭 Открытых позиций нет."
        lines = ["*Открытые позиции:*\n"]
        for p in positions:
            sym = p.get("symbol", "?")
            side = p.get("side", "?").upper()
            entry = p.get("entry_price", 0.0)
            size = p.get("size", 0.0)
            pnl = p.get("unrealized_pnl", 0.0)
            icon = "🟢" if pnl >= 0 else "🔴"
            lines.append(
                f"{icon} `{sym}` {side}  "
                f"qty `{size}`  entry `{entry:.4f}`  "
                f"PnL `{pnl:+.2f}$`"
            )
        return "\n".join(lines)

    # ── Команды ───────────────────────────────────────────────────────────────

    async def _cmd_help(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._authorized(update):
            return
        text, kb = self._build_status()
        intro = "👋 *Добро пожаловать в BitbotBY!*\n\n" + text
        await self._reply(update, intro, _kb_main())

    async def _cmd_settings(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._authorized(update):
            return
        text, kb = self._build_settings()
        await self._reply(update, text, kb)

    async def _cmd_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._authorized(update):
            return
        text, kb = self._build_status()
        await self._reply(update, text, kb)

    async def _cmd_pause(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._authorized(update):
            return
        self._rc.set_paused(True)
        await self._reply(
            update,
            "⏸ *Торговля приостановлена.*\n\n"
            "Открытые позиции продолжают мониториться (SL/TP работают).\n"
            "Новые сделки не открываются.",
            InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton("▶️ Возобновить", callback_data="resume"),
                        InlineKeyboardButton("📊 Статус", callback_data="status"),
                    ]
                ]
            ),
        )

    async def _cmd_resume(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._authorized(update):
            return
        self._rc.set_paused(False)
        await self._reply(
            update,
            "▶️ *Торговля возобновлена.*",
            InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton("⏸ Пауза", callback_data="pause"),
                        InlineKeyboardButton("📊 Статус", callback_data="status"),
                    ]
                ]
            ),
        )

    async def _cmd_mode(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._authorized(update):
            return
        args = context.args or []
        if not args:
            current = self._rc.get_mode()
            await self._reply(
                update,
                f"Текущий режим: `{current}`\nВыбери новый:",
                _kb_mode_menu(),
            )
            return
        mode = args[0].lower()
        if self._rc.set_mode(mode):
            await self._reply(
                update,
                f"✅ Режим → `{mode}`\nПрименится на следующем цикле.",
                _kb_after_action(),
            )
        else:
            await self._reply(
                update,
                "❌ Неверный режим. Доступны: `ai`, `local`, `hybrid`, `dqn`",
                _kb_mode_menu(),
            )

    async def _cmd_scan(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._authorized(update):
            return
        args = context.args or []
        if not args:
            n = self._rc.get_scan_top_n()
            await self._reply(
                update,
                f"Сейчас: `{n}` символов\nВыбери или введи `/scan N`:",
                _kb_scan_menu(),
            )
            return
        try:
            n = int(args[0])
        except ValueError:
            await self._reply(update, "❌ Укажи число: `/scan 50`", _kb_scan_menu())
            return
        if self._rc.set_scan_top_n(n):
            await self._reply(
                update, f"✅ Теперь сканируется `{n}` символов.", _kb_after_action()
            )
        else:
            await self._reply(update, "❌ Диапазон: 1–200", _kb_scan_menu())

    async def _cmd_trainn(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._authorized(update):
            return
        args = context.args or []
        n = self._rc.get_train_top_n()
        if not args:
            await self._reply(
                update,
                f"Сейчас для обучения SAC используется *{n}* символов.\n"
                f"Введи `/trainn N` чтобы изменить (1–100).\n\n"
                f"_Применится при следующем запуске обучения._",
            )
            return
        try:
            n = int(args[0])
        except ValueError:
            await self._reply(update, "❌ Укажи число: `/trainn 30`")
            return
        if self._rc.set_train_top_n(n):
            await self._reply(
                update,
                f"✅ Обучение SAC будет использовать *{n}* символов.\n"
                f"_Применится при следующем запуске обучения._",
                _kb_after_action(),
            )
        else:
            await self._reply(update, "❌ Диапазон: 1–100")

    async def _cmd_lev(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._authorized(update):
            return
        args = context.args or []

        if not args:
            mode = self._rc.get_leverage_mode()
            risk = self._rc.get_leverage_target_risk()
            dd = self._rc.get_max_drawdown_percent()
            mode_desc = {
                "fixed": "Фиксированное (LEVERAGE из .env)",
                "volatility": "ATR-таргетинг (авто по волатильности) ✅",
                "full": "ATR + режим рынка + просадка",
            }.get(mode, mode)
            dd_info = (
                f"\nПорог просадки: `{dd:.0%}` — при достижении плечо снижается до 30%"
                if mode == "full"
                else ""
            )
            await self._reply(
                update,
                f"📊 *Управление плечом*\n\n"
                f"Режим: *{mode}*\n{mode_desc}\n"
                f"Целевой риск на ATR: `{risk:.3f}` ({risk*100:.1f}%)"
                f"{dd_info}\n\n"
                f"*Команды:*\n"
                f"`/lev fixed` — всегда фиксированное плечо из LEVERAGE\n"
                f"`/lev volatility` — авто под волатильность монеты\n"
                f"`/lev full` — авто + режим рынка + защита при просадке\n"
                f"`/lev target 0.02` — целевой риск на ATR (0.5%–10%)\n"
                f"`/lev drawdown 0.15` — порог просадки для режима full (5%–50%)",
            )
            return

        sub = args[0].lower()

        if sub in ("fixed", "volatility", "full"):
            if self._rc.set_leverage_mode(sub):
                desc = {
                    "fixed": "фиксированное плечо из LEVERAGE",
                    "volatility": "ATR-таргетинг",
                    "full": "ATR + режим рынка + просадка",
                }[sub]
                extra = (
                    f"\n_Текущий порог просадки: "
                    f"{self._rc.get_max_drawdown_percent():.0%} "
                    f"(изменить: `/lev drawdown 0.15`)_"
                    if sub == "full"
                    else ""
                )
                await self._reply(
                    update,
                    f"✅ Режим плеча: *{sub}* ({desc}){extra}\n"
                    f"_Применится к следующим ордерам._",
                    _kb_after_action(),
                )
            else:
                await self._reply(update, "❌ Ошибка сохранения")
            return

        if sub == "target" and len(args) >= 2:
            try:
                risk = float(args[1])
            except ValueError:
                await self._reply(update, "❌ Укажи число: `/lev target 0.02`")
                return
            if self._rc.set_leverage_target_risk(risk):
                await self._reply(
                    update,
                    f"✅ Целевой риск на ATR: `{risk:.3f}` ({risk*100:.1f}%)\n"
                    f"_Применится к следующим ордерам._",
                    _kb_after_action(),
                )
            else:
                await self._reply(update, "❌ Диапазон: 0.005–0.1 (0.5%–10%)")
            return

        if sub == "drawdown" and len(args) >= 2:
            try:
                pct = float(args[1])
            except ValueError:
                await self._reply(update, "❌ Укажи число: `/lev drawdown 0.15`")
                return
            if self._rc.set_max_drawdown_percent(pct):
                await self._reply(
                    update,
                    f"✅ Порог просадки: `{pct:.0%}`\n"
                    f"При просадке баланса ≥ {pct:.0%} плечо снизится до 30%.\n"
                    f"_Работает только в режиме_ `full`.\n"
                    f"_Применится к следующим ордерам._",
                    _kb_after_action(),
                )
            else:
                await self._reply(update, "❌ Диапазон: 0.05–0.5 (5%–50%)")
            return

        await self._reply(
            update,
            "❌ Неизвестная команда.\n"
            "Используй: `/lev fixed` | `/lev volatility` | `/lev full`\n"
            "`/lev target 0.02` | `/lev drawdown 0.15`",
        )

    async def _cmd_provider(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._authorized(update):
            return
        args = context.args or []

        _PROVIDERS = {
            "auto": "Авто (Claude → OpenAI → DeepSeek → Groq)",
            "anthropic": "Claude (Anthropic) — лучшее качество",
            "openai": "ChatGPT (OpenAI)",
            "deepseek": "DeepSeek — самый дешёвый",
            "groq": "Groq / Llama 3.3 70B — бесплатный, быстрый",
        }

        if not args:
            current = self._rc.get_ai_provider()
            desc = _PROVIDERS.get(current, current)
            lines = "\n".join(f"`/provider {k}` — {v}" for k, v in _PROVIDERS.items())
            await self._reply(
                update,
                f"🤖 *AI провайдер*\n\n"
                f"Текущий: *{current}* ({desc})\n\n"
                f"*Доступные:*\n{lines}",
            )
            return

        provider = args[0].lower()
        if self._rc.set_ai_provider(provider):
            desc = _PROVIDERS.get(provider, provider)
            await self._reply(
                update,
                f"✅ AI провайдер: *{provider}*\n{desc}\n"
                f"_Применится к следующему циклу анализа._",
                _kb_after_action(),
            )
        else:
            opts = " | ".join(f"`{k}`" for k in _PROVIDERS)
            await self._reply(update, f"❌ Неизвестный провайдер.\nДоступные: {opts}")

    async def _cmd_pnl(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._authorized(update):
            return
        await self._reply(update, self._build_pnl(), _kb_pnl())

    async def _cmd_pos(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._authorized(update):
            return
        await self._reply(update, self._build_pos(), _kb_pos())

    async def _cmd_add(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._authorized(update):
            return
        args = context.args or []
        if not args:
            await self._reply(update, "Использование: `/add SOL` или `/add SOL/USDT`")
            return
        sym = args[0].upper()
        if "/" not in sym:
            sym = f"{sym}/USDT"
        self._rc.add_forced_symbol(sym)
        self._rc.remove_excluded_symbol(sym)
        await self._reply(
            update, f"✅ `{sym}` добавлен в список сканирования.", _kb_after_action()
        )

    async def _cmd_remove(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._authorized(update):
            return
        args = context.args or []
        if not args:
            await self._reply(
                update, "Использование: `/remove HYPE` или `/remove HYPE/USDT`"
            )
            return
        sym = args[0].upper()
        if "/" not in sym:
            sym = f"{sym}/USDT"
        self._rc.remove_forced_symbol(sym)
        self._rc.add_excluded_symbol(sym)
        await self._reply(
            update, f"✅ `{sym}` исключён из сканирования.", _kb_after_action()
        )

    async def _cmd_strategies(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._authorized(update):
            return
        text, kb = self._build_strategies()
        await self._reply(update, text, kb)

    async def _cmd_risk(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._authorized(update):
            return
        text, kb = self._build_risk()
        await self._reply(update, text, kb)

    async def _cmd_hours(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._authorized(update):
            return
        args = context.args or []
        if not args:
            await self._reply(update, self._build_hours_text(), _kb_hours_info())
            return
        val = args[0].strip() if args else ""
        if not self._rc.set_trading_hours(val):
            await self._reply(
                update,
                "❌ Неверный формат. Примеры: `/hours 9-17` или `/hours` для сброса",
            )
            return
        label = "24/7 (без ограничений)" if not val or val == "0" else f"`{val}` UTC"
        await self._reply(
            update,
            f"✅ Часы торговли: {label}",
            _kb_after_action(),
        )

    # ── Callback-handler (все кнопки) ─────────────────────────────────────────

    async def _run_sac_training(self) -> None:
        """Запускает train_sac.py как subprocess, не блокируя event loop."""
        import asyncio as _asyncio
        import os
        try:
            env = {**os.environ, "PYTHONPATH": "/app"}
            proc = await _asyncio.create_subprocess_exec(
                "python", "reinforcement_learning/train_sac.py",
                env=env,
                stdout=_asyncio.subprocess.DEVNULL,
                stderr=_asyncio.subprocess.PIPE,
            )
            await self._notifier.notify(
                "🧠 *Обучение SAC запущено*\n\n"
                "Процесс идёт в фоне, бот продолжает торговать.\n"
                "⏱ Ожидаемое время: ~60 мин на CPU.\n"
                "_Пришлю уведомление когда модель будет готова._"
            )
            _, stderr = await proc.communicate()
            if proc.returncode == 0:
                await self._notifier.notify(
                    "✅ *SAC модель обучена!*\n\n"
                    "Модель сохранена в `models/sac_model.zip`.\n"
                    "Бот автоматически подхватит её в следующем цикле.",
                    reply_markup=_kb_main(),
                )
            else:
                err = stderr.decode()[-300:] if stderr else "нет деталей"
                await self._notifier.notify(
                    f"❌ *Ошибка обучения SAC*\n\n`{err}`\n\nПроверьте: `make logs`",
                    reply_markup=_kb_main(),
                )
        except Exception as e:
            await self._notifier.notify(f"❌ SAC обучение упало: {e}")
        finally:
            self._sac_training = False

    async def _run_sac_tune_and_train(self) -> None:
        """Запускает tune_sac.py, затем train_sac.py последовательно."""
        import asyncio as _asyncio
        import os
        env = {**os.environ, "PYTHONPATH": "/app"}
        try:
            # ── Этап 1: тюнинг ───────────────────────────────────────────────
            proc = await _asyncio.create_subprocess_exec(
                "python", "reinforcement_learning/tune_sac.py",
                env=env,
                stdout=_asyncio.subprocess.DEVNULL,
                stderr=_asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                err = stderr.decode()[-300:] if stderr else "нет деталей"
                await self._notifier.notify(
                    f"❌ *Тюнинг завершился с ошибкой*\n\n`{err}`",
                    reply_markup=_kb_main(),
                )
                return
            await self._notifier.notify(
                "✅ *Тюнинг завершён!*\n\n"
                "Лучшие параметры сохранены.\n"
                "⏳ Запускаю обучение SAC (~60 мин)..."
            )
            # ── Этап 2: обучение с найденными параметрами ────────────────────
            proc2 = await _asyncio.create_subprocess_exec(
                "python", "reinforcement_learning/train_sac.py",
                env=env,
                stdout=_asyncio.subprocess.DEVNULL,
                stderr=_asyncio.subprocess.PIPE,
            )
            _, stderr2 = await proc2.communicate()
            if proc2.returncode == 0:
                await self._notifier.notify(
                    "✅ *Модель обновлена с лучшими параметрами!*\n\n"
                    "SAC обучен на результатах тюнинга.\n"
                    "Бот автоматически подхватит новую модель.",
                    reply_markup=_kb_main(),
                )
            else:
                err2 = stderr2.decode()[-300:] if stderr2 else "нет деталей"
                await self._notifier.notify(
                    f"❌ *Ошибка обучения после тюнинга*\n\n`{err2}`",
                    reply_markup=_kb_main(),
                )
        except Exception as e:
            await self._notifier.notify(f"❌ Тюнинг упал: {e}")
        finally:
            self._sac_training = False

    async def _handle_callback(
        self, update: Update, _context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        await query.answer()

        chat_id = str(getattr(update.effective_chat, "id", ""))
        if chat_id != str(self._notifier._chat_id):
            return

        data = query.data or ""

        # ── Навигация ────────────────────────────────────────────────────────
        if data == "main":
            text, _ = self._build_status()
            await self._edit(query, "🏠 *Главная панель*\n\n" + text, _kb_main())

        elif data == "status":
            text, kb = self._build_status()
            await self._edit(query, text, kb)

        elif data == "pnl":
            await self._edit(query, self._build_pnl(), _kb_pnl())

        elif data == "pos":
            await self._edit(query, self._build_pos(), _kb_pos())

        elif data == "settings":
            text, kb = self._build_settings()
            await self._edit(query, text, kb)

        # ── Управление ───────────────────────────────────────────────────────
        elif data == "pause":
            self._rc.set_paused(True)
            text, kb = self._build_settings()
            await self._edit(query, "⏸ *Пауза включена*\n\n" + text, kb)

        elif data == "resume":
            self._rc.set_paused(False)
            text, kb = self._build_settings()
            await self._edit(query, "▶️ *Торговля возобновлена*\n\n" + text, kb)

        elif data == "toggle_auto_exec":
            self._rc.set_auto_execute(not self._rc.get_auto_execute())
            text, kb = self._build_settings()
            await self._edit(query, text, kb)

        elif data == "mode_menu":
            mode = self._rc.get_mode()
            await self._edit(
                query, f"Текущий режим: `{mode}`\nВыбери новый:", _kb_mode_menu()
            )

        elif data == "scan_menu":
            n = self._rc.get_scan_top_n()
            await self._edit(query, f"Сейчас: `{n}` символов\nВыбери:", _kb_scan_menu())

        elif data.startswith("mode:"):
            mode = data.split(":", 1)[1]
            if self._rc.set_mode(mode):
                text, kb = self._build_settings()
                if mode == "local":
                    hint = (
                        "✅ Режим → `local`\n\n"
                        "💡 Стратегия выбирается автоматически по индикаторам. "
                        "Можно отключить ненужные через меню ниже.\n\n"
                    )
                    kb_local = InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton(
                                    "📐 Настроить стратегии →",
                                    callback_data="strategies",
                                )
                            ],
                            [
                                InlineKeyboardButton(
                                    "⚙️ К настройкам", callback_data="settings"
                                )
                            ],
                        ]
                    )
                    await self._edit(query, hint + text, kb_local)
                else:
                    await self._edit(query, f"✅ Режим → `{mode}`\n\n" + text, kb)
            else:
                await self._edit(query, "❌ Неверный режим", _kb_mode_menu())

        elif data.startswith("scan:"):
            try:
                n = int(data.split(":", 1)[1])
            except ValueError:
                return
            if self._rc.set_scan_top_n(n):
                text, kb = self._build_settings()
                await self._edit(query, f"✅ Символов → `{n}`\n\n" + text, kb)
            else:
                await self._edit(query, "❌ Недопустимое значение", _kb_scan_menu())

        # ── Справка ──────────────────────────────────────────────────────────
        elif data == "help_menu":
            await self._edit(
                query,
                "❓ *Справка по боту*\n\nВыбери раздел:",
                _kb_help_menu(),
            )

        elif data == "help:about":
            await self._edit(query, _HELP_ABOUT, _kb_help_back())

        elif data == "help:modes":
            await self._edit(query, _HELP_MODES, _kb_help_back())

        elif data == "help:settings":
            await self._edit(query, _HELP_SETTINGS, _kb_help_back())

        elif data == "help:commands":
            await self._edit(query, _HELP_COMMANDS, _kb_help_back())

        elif data == "help:symbols":
            await self._edit(query, _HELP_SYMBOLS, _kb_help_back())

        elif data == "help:risk":
            await self._edit(query, _HELP_RISK, _kb_help_back())

        elif data == "help:security":
            await self._edit(query, _HELP_SECURITY, _kb_help_back())

        # ── Сброс к .env ─────────────────────────────────────────────────────
        elif data == "reset_defaults":
            self._rc.reset_to_defaults()
            text, kb = self._build_settings()
            await self._edit(
                query,
                "🔄 *Настройки сброшены к значениям по умолчанию*\n\n" + text,
                kb,
            )

        # ── Риск-профиль ─────────────────────────────────────────────────────
        elif data == "risk_menu":
            text, kb = self._build_risk()
            await self._edit(query, text, kb)

        elif data.startswith("risk:"):
            preset = data.split(":", 1)[1]
            if self._rc.apply_risk_preset(preset):
                text, kb = self._build_risk()
                names = {
                    "conservative": "🟢 Консервативный",
                    "moderate": "🟡 Умеренный",
                    "aggressive": "🔴 Агрессивный",
                }
                label = names.get(preset, preset)
                await self._edit(query, f"✅ {label}\n\n" + text, kb)

        elif data == "toggle_drawdown":
            current = self._rc.get_drawdown_scale_enabled()
            self._rc.set_drawdown_scale_enabled(not current)
            text, kb = self._build_risk()
            icon = "✅" if not current else "❌"
            await self._edit(query, f"{icon} Защита просадки изменена\n\n" + text, kb)

        # ── Часы торговли ────────────────────────────────────────────────────
        elif data == "hours_info":
            await self._edit(query, self._build_hours_text(), _kb_hours_info())

        elif data.startswith("hours:"):
            val = data.split(":", 1)[1]
            self._rc.set_trading_hours(val)
            label = "24/7" if not val else f"`{val}` UTC"
            await self._edit(
                query,
                f"✅ Часы торговли: {label}\n\n" + self._build_hours_text(),
                _kb_hours_info(),
            )

        # ── Плечо ────────────────────────────────────────────────────────────
        elif data == "lev_menu":
            mode = self._rc.get_leverage_mode()
            target = self._rc.get_leverage_target_risk()
            dd = self._rc.get_max_drawdown_percent()
            mode_desc = {
                "fixed": "Фиксированное плечо из LEVERAGE",
                "volatility": "ATR-таргетинг (авто по волатильности)",
                "full": "ATR + режим рынка + защита просадки",
            }.get(mode, mode)
            text = (
                f"📊 *Управление плечом*\n\n"
                f"Режим: *{mode}* — {mode_desc}\n"
                f"Риск на ATR: `{target*100:.1f}%`\n"
            )
            if mode == "full":
                text += f"Порог просадки: `{dd*100:.0f}%`\n"
            text += "\n_Выбери режим или параметр:_"
            await self._edit(query, text, _kb_lev_menu(mode, target, dd))

        elif data.startswith("lev:mode:"):
            val = data.split(":", 2)[2]
            self._rc.set_leverage_mode(val)
            mode = val
            target = self._rc.get_leverage_target_risk()
            dd = self._rc.get_max_drawdown_percent()
            desc = {
                "fixed": "Фиксированное плечо из LEVERAGE",
                "volatility": "ATR-таргетинг (авто по волатильности)",
                "full": "ATR + режим рынка + защита просадки",
            }.get(val, val)
            await self._edit(
                query,
                f"✅ Режим плеча: *{val}*\n{desc}\n\n_Выбери режим или параметр:_",
                _kb_lev_menu(mode, target, dd),
            )

        elif data.startswith("lev:target:"):
            val_f = float(data.split(":", 2)[2])
            self._rc.set_leverage_target_risk(val_f)
            mode = self._rc.get_leverage_mode()
            dd = self._rc.get_max_drawdown_percent()
            await self._edit(
                query,
                f"✅ Риск на ATR: `{val_f*100:.1f}%`\n\n_Выбери режим или параметр:_",
                _kb_lev_menu(mode, val_f, dd),
            )

        elif data.startswith("lev:drawdown:"):
            val_f = float(data.split(":", 2)[2])
            self._rc.set_max_drawdown_percent(val_f)
            mode = self._rc.get_leverage_mode()
            target = self._rc.get_leverage_target_risk()
            await self._edit(
                query,
                f"✅ Порог просадки: `{val_f*100:.0f}%`\n\n_Выбери режим или параметр:_",
                _kb_lev_menu(mode, target, val_f),
            )

        elif data == "lev:manual":
            mode = self._rc.get_leverage_mode()
            target = self._rc.get_leverage_target_risk()
            dd = self._rc.get_max_drawdown_percent()
            await self._edit(
                query,
                "✏️ *Ручной ввод параметров плеча*\n\n"
                "Напиши команду в чат:\n"
                "`/lev target 0.015` — риск на ATR (0.005–0.1)\n"
                "`/lev drawdown 0.12` — порог просадки (0.05–0.5)\n\n"
                "_После отправки команды вернись в меню: /lev_",
                _kb_lev_menu(mode, target, dd),
            )

        elif data == "noop":
            await query.answer()

        # ── AI-провайдер ─────────────────────────────────────────────────────
        elif data == "provider_menu":
            current = self._rc.get_ai_provider()
            desc = {
                "auto": "Claude → OpenAI → DeepSeek → Groq",
                "groq": "Llama 3.3 70B — бесплатный, быстрый",
                "anthropic": "Claude — лучшее качество",
                "deepseek": "Самый дешёвый платный",
                "openai": "ChatGPT GPT-4o-mini",
            }.get(current, current)
            await self._edit(
                query,
                f"🤖 *AI-провайдер*\n\n"
                f"Текущий: *{current}*\n{desc}\n\n"
                f"_При ошибке основного — автоматически пробует следующий с ключом._",
                _kb_provider_menu(current),
            )

        elif data.startswith("provider:"):
            val = data.split(":", 1)[1]
            self._rc.set_ai_provider(val)
            desc = {
                "auto": "Claude → OpenAI → DeepSeek → Groq",
                "groq": "Llama 3.3 70B — бесплатный, быстрый",
                "anthropic": "Claude — лучшее качество",
                "deepseek": "Самый дешёвый платный",
                "openai": "ChatGPT GPT-4o-mini",
            }.get(val, val)
            await self._edit(
                query,
                f"✅ AI-провайдер: *{val}*\n{desc}\n\n"
                f"_Применится к следующему циклу анализа._",
                _kb_provider_menu(val),
            )

        # ── Стратегии ────────────────────────────────────────────────────────
        elif data == "strategies":
            text, kb = self._build_strategies()
            await self._edit(query, text, kb)

        elif data == "strat_reset":
            self._rc.reset_strategies()
            text, kb = self._build_strategies()
            await self._edit(query, "🔁 *Все стратегии включены*\n\n" + text, kb)

        elif data.startswith("strat:"):
            name = data.split(":", 1)[1]
            now_enabled = self._rc.toggle_strategy(name)
            icon = "✅" if now_enabled else "❌"
            text, kb = self._build_strategies()
            short = _STRAT_SHORT.get(name, name)
            state = "включена" if now_enabled else "отключена"
            await self._edit(query, f"{icon} `{short}` {state}\n\n" + text, kb)

        # ── Таймаут подтверждения ─────────────────────────────────────────────
        elif data == "timeout_menu":
            cur = self._rc.get_confirm_timeout()
            auto = self._rc.get_auto_execute()
            mode_hint = "авто-исполнение" if auto else "авто-пропуск"
            await self._edit(
                query,
                f"⏱ *Таймаут подтверждения сделки*\n\n"
                f"Сейчас: *{cur}с* → {mode_hint} если нет ответа\n\n"
                f"Выбери новое значение:",
                _kb_timeout_menu(cur),
            )

        elif data.startswith("timeout:"):
            try:
                sec = int(data.split(":", 1)[1])
            except (ValueError, IndexError):
                return
            ok = self._rc.set_confirm_timeout(sec)
            if ok:
                auto = self._rc.get_auto_execute()
                mode_hint = "авто-исполнение" if auto else "авто-пропуск"
                await self._edit(
                    query,
                    f"✅ Таймаут: *{sec}с* → {mode_hint}\n\n"
                    f"_Применится к следующей сделке._",
                    _kb_timeout_menu(sec),
                )

        # ── Тюнинг SAC ───────────────────────────────────────────────────────
        elif data == "tune_sac_menu":
            await self._edit(
                query,
                "🔬 *Тюнинг SAC*\n\n"
                "Optuna перебирает 30 вариантов параметров нейросети,\n"
                "затем обучает модель с лучшими из них.\n\n"
                "⏱ Общее время: ~3 часа на CPU.\n"
                "Бот продолжает торговать.\n\n"
                "_Рекомендуется запускать раз в 1-2 месяца._",
                _kb_tune_sac(),
            )

        elif data == "tune_sac_now":
            if self._sac_training:
                await self._edit(
                    query, "⏳ Обучение уже запущено, ожидайте...", _kb_after_action()
                )
                return
            self._sac_training = True
            await self._edit(
                query,
                "🔬 *Тюнинг SAC запущен...*\n\n"
                "⏱ Этап 1/2: поиск параметров (~2 ч)\n"
                "Бот продолжает торговать.\n"
                "Пришлю уведомления по каждому этапу.",
                _kb_after_action(),
            )
            import asyncio as _asyncio
            task = _asyncio.ensure_future(self._run_sac_tune_and_train())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        elif data == "tune_sac_skip":
            await self._edit(
                query,
                "⏭ *Тюнинг отменён*\n\n"
                "Вернуться к тюнингу можно в ⚙️ Настройки.",
                _kb_main(),
            )

        # ── SAC обучение ──────────────────────────────────────────────────────
        elif data == "train_sac_now":
            if self._sac_training:
                await self._edit(
                    query, "⏳ Обучение уже запущено, ожидайте...", _kb_after_action()
                )
                return
            self._sac_training = True
            await self._edit(
                query,
                "🚀 *Запускаю обучение SAC...*\n\n"
                "⏱ Это займёт ~60 мин на CPU.\n"
                "Бот продолжает торговать.\n"
                "Пришлю уведомление когда готово.",
                _kb_after_action(),
            )
            import asyncio as _asyncio
            task = _asyncio.ensure_future(self._run_sac_training())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        elif data == "train_sac_skip":
            await self._edit(
                query,
                "⏭ *Обучение пропущено*\n\n"
                "Бот работает на AI + локальных стратегиях.\n"
                "Запустить позже: /trainn",
                _kb_main(),
            )

        else:
            await query.answer("Неизвестная команда")
