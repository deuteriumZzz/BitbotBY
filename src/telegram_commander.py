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
  /mode MODE     — сменить режим ai / local / hybrid / sac
  /scan N        — кол-во сканируемых символов
  /trainn N      — кол-во символов для обучения SAC (1–100, default 20)
  /lev           — режим плеча: fixed | volatility | full
  /lev fixed     — фиксированное плечо из LEVERAGE
  /lev volatility — ATR-таргетинг (авто по волатильности)
  /lev full      — ATR + режим рынка + просадка
  /lev target N    — целевой риск на ATR (0.005–0.1, default 0.01)
  /lev drawdown N  — порог просадки для режима full (0.05–0.5, default 0.15)
  /provider      — показать текущий AI провайдер
  /provider groq — переключить на Groq (auto|anthropic|openai|deepseek|groq|gemini)
  /pnl           — P&L за день и всего
  /pos           — открытые позиции
  /closeall      — закрыть все открытые позиции (с подтверждением)
  /add SYM       — добавить символ (напр. /add SOL)
  /remove SYM    — исключить символ из сканирования
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

from config import Config
from src import ai_providers, login_link
from src.runtime_config import RISK_PRESETS

logger = logging.getLogger(__name__)

try:
    from telegram import (
        InlineKeyboardButton,
        InlineKeyboardMarkup,
        MenuButtonWebApp,
        Update,
        WebAppInfo,
    )
    from telegram.ext import (
        CallbackQueryHandler,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )

    _TG_AVAILABLE = True
except ImportError:
    _TG_AVAILABLE = False


# ── Клавиатуры ────────────────────────────────────────────────────────────────


def _webapp_url() -> str:
    """Адрес MiniApp из WEBAPP_URL. Telegram открывает только https — иначе пусто."""
    url = os.getenv("WEBAPP_URL", "").strip()
    return url if url.startswith("https://") else ""


def _kb_main(paper: "bool | None" = None) -> "InlineKeyboardMarkup":
    """Главная панель — появляется после /start и /help."""
    if paper is None:
        paper = Config.PAPER_TRADING
    mode_label = "📄 Режим: PAPER  →  Live" if paper else "💰 Режим: LIVE  →  Paper"
    url = _webapp_url()
    app_row = (
        [[InlineKeyboardButton("🖥 Открыть терминал", web_app=WebAppInfo(url=url))]]
        if url
        else []
    )
    return InlineKeyboardMarkup(
        app_row
        + [
            [
                InlineKeyboardButton("📊 Статус", callback_data="status"),
                InlineKeyboardButton("📈 P&L", callback_data="pnl"),
                InlineKeyboardButton("📋 Позиции", callback_data="pos"),
            ],
            [
                InlineKeyboardButton("⚙️ Настройки", callback_data="settings"),
                InlineKeyboardButton("❓ Справка", callback_data="help_menu"),
            ],
            [InlineKeyboardButton("🤖 AI-провайдеры", callback_data="provider_menu")],
            [InlineKeyboardButton(mode_label, callback_data="switch_trading_mode")],
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


def _kb_chronos_prompt() -> "InlineKeyboardMarkup":
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ Включить (рекомендуется)", callback_data="chronos_on"
                )
            ],
            [InlineKeyboardButton("⚡ Стандартный режим", callback_data="chronos_off")],
        ]
    )


def _kb_settings(
    paused: bool,
    auto_exec: bool,
    mode: str = "",
    chronos: bool = False,
    paper: bool = False,
    season_mode: str = "alert",
    sac_training: bool = False,
    is_tuning: bool = False,
) -> "InlineKeyboardMarkup":
    pause_btn = (
        InlineKeyboardButton("▶️ Возобновить торговлю", callback_data="resume")
        if paused
        else InlineKeyboardButton("⏸ Поставить на паузу", callback_data="pause")
    )
    exec_label = (
        "✅ Авто-сделки: ВКЛ  →  выкл" if auto_exec else "❌ Авто-сделки: ВЫКЛ  →  вкл"
    )
    mode_switch_label = (
        "📄 Режим: PAPER  →  Live" if paper else "💰 Режим: LIVE  →  Paper"
    )
    rows = [
        [InlineKeyboardButton(exec_label, callback_data="toggle_auto_exec")],
        [pause_btn],
        [
            InlineKeyboardButton("🤖 Режим торговли", callback_data="mode_menu"),
            InlineKeyboardButton("🔢 Кол-во символов", callback_data="scan_menu"),
        ],
        [InlineKeyboardButton("🎯 Профиль рынка", callback_data="market_profile_menu")],
        [
            InlineKeyboardButton(
                "💱 Рынок: Spot/Futures", callback_data="market_type_menu"
            )
        ],
        [
            InlineKeyboardButton(
                (
                    "🔔 Сезон: Алерт  →  авто"
                    if season_mode == "alert"
                    else "🤖 Сезон: Авто  →  алерт"
                ),
                callback_data="toggle_season_mode",
            )
        ],
        [InlineKeyboardButton("⚖️ Риск-профиль", callback_data="risk_menu")],
        [InlineKeyboardButton("🕐 Часы торговли", callback_data="hours_info")],
        [InlineKeyboardButton("🤖 AI-провайдер", callback_data="provider_menu")],
    ]
    if mode == "hybrid":
        chronos_label = (
            "🛡 Chronos: ВКЛ  →  выкл" if chronos else "⚡ Chronos: ВЫКЛ  →  вкл"
        )
        rows.append(
            [InlineKeyboardButton(chronos_label, callback_data="toggle_chronos")]
        )
    rows.append(
        [InlineKeyboardButton(mode_switch_label, callback_data="switch_trading_mode")]
    )
    if sac_training:
        train_btn = InlineKeyboardButton(
            "⏳ Идёт тюнинг SAC..." if is_tuning else "⏳ Идёт обучение SAC...",
            callback_data="train_progress",
        )
        rows.append([train_btn])
    else:
        rows.append(
            [
                InlineKeyboardButton(
                    "🧠 Обучить SAC (~60 мин)", callback_data="train_sac_now"
                ),
                InlineKeyboardButton(
                    "🔬 Тюнинг + обучение (~3ч)", callback_data="tune_sac_menu"
                ),
            ]
        )
    rows += [
        [InlineKeyboardButton("📊 Бэктест стратегий", callback_data="backtest_menu")],
        [InlineKeyboardButton("🔔 Уведомления", callback_data="alerts_menu")],
        [InlineKeyboardButton("⏱ Таймаут подтверждения", callback_data="timeout_menu")],
        [InlineKeyboardButton("📈 Прогресс обучения", callback_data="train_progress")],
        [InlineKeyboardButton("🔄 Сброс настроек", callback_data="reset_defaults")],
        [InlineKeyboardButton("« Главная", callback_data="main")],
    ]
    return InlineKeyboardMarkup(rows)


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


def _kb_market_profile_menu(
    current: str = "",
    bluechip_model: bool = False,
    altcoin_model: bool = False,
    meme_model: bool = False,
) -> "InlineKeyboardMarkup":
    import os as _os

    def _mark(name: str) -> str:
        return "✅ " if current == name else ""

    def _model_icon(exists: bool) -> str:
        return "🧠" if exists else "⚠️"

    bc_has = bluechip_model or _os.path.exists("models/sac_model.zip")
    alt_has = altcoin_model or _os.path.exists("models/sac_model_altcoin.zip")
    meme_has = meme_model or _os.path.exists("models/sac_model_meme.zip")

    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    f"{_mark('bluechip')}🔵 Блючипы {_model_icon(bc_has)}",
                    callback_data="market_profile:bluechip",
                ),
            ],
            [
                InlineKeyboardButton(
                    f"{_mark('altcoin')}🟡 Альткоины {_model_icon(alt_has)}",
                    callback_data="market_profile:altcoin",
                ),
            ],
            [
                InlineKeyboardButton(
                    f"{_mark('meme')}🚀 Мемкоины {_model_icon(meme_has)}",
                    callback_data="market_profile:meme",
                ),
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


def _kb_timeout_menu(
    current: int, auto_exec: bool = False, on_timeout: str = "execute"
) -> "InlineKeyboardMarkup":
    """Как открывать сделки: авто или с подтверждением (10–120 с), что без ответа."""

    def _btn(sec: int) -> "InlineKeyboardButton":
        selected = not auto_exec and sec == current
        label = f"{'✅ ' if selected else ''}{sec}с"
        return InlineKeyboardButton(label, callback_data=f"timeout:{sec}")

    def _on(value: str, label: str) -> "InlineKeyboardButton":
        tick = "✅ " if not auto_exec and on_timeout == value else ""
        return InlineKeyboardButton(tick + label, callback_data=f"on_timeout:{value}")

    auto_label = f"{'✅ ' if auto_exec else ''}🤖 Авто (без подтверждения)"
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(auto_label, callback_data="timeout:auto")],
            [_btn(15), _btn(30), _btn(60), _btn(120)],
            [
                _on("execute", "▶️ Без ответа — открыть"),
                _on("skip", "⏭ Без ответа — пропустить"),
            ],
            [InlineKeyboardButton("« Настройки", callback_data="settings")],
        ]
    )


def _kb_backtest_menu() -> "InlineKeyboardMarkup":
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "▶ Запустить (~5-10 мин)", callback_data="backtest_now"
                )
            ],
            [
                InlineKeyboardButton(
                    "📈 Значимость альфы", callback_data="alpha_report_now"
                )
            ],
            [
                InlineKeyboardButton(
                    "📐 Live vs Backtest", callback_data="live_divergence_now"
                )
            ],
            [
                InlineKeyboardButton(
                    "🤖 AI-изоляция (тратит AI-лимиты)",
                    callback_data="ai_isolation_menu",
                )
            ],
            [InlineKeyboardButton("« Настройки", callback_data="settings")],
        ]
    )


def _kb_ai_isolation_confirm() -> "InlineKeyboardMarkup":
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ Запустить (~150 AI-вызовов)", callback_data="ai_isolation_now"
                ),
                InlineKeyboardButton("⏭ Отмена", callback_data="ai_isolation_skip"),
            ]
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
            [InlineKeyboardButton("🗑 Закрыть все позиции", callback_data="closeall")],
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


def _kb_after_training(worse: bool = False) -> "InlineKeyboardMarkup":
    rows = []
    if worse:
        rows.append(
            [InlineKeyboardButton("🔄 Откатить модель", callback_data="rollback_sac")]
        )
    rows += [
        [InlineKeyboardButton("📊 Запустить бэктест", callback_data="backtest_now")],
        [
            InlineKeyboardButton("📊 Статус", callback_data="status"),
            InlineKeyboardButton("⚙️ Настройки", callback_data="settings"),
        ],
    ]
    return InlineKeyboardMarkup(rows)


def _kb_risk_menu(
    drawdown_on: bool,
    max_pos: int = 3,
    confirm_cycles: int = 3,
    conf_live: float = 0.65,
    conf_paper: float = 0.60,
    guards_off: "bool | None" = None,
) -> "InlineKeyboardMarkup":
    dd_label = (
        "✅ Защита просадки: ВКЛ  →  выкл"
        if drawdown_on
        else "❌ Защита просадки: ВЫКЛ  →  вкл"
    )
    rows = [
        [
            InlineKeyboardButton("🟢 Консерватив.", callback_data="risk:conservative"),
            InlineKeyboardButton("🟡 Умеренный", callback_data="risk:moderate"),
            InlineKeyboardButton("🔴 Агрессивный", callback_data="risk:aggressive"),
        ],
        [
            InlineKeyboardButton("➖", callback_data="pos_less"),
            InlineKeyboardButton(f"📊 Позиций: {max_pos}", callback_data="pos_noop"),
            InlineKeyboardButton("➕", callback_data="pos_more"),
        ],
        [InlineKeyboardButton(dd_label, callback_data="toggle_drawdown")],
        *(  # только в Paper: guards_off=None означает Live, кнопки нет
            []
            if guards_off is None
            else [
                [
                    InlineKeyboardButton(
                        (
                            "⚠️ Paper без защит портфеля: ВКЛ  →  выкл"
                            if guards_off
                            else "🛡 Paper без защит портфеля: ВЫКЛ  →  вкл"
                        ),
                        callback_data="toggle_paper_guards",
                    )
                ]
            ]
        ),
        [
            InlineKeyboardButton("➖", callback_data="dd_confirm_less"),
            InlineKeyboardButton(
                f"🛡 Flash-защита: {confirm_cycles} цикл.",
                callback_data="dd_confirm_noop",
            ),
            InlineKeyboardButton("➕", callback_data="dd_confirm_more"),
        ],
        [
            InlineKeyboardButton(
                f"🎯 Live ▼ {conf_live:.2f}", callback_data="conf_live_down"
            ),
            InlineKeyboardButton(
                f"🎯 Live ▲ {conf_live:.2f}", callback_data="conf_live_up"
            ),
        ],
        [
            InlineKeyboardButton(
                f"📄 Paper ▼ {conf_paper:.2f}", callback_data="conf_paper_down"
            ),
            InlineKeyboardButton(
                f"📄 Paper ▲ {conf_paper:.2f}", callback_data="conf_paper_up"
            ),
        ],
        [InlineKeyboardButton("📊 Управление плечом →", callback_data="lev_menu")],
        [InlineKeyboardButton("« Настройки", callback_data="settings")],
    ]
    if Config.PAPER_TRADING:
        rows.insert(
            2,
            [InlineKeyboardButton("🚀 Paper макс (15)", callback_data="pos_paper_max")],
        )
    return InlineKeyboardMarkup(rows)


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


def _kb_market_type_menu(current: str) -> "InlineKeyboardMarkup":
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    ("✅ " if current == "spot" else "") + "💵 Spot",
                    callback_data="market_type:spot",
                ),
                InlineKeyboardButton(
                    ("✅ " if current == "linear" else "") + "📐 Futures",
                    callback_data="market_type:linear",
                ),
            ],
            [InlineKeyboardButton("« Настройки", callback_data="settings")],
        ]
    )


def _kb_provider_menu(current: str, single: bool = False) -> "InlineKeyboardMarkup":
    """Провайдеры с ключом (и выбранный сейчас) по три в ряд; 🆓 — бесплатные.

    Последняя строка — как работать: цепочкой или одним провайдером.
    """
    shown = [
        p
        for p in ai_providers.PROVIDERS
        if (p.key_env and os.getenv(p.key_env)) or p.id == current
    ]

    def _p(p: "ai_providers.Provider") -> "InlineKeyboardButton":
        tick = "✅ " if p.id == current else ""
        free = "🆓 " if p.free_tier else ""
        return InlineKeyboardButton(
            f"{tick}{free}{p.name}", callback_data=f"provider:{p.id}"
        )

    auto_tick = "✅ " if current == "auto" else ""
    rows = [[_p(p) for p in shown[i : i + 3]] for i in range(0, len(shown), 3)]
    return InlineKeyboardMarkup(
        rows
        + [
            [
                InlineKeyboardButton(
                    f"{auto_tick}🔄 auto (цепочка)", callback_data="provider:auto"
                )
            ],
            [
                InlineKeyboardButton(
                    "🎯 Один провайдер ✅" if single else "🎯 Один провайдер",
                    callback_data="ai_single:1",
                ),
                InlineKeyboardButton(
                    "🔗 Цепочка" if single else "🔗 Цепочка ✅",
                    callback_data="ai_single:0",
                ),
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
    "volume_spike": "VolSpk🔴",
    "momentum_burst": "MomB🔴",
    "pump_exit": "PumpX🔴",
    "sentiment": "Sent🔴",
    # ICT tier-1
    "support_resistance": "S/R",
    "fibonacci_retracement": "Fib",
    "cvd_divergence": "CVD🔴",
    "fair_value_gap": "FVG",
    "inverse_fair_value_gap": "IFVG🔴",
    "liquidity_sweep": "Sweep🔴",
    "amd": "AMD🔴",
    # из Argent-Systems (экспериментальные)
    "supertrend": "SupTr",
    "dual_thrust": "DualT",
    "ttm_squeeze": "Squeeze",
    "adx_trend_birth": "ADXb",
    "vwap_reversion": "VWAP↩",
    "crash_bounce": "Bounce🔴",
    "bear_hedge": "Bear🔴",
    "ichimoku": "Ichi",
    "parabolic_sar": "SAR",
    "vortex": "Vortex",
    "stoch_rsi": "StRSI",
    "williams_r": "W%R",
    "keltner_fade": "KeltF",
    "obv_trend": "OBV",
    "mfi_trend": "MFI",
    "chaikin_osc": "ChOsc",
    "coppock": "Coppock",
    "coral_trend": "Coral",
    "kst": "KST",
    "rvi": "RVI",
    "waddah_attar": "WAE",
    "zlsma": "ZLSMA",
    "cci_aroon": "CCIAr",
    "engulfing_trend": "Engulf",
    "chaikin_money_flow": "CMF",
    "fisher_transform": "Fisher",
    "kdj": "KDJ",
    "klinger": "Kling",
    "awesome_saucer": "AO",
    "heikin_ashi": "HA",
    "rsi_divergence": "RSIdv",
    "shooting_star": "Star🔴",
    "cci_reversion": "CCI",
    "ifr2": "IFR2",
    "stochastic": "Stoch",
    "adx_dmi": "DMI",
    "chandelier": "Chand",
    "halftrend": "HalfT",
    "ssl_channel": "SSL",
    "ad_line_trend": "A/D",
    "ease_of_movement": "EoM",
    "elder_force": "Elder",
    "vwap_trend": "VWAPt",
    "vw_macd": "VWMACD",
    "poc_reversion": "POC",
    "bollinger_reentry": "BBre",
    "dual_momentum": "DualM",
    "gtaa_ma": "GTAA",
    "spike_reversal": "VSrev",
}


_ALERTS_MENU_TEXT = (
    "🔔 *Управление уведомлениями*\n\n"
    "⚡ ATR Spike и 🔇 BotStalled работают в *обоих* режимах (paper и live).\n\n"
    "📉 Drawdown и 🔒 Daily limit — только в *paper*: "
    "в live-режиме бот сам закрывает позиции или блокирует входы "
    "и отправляет одно сообщение о действии — повторных алертов нет."
)


def _kb_alerts(rc: "Any") -> "InlineKeyboardMarkup":
    """Клавиатура управления уведомлениями: вкл/выкл + настройка cooldown."""
    defaults = rc._ALERT_DEFAULTS
    rows = []
    for name, meta in defaults.items():
        enabled = rc.get_alert_enabled(name)
        cooldown = rc.get_alert_cooldown_mins(name)
        toggle_icon = "✅" if enabled else "❌"
        toggle_btn = InlineKeyboardButton(
            f"{toggle_icon} {meta['label']}",
            callback_data=f"alert_toggle:{name}",
        )
        if cooldown > 0:
            cooldown_btn = InlineKeyboardButton(
                f"⏱ {cooldown} мин",
                callback_data=f"alert_cooldown:{name}",
            )
            rows.append([toggle_btn, cooldown_btn])
        else:
            rows.append([toggle_btn])
    rows.append([InlineKeyboardButton("« Настройки", callback_data="settings")])
    return InlineKeyboardMarkup(rows)


STRAT_PAGE = 12  # экспериментальных на одной странице меню


def _strat_button(s: dict, page: "int | None" = None) -> "InlineKeyboardButton":
    icon = "✅" if s["enabled"] else "❌"
    label = f"{icon} {_STRAT_SHORT.get(s['name'], s['name'])}"
    tail = "" if page is None else f":{page}"
    return InlineKeyboardButton(label, callback_data=f"strat:{s['name']}{tail}")


def _rows_of_three(buttons: list) -> list:
    return [buttons[i : i + 3] for i in range(0, len(buttons), 3)]


def _kb_strategies(
    core: list, exp_total: int = 0, exp_on: int = 0
) -> "InlineKeyboardMarkup":
    """Основные стратегии кнопками; экспериментальные — одним переходом ниже."""
    rows = _rows_of_three([_strat_button(s) for s in core])
    if exp_total:
        rows.append(
            [
                InlineKeyboardButton(
                    f"🧪 Экспериментальные ({exp_on}/{exp_total}) →",
                    callback_data="strat_exp:0",
                )
            ]
        )
    rows.append(
        [
            InlineKeyboardButton("🔁 Основные ВКЛ", callback_data="strat_reset"),
            InlineKeyboardButton("« Настройки", callback_data="settings"),
        ]
    )
    return InlineKeyboardMarkup(rows)


def _kb_strategies_exp(items: list, page: int, pages: int) -> "InlineKeyboardMarkup":
    """Страница экспериментальных стратегий с листанием."""
    rows = _rows_of_three([_strat_button(s, page) for s in items])
    nav = []
    if page > 0:
        nav.append(
            InlineKeyboardButton("‹ Назад", callback_data=f"strat_exp:{page - 1}")
        )
    nav.append(InlineKeyboardButton(f"{page + 1}/{pages}", callback_data="noop"))
    if page < pages - 1:
        nav.append(
            InlineKeyboardButton("Дальше ›", callback_data=f"strat_exp:{page + 1}")
        )
    rows.append(nav)
    rows.append([InlineKeyboardButton("« Основные", callback_data="strategies")])
    return InlineKeyboardMarkup(rows)


# ── Тексты справки ────────────────────────────────────────────────────────────

_HELP_ABOUT = (
    "📖 *О боте BitbotBY*\n\n"
    "Гибридный крипто-трейдинг бот для Bybit Futures.\n\n"
    "*Как работает:*\n"
    "1️⃣ Сканирует топ-N монет по объёму каждые 30 сек\n"
    "2️⃣ Загружает OHLCV + технические индикаторы\n"
    "3️⃣ Анализирует через AI (Claude / DeepSeek / Groq / Gemini)\n"
    "4️⃣ Комбинирует AI-сигнал с локальными стратегиями\n"
    "5️⃣ Без авто-сделок — спрашивает подтверждение в Telegram (10–120 с)\n"
    "6️⃣ Управляет позицией: SL/TP, трейлинг-стоп, circuit breaker\n\n"
    "*Paper trading:* виртуальные сделки, реальные данные биржи\n"
    "*Live trading:* реальные ордера через Bybit API"
)

_HELP_MODES = (
    "💡 *Режимы торговли*\n\n"
    "🤖 *AI* — всё решает Claude / DeepSeek / Groq / Gemini\n"
    "   Лучший режим при наличии API ключа\n\n"
    "📐 *Local* — локальные стратегии без AI\n"
    "   RSI, EMA, Bollinger, Breakout и др.\n"
    "   Работает без интернет-зависимостей\n\n"
    "🔀 *Hybrid* — AI + SAC + локальные стратегии совместно\n"
    "   Сигналы комбинируются и взвешиваются\n\n"
    "🧠 *SAC* — нейросеть (Soft Actor-Critic), обученная на истории\n"
    "   Требует предварительного обучения тренером\n\n"
    "🎯 *Профили рынка* (Настройки → Профиль рынка):\n"
    "   🔵 Блючипы — топ-20, 15m, риск 2%\n"
    "   🟡 Альткоины — топ-100, 5m, риск 1%\n"
    "   🚀 Мемкоины — топ-200, 3m, риск 0.5%\n"
    "   + памп-стратегии + Twitter/Telegram сентимент\n\n"
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
    "`/mode ai`     — сменить режим (ai/local/hybrid/sac)\n"
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
    "10 базовых стратегий — каждую можно вкл/выкл:\n"
    "EMA↕ · RSI · MACD↕ · BB · Scalp🔴 · Swing · Break🔴 · Mean↩ · Trend🟢 · VolumeSpike\n"
    "🟢 низкий риск · 🟡 средний · 🔴 высокий\n"
    "Профильные (авто при смене профиля):\n"
    "🔵 volume\\_spike\\_bluechip · 🟡 volume\\_spike\\_altcoin\n"
    "🚀 volume\\_spike\\_meme · momentum\\_burst · pump\\_exit · sentiment\n"
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
    "🛡 Только ты (твой `chat_id`) можешь управлять ботом"
)


def _parse_train_result(stdout: str) -> "dict | None":
    import json as _json

    for line in stdout.splitlines():
        if line.startswith("TRAIN_RESULT:"):
            try:
                return _json.loads(line[len("TRAIN_RESULT:") :])
            except Exception as exc:
                logger.warning("Не удалось разобрать TRAIN_RESULT: %s", exc)
                return None
    return None


def _is_worse(result: "dict | None") -> bool:
    if not result:
        return False
    return result.get("sac_pct", 0) < result.get("bh_pct", 0) - 10


def _format_train_result(result: "dict | None") -> str:
    if not result:
        return ""
    sac = result.get("sac_pct", 0)
    bh = result.get("bh_pct", 0)
    icon = "✅" if sac >= bh else "⚠️"
    return (
        f"\n📊 *Тестовая выборка (holdout 20%):*\n"
        f"  SAC: `{sac:+.1f}%`  |  Buy\\&Hold: `{bh:+.1f}%`  {icon}"
    )


# ── Управление обычными словами ───────────────────────────────────────────────
# Только то же, что уже умеют команды: фраза → команда с аргументом. Переключение
# Paper/Live словами нельзя — для него есть кодовое слово.

_MODE_WORDS = {
    "ai": "ai",
    "аи": "ai",
    "ии": "ai",
    "local": "local",
    "локальный": "local",
    "локально": "local",
    "hybrid": "hybrid",
    "гибрид": "hybrid",
    "гибридный": "hybrid",
    "sac": "sac",
}
_PHRASES: "list[tuple[str, re.Pattern[str]]]" = [
    ("help", re.compile(r"(помощь|помоги|что умеешь|что ты умеешь|команды|\?)")),
    (
        "pause",
        re.compile(
            r"(пауза|на паузу|поставь на паузу|приостанови\w*|останови\w*( торговлю)?)"
        ),
    ),
    (
        "resume",
        re.compile(r"(продолжи\w*|возобнови\w*|запусти\w*|сними с паузы)( торговлю)?"),
    ),
    (
        "status",
        re.compile(r"(статус|как дела|что делаешь|что ты делаешь|что происходит)"),
    ),
    ("pnl", re.compile(r"(pnl|прибыль|результат|результаты|сколько заработал\w*)")),
    ("pos", re.compile(r"(позиции|открытые позиции|сделки|что открыто)")),
    ("closeall", re.compile(r"(закрой|закрыть) (все|всё)( позиции| сделки)?")),
    ("mode", re.compile(r"(режим|mode) (?P<arg>\w+)")),
    (
        "risk",
        re.compile(
            r"((поставь|установи|сделай) )?риск( на сделку)? "
            r"(?P<arg>\d+(?:[.,]\d+)?) ?%?"
        ),
    ),
    (
        "slots",
        re.compile(
            r"((поставь|установи|сделай) )?(сделок|позиций)"
            r"( сразу| одновременно)? (?P<arg>\d+)"
        ),
    ),
    ("add", re.compile(r"(добавь|следи за) (?P<arg>[a-z0-9]{2,10})(/usdt)?")),
    ("remove", re.compile(r"(убери|удали|исключи) (?P<arg>[a-z0-9]{2,10})(/usdt)?")),
    ("strat_on", re.compile(r"включи стратеги\w* (?P<arg>.+)")),
    ("strat_off", re.compile(r"(выключи|отключи) стратеги\w* (?P<arg>.+)")),
]
_PHRASE_HELP = (
    "Можно писать словами:\n"
    "• статус · прибыль · позиции\n"
    "• пауза · продолжи · закрой все\n"
    "• режим hybrid · риск 1.5% · сделок 3\n"
    "• добавь PEPE · убери PEPE\n"
    "• включи стратегию rsi · выключи стратегию bb\n"
    "Или команды: /help"
)


def parse_phrase(text: str) -> "Optional[tuple[str, str]]":
    """Фраза владельца → (действие, аргумент); None — не поняли."""
    t = text.lower().replace("ё", "е").strip()
    t = re.sub(r"[.,!]+$", "", t)
    t = re.sub(r"^(бот|пожалуйста)[ ,]+|[ ,]+пожалуйста$", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    for intent, rx in _PHRASES:
        m = rx.fullmatch(t)
        if m:
            return intent, (m.groupdict().get("arg") or "").strip()
    return None


def find_strategy(word: str, names: "list[str]") -> "list[str]":
    """Стратегия по имени, началу имени или короткой метке из меню (rsi, bb, fvg)."""
    w = word.strip().lower().replace(" ", "_")
    exact = [
        n
        for n in names
        if n == w or _STRAT_SHORT.get(n, "").lower().rstrip("🔴🟢↕↩") == w
    ]
    return exact or [n for n in names if n.startswith(w)]


_PRESET_LABEL = {
    "conservative": "🟢 Конс.",
    "moderate": "🟡 Умер.",
    "aggressive": "🔴 Агр.",
}
_LEV_LABEL = {
    "fixed": "плечо фикс",
    "volatility": "плечо по волатильности",
    "full": "плечо макс",
}


def _presets_text() -> str:
    """Строки пресетов из RISK_PRESETS — те же значения, что в приложении."""
    rows = []
    for name, p in RISK_PRESETS.items():
        dd = "ВКЛ" if p["DRAWDOWN_SCALE_ENABLED"] else "ВЫКЛ"
        rows.append(
            f"{_PRESET_LABEL.get(name, name)} — {p['MAX_POSITIONS']} поз, "
            f"{p['RISK_PER_TRADE'] * 100:g}%, защита {dd}, "
            f"{_LEV_LABEL.get(p['LEVERAGE_MODE'], p['LEVERAGE_MODE'])}\n"
        )
    return "".join(rows)


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
        on_mode_switch: "Optional[Callable[[], Any]]" = None,
        on_restart: "Optional[Callable[[], Any]]" = None,
        on_close_all: "Optional[Callable[[], Any]]" = None,
    ) -> None:
        self._notifier = notifier
        self._rc = runtime_config
        self._get_state = get_state
        self._on_mode_switch = on_mode_switch
        self._on_restart = on_restart
        self._on_close_all = on_close_all
        self._sac_training = False
        self._backtesting = False
        self._alpha_report_running = False
        self._live_divergence_running = False
        self._ai_isolation_running = False
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
            ("symbols", self._cmd_symbols),
            ("pnl", self._cmd_pnl),
            ("pos", self._cmd_pos),
            ("closeall", self._cmd_closeall),
            ("add", self._cmd_add),
            ("remove", self._cmd_remove),
            ("strategies", self._cmd_strategies),
            ("hours", self._cmd_hours),
            ("risk", self._cmd_risk),
            ("trainn", self._cmd_trainn),
            ("lev", self._cmd_lev),
            ("provider", self._cmd_provider),
            ("alerts", self._cmd_alerts),
            ("app", self._cmd_app),
        ]
        for name, handler in commands:
            app.add_handler(CommandHandler(name, handler))
        app.add_handler(CallbackQueryHandler(self._handle_callback))
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text)
        )

        bot_commands = [
            ("start", "Главная панель"),
            ("status", "Текущий статус бота"),
            ("settings", "Все настройки"),
            ("pnl", "Прибыль и убытки"),
            ("pos", "Открытые позиции"),
            ("closeall", "Закрыть все позиции"),
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
            ("alerts", "Управление уведомлениями"),
            ("app", "Ссылка входа в приложение"),
            ("trainn", "Переобучить SAC"),
            ("help", "Справка"),
        ]
        setup = [app.bot.set_my_commands(bot_commands)]
        url = _webapp_url()
        chat = str(self._notifier._chat_id or "")
        if url and chat.lstrip("-").isdigit():
            # Кнопка у поля ввода — открывает MiniApp с initData владельца.
            setup.append(
                app.bot.set_chat_menu_button(
                    chat_id=int(chat),
                    menu_button=MenuButtonWebApp("Терминал", WebAppInfo(url=url)),
                )
            )
        try:
            import asyncio as _asyncio

            loop = _asyncio.get_event_loop()
            for coro in setup:
                if loop.is_running():
                    _asyncio.ensure_future(coro)
                else:
                    loop.run_until_complete(coro)
        except Exception as e:
            logger.warning("Telegram commands/menu setup failed: %s", e)
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
            err = str(_e).lower()
            if "message is not modified" in err:
                return
            # Telegram rejected the edit (parse error, too old, etc.) — send new message
            logger.warning("edit_message_text failed (%s), sending new message", _e)
            try:
                await query.message.reply_text(
                    text, parse_mode="Markdown", reply_markup=kb
                )
            except Exception as _e2:
                logger.error("fallback reply_text also failed: %s", _e2)

    # ── Builders ──────────────────────────────────────────────────────────────

    _MODE_DISPLAY = {"dqn": "SAC", "ai": "AI", "hybrid": "Hybrid", "local": "Локальные"}

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
        configured_provider = self._rc.get_ai_provider()
        last_provider = self._rc.get_last_ai_provider()
        provider = last_provider if last_provider else configured_provider
        if provider == "local":
            provider = "🔴 local"
        hours = self._rc.get_trading_hours()

        state_icon = "⏸" if paused else "🟢"
        exec_icon = "✅" if auto_exec else "❌"
        hours_str = f"`{hours}`" if hours else "`24/7`"

        profile_labels = {
            "bluechip": "🔵 Блючипы",
            "altcoin": "🟡 Альткоины",
            "meme": "🚀 Мемкоины",
        }
        market_profile = self._rc.get_market_profile()
        profile_str = profile_labels.get(market_profile, "⚪ не задан")
        season_data = self._rc.get_season_index()
        if season_data:
            _sig = season_data.get("signal")
            _ai = season_data.get("altcoin_index") or 0.0
            _sig_label = (
                "🟡 Альтсезон"
                if _sig == "altcoin"
                else "🔵 BTC Season" if _sig == "bluechip" else "⚪ Нейтрально"
            )
            season_line = f"🌡 Сезон: {_sig_label} (индекс {_ai:.0f}/100)\n"
        else:
            season_line = "🌡 Сезон: ⏳ загружается...\n"

        fng_data = self._rc.get_fear_greed()
        if fng_data:
            _fv = fng_data.get("value", 0)
            _fl = fng_data.get("label", "")
            _fng_icon = (
                "😱"
                if _fv < 25
                else (
                    "😨"
                    if _fv < 45
                    else "😐" if _fv < 55 else "😊" if _fv < 75 else "🤑"
                )
            )
            fng_line = f"{_fng_icon} Страх/Жадность: `{_fv}` ({_fl})\n"
        else:
            fng_line = "😐 Страх/Жадность: ⏳ загружается...\n"

        chronos_enabled = mode == "hybrid" and self._rc.get_chronos_enabled()
        mode_label = self._MODE_DISPLAY.get(mode, mode.upper())
        if chronos_enabled:
            mode_label += " +Chronos"

        _ai_extra = ""
        if provider == "deepseek":
            _bal = self._rc.get_deepseek_balance()
            if _bal:
                _ai_extra = f" — баланс ${_bal['balance']:.2f}"
        elif provider in ai_providers.BY_ID:
            _used = self._rc.get_token_usage_today(provider)
            if _used:
                _ai_extra = f" — сегодня {_used:,} токенов".replace(",", " ")

        text = (
            f"{state_icon} *BitbotBY [{paper}]*\n\n"
            f"💰 Баланс: `${balance:,.2f}` ({pnl_pct:+.2f}%)\n"
            f"📊 Профиль: {profile_str}\n"
            f"{season_line}"
            f"{fng_line}"
            f"🤖 Режим: `{mode_label}`\n"
            f"🧠 AI-провайдер: `{provider}`{_ai_extra}\n"
            f"🔢 Символов: `{n}` | Позиций: `{len(positions)}`\n"
            f"⏱ Цикл: `{self._rc.get_trading_interval()}с`"
            f" | Плечо: `{lev_mode}` ({lev_target*100:.1f}% ATR)\n"
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
        lev_dd = self._rc.get_leverage_dd_threshold()
        provider = self._rc.get_ai_provider()
        hours = self._rc.get_trading_hours()
        max_pos = self._rc.get_max_positions()
        risk_pt = self._rc.get_risk_per_trade()
        dd_scale = self._rc.get_drawdown_scale_enabled()
        train_n = self._rc.get_train_top_n()
        disabled_strats = self._rc.get_disabled_strategies()
        market_profile = self._rc.get_market_profile()
        timeframe = self._rc.get_timeframe()

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

        chronos = self._rc.get_chronos_enabled() if mode == "hybrid" else False
        chronos_str = ""
        if mode == "hybrid":
            chronos_str = (
                f"  Chronos (усил. контроль): {'✅ ВКЛ' if chronos else '❌ ВЫКЛ'}\n"
            )

        profile_labels = {
            "bluechip": "🔵 Блючипы",
            "altcoin": "🟡 Альткоины",
            "meme": "🚀 Мемкоины",
        }
        profile_str = profile_labels.get(market_profile, "⚪ не задан")

        season_data = self._rc.get_season_index()
        if season_data:
            _sig = season_data.get("signal")
            _ai = season_data.get("altcoin_index")
            _dom = season_data.get("btc_dominance")
            _sig_label = (
                "🟡 Альтсезон"
                if _sig == "altcoin"
                else "🔵 BTC Season" if _sig == "bluechip" else "⚪ Нейтрально"
            )
            _bar = "█" * int((_ai or 0) / 10) + "░" * (10 - int((_ai or 0) / 10))
            season_str = (
                f"{_sig_label}  `[{_bar}]` *{_ai:.0f}/100*" f"  BTC dom: *{_dom:.1f}%*"
            )
        else:
            season_str = "⏳ ожидание данных CoinGecko"

        text = (
            f"⚙️ *Настройки бота*\n\n"
            f"*Статус:* {state_str}\n"
            f"*Профиль рынка:* {profile_str}\n"
            f"*Сезон (CoinGecko):* {season_str}\n"
            f"*Режим торговли:* `{self._MODE_DISPLAY.get(mode, mode.upper())}`\n"
            f"{chronos_str}"
            f"*Таймфрейм:* `{timeframe}`\n"
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
        season_mode = self._rc.get_season_switch_mode()
        is_tuning = self._sac_training and bool(self._rc.get_tune_progress())
        return text, _kb_settings(
            paused,
            auto_exec,
            mode=mode,
            chronos=chronos,
            paper=Config.PAPER_TRADING,
            season_mode=season_mode,
            sac_training=self._sac_training,
            is_tuning=is_tuning,
        )

    def _build_risk(self) -> tuple[str, Any]:
        r = self._rc.get_risk_summary()
        max_pos = r["max_positions"]
        rpt = r["risk_per_trade"]
        dd = r["drawdown_scale_enabled"]
        confirm = r["drawdown_confirm_cycles"]
        dd_icon = "✅" if dd else "❌"
        cl = self._rc.get_signal_confidence(paper=False)
        cp = self._rc.get_signal_confidence(paper=True)
        text = (
            f"⚖️ *Риск-профиль*\n\n"
            f"Макс. позиций: `{max_pos}`\n"
            f"Риск на сделку: `{rpt * 100:.1f}%`\n"
            f"Защита просадки: {dd_icon}\n"
            f"Flash-защита (циклов): `{confirm}` × 30с = `{confirm * 30}с`\n"
            f"Порог сигнала — Live: `{cl:.2f}` / Paper: `{cp:.2f}`\n\n"
            f"_Пресеты:_\n" + _presets_text()
        )
        paper_off = self._rc.get_paper_guards_off() if Config.PAPER_TRADING else None
        if paper_off:
            text += "\n⚠️ *Paper без защит портфеля* — паузы и лимиты не действуют.\n"
        return text, _kb_risk_menu(dd, max_pos, confirm, cl, cp, paper_off)

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

    def _strategy_rows(self) -> "tuple[list, list, int]":
        """(основные, экспериментальные, скрыто) с учётом рынка: на споте нет шортов."""
        strats = self._get_state().get("strategies", [])
        spot = self._rc.get_market_type() == "spot"
        want = "spot" if spot else "futures"
        fits = [s for s in strats if s.get("markets", "both") in ("both", want)]
        core = [s for s in fits if not s.get("experimental")]
        exp = [s for s in fits if s.get("experimental")]
        return core, exp, len(strats) - len(fits)

    def _build_strategies(self, exp_page: "int | None" = None) -> tuple[str, Any]:
        core, exp, hidden = self._strategy_rows()
        exp_on = sum(1 for s in exp if s["enabled"])
        if exp_page is not None:
            return self._build_strategies_exp(exp, exp_page)
        total = len(core)
        enabled_count = sum(1 for s in core if s["enabled"])
        mode = self._rc.get_mode()

        if mode == "local":
            header = (
                "📐 *Стратегии* — включено {}/{}\n\n"
                "⚠️ Режим *Local* — только математика, без AI.\n"
                "_Используй на свой страх и риск в реальной торговле._\n"
            ).format(enabled_count, total)
            note = (
                "\n\n🟢 низкий риск · 🟡 средний · 🔴 высокий\n"
                "_Стратегия выбирается автоматически по индикаторам рынка._"
            )
        elif mode == "hybrid":
            header = f"📐 *Стратегии* — включено {enabled_count}/{total}\n"
            note = "\n_Hybrid: стратегии комбинируются с AI-сигналом._"
        else:
            header = f"📐 *Стратегии* — включено {enabled_count}/{total}\n"
            note = "\n_В режиме AI стратегии передаются как контекст в промпт._"

        lines = [header]
        for s in core:
            icon = "✅" if s["enabled"] else "❌"
            risk = _RISK_ICON.get(s.get("risk_level", "medium"), "🟡")
            mtype = s.get("market_type", "any")
            lines.append(f"{icon} `{s['name']}` {risk} _{mtype}_")
        if exp:
            lines.append(
                f"\n🧪 Экспериментальных: {len(exp)}, включено {exp_on} "
                "(без доказанного преимущества, выключены по умолчанию)."
            )
        if hidden:
            lines.append(f"_Скрыто {hidden}: не подходят рынку «Спот» (нужны шорты)._")
        return "\n".join(lines) + note, _kb_strategies(core, len(exp), exp_on)

    def _build_strategies_exp(self, exp: list, page: int) -> tuple[str, Any]:
        pages = max(1, -(-len(exp) // STRAT_PAGE))
        page = max(0, min(page, pages - 1))
        items = exp[page * STRAT_PAGE : (page + 1) * STRAT_PAGE]
        on = sum(1 for s in exp if s["enabled"])
        lines = [
            f"🧪 *Экспериментальные* — включено {on}/{len(exp)} "
            f"(стр. {page + 1}/{pages})\n",
            "_Перенесены из Argent-Systems. Бэктест не нашёл у них преимущества: "
            "включайте осознанно, лучше по одной и в Paper._\n",
        ]
        for s in items:
            icon = "✅" if s["enabled"] else "❌"
            lines.append(f"{icon} `{s['name']}` _{s.get('market_type', 'any')}_")
        return "\n".join(lines), _kb_strategies_exp(items, page, pages)

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
            side = p.get("side", "buy").upper()
            entry = p.get("entry_price", 0.0)
            qty = p.get("qty", 0.0)
            pnl_pct = p.get("pnl_pct", 0.0)
            sl = p.get("stop_loss", 0.0)
            tp = p.get("take_profit", 0.0)
            icon = "🟢" if pnl_pct >= 0 else "🔴"
            sl_str = f"  SL `{sl:.4f}`" if sl else ""
            tp_str = f"  TP `{tp:.4f}`" if tp else ""
            lines.append(
                f"{icon} *{sym}* {side}\n"
                f"   qty `{qty:.6f}`  entry `${entry:.4f}`\n"
                f"   PnL `{pnl_pct*100:+.2f}%`{sl_str}{tp_str}"
            )
        return "\n".join(lines)

    # ── Команды ───────────────────────────────────────────────────────────────

    async def _cmd_help(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._authorized(update):
            return
        text, kb = self._build_status()
        intro = (
            "👋 *Добро пожаловать в BitbotBY!*\n"
            "Можно писать словами: «статус», «пауза», «риск 1%» — список по «?».\n\n"
            + text
        )
        await self._reply(update, intro, _kb_main())

    async def _cmd_app(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Одноразовая ссылка входа в приложение — только на компьютере с ботом."""
        if not self._authorized(update):
            return
        port = os.getenv("APP_PORT", "")
        if not port.isdigit():
            await self._reply(
                update,
                "Ссылка входа работает только при запуске бота на своём компьютере "
                "(ярлык BitbotBY или launcher/launch.py).",
            )
            return
        token = login_link.issue(self._rc._r.redis_client)
        await self._reply(
            update,
            "🔐 Ссылка входа в приложение — откройте её на компьютере, где запущен "
            "бот (нажмите, чтобы скопировать):\n\n"
            f"`http://localhost:{port}/app/#local={token}`\n\n"
            f"Работает {login_link.TTL // 60} минут и только один раз. "
            "Если вы её не запрашивали — просто не открывайте.",
        )

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
        # В боте режим SAC называется dqn; «sac» пишут люди и подсказка ниже.
        mode = {"sac": "dqn"}.get(args[0].lower(), args[0].lower())
        if self._rc.set_mode(mode):
            await self._reply(
                update,
                f"✅ Режим → `{mode}`\nПрименится на следующем цикле.",
                _kb_after_action(),
            )
        else:
            await self._reply(
                update,
                "❌ Неверный режим. Доступны: `ai`, `local`, `hybrid`, `sac`",
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

    async def _cmd_symbols(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._authorized(update):
            return
        data = self._rc.get_current_symbols()
        if not data or not data.get("symbols"):
            await self._reply(
                update,
                "Список ещё не сформирован — дождись первого торгового цикла.",
            )
            return
        symbols = data["symbols"]
        profile = data.get("profile") or "-"
        updated_at = data.get("updated_at", "")
        lines = "\n".join(f"  {s}" for s in symbols)
        await self._reply(
            update,
            f"📋 Профиль *{profile}* — `{len(symbols)}` монет "
            f"(обновлено {updated_at}):\n```\n{lines}\n```",
        )

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
            dd = self._rc.get_leverage_dd_threshold()
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
                    f"{self._rc.get_leverage_dd_threshold():.0%} "
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
            if self._rc.set_leverage_dd_threshold(pct):
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
            "auto": "Авто — цепочка (порядок и модели — в приложении, раздел AI)",
            **{p.id: f"{p.name} — {p.note}" for p in ai_providers.PROVIDERS},
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

    async def _cmd_closeall(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._authorized(update):
            return
        positions = self._get_state().get("positions", [])
        if not positions:
            await self._reply(update, "📭 Нет открытых позиций.", _kb_pos())
            return
        count = len(positions)
        paper = self._get_state().get("paper_trading", True)
        mode_note = " *(paper режим)*" if paper else " ⚠️ *РЕАЛЬНЫЕ ДЕНЬГИ*"
        await self._reply(
            update,
            f"🗑 *Закрыть все позиции?*{mode_note}\n\n"
            f"Будет закрыто: *{count}* позиций\n\n"
            f"_Это действие нельзя отменить._",
            InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "✅ Закрыть все", callback_data="closeall_confirm"
                        ),
                        InlineKeyboardButton("❌ Отмена", callback_data="pos"),
                    ]
                ]
            ),
        )

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

    async def _cmd_alerts(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._authorized(update):
            return
        await self._reply(update, _ALERTS_MENU_TEXT, _kb_alerts(self._rc))

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

    async def _run_backtest(self) -> None:
        """Запускает backtest.py как subprocess и присылает итоговый отчёт."""
        import asyncio as _asyncio
        import json
        import os

        try:
            env = {**os.environ, "PYTHONPATH": "/app"}
            proc = await _asyncio.create_subprocess_exec(
                "python",
                "backtest.py",
                env=env,
                stdout=_asyncio.subprocess.PIPE,
                stderr=_asyncio.subprocess.PIPE,
            )
            self._rc.clear_backtest_progress()

            stderr_chunks: list[bytes] = []

            async def _read_bt_stdout() -> None:
                assert proc.stdout
                async for raw in proc.stdout:
                    line = raw.decode(errors="replace").strip()
                    if line.startswith("BACKTEST_PROGRESS:"):
                        try:
                            data = json.loads(line[len("BACKTEST_PROGRESS:") :])
                            self._rc.set_backtest_progress(data)
                        except Exception:
                            pass

            async def _read_bt_stderr() -> None:
                assert proc.stderr
                async for raw in proc.stderr:
                    stderr_chunks.append(raw)

            await _asyncio.gather(_read_bt_stdout(), _read_bt_stderr())
            await proc.wait()

            if proc.returncode != 0:
                err = b"".join(stderr_chunks).decode()[-300:]
                await self._notifier.notify(f"❌ *Бэктест упал*\n```{err}```")
                return

            results_path = os.path.join("data", "backtest_results.json")
            try:
                with open(results_path, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                await self._notifier.notify(
                    "✅ Бэктест завершён. Файл результатов недоступен."
                )
                return

            results = data.get("results", [])
            if not results:
                await self._notifier.notify("✅ Бэктест завершён, но результатов нет.")
                return

            best = max(results, key=lambda r: r.get("expected_value", 0))
            mode = self._rc.get_mode() if hasattr(self._rc, "get_mode") else "ai"
            lines = [
                "📊 *Бэктест локальных стратегий завершён*",
                f"`{data.get('symbol', '?')}` | "
                f"{data.get('timeframe', '?')} | "
                f"{data.get('months', '?')} мес.\n",
                f"🏆 Лучшая: `{best['strategy']}`",
                f"  Win rate: *{best['win_rate']:.0%}*"
                f"  |  Sharpe: *{best['sharpe_ratio']:.2f}*",
                f"  Доходность: *{best['total_return_pct']:.1f}%*"
                f"  |  Drawdown: *{best['max_drawdown_pct']:.1f}%*\n",
            ]
            if mode in ("ai", "hybrid"):
                lines.append(
                    f"ℹ️ Ты в режиме `{mode}` — AI генерирует сигналы сам,\n"
                    "локальные стратегии используются как вспомогательный фильтр.\n"
                    "Бэктест показывает их историческую силу как ориентир."
                )
            else:
                if best["win_rate"] >= 0.5 and best["sharpe_ratio"] >= 1.0:
                    lines.append(
                        f"✅ Стратегии работают — можно переходить дальше.\n"
                        f"Рекомендуем: `DEFAULT_STRATEGY={best['strategy']}`"
                    )
                else:
                    lines.append(
                        "⚠️ Win rate или Sharpe ниже нормы — "
                        "не спеши с реальными деньгами"
                    )
            await self._notifier.notify("\n".join(lines))
        except Exception as e:
            await self._notifier.notify(f"❌ Бэктест: непредвиденная ошибка\n`{e}`")
        finally:
            self._backtesting = False
            self._rc.end_job("backtest")
            self._rc.clear_backtest_progress()

    @staticmethod
    def _truncate_report(text: str, limit: int = 3500) -> str:
        """Обрезает начало отчёта, если он не влезает в лимит сообщения
        Telegram (~4096 символов) — итог/сводка обычно внизу отчёта."""
        text = text.strip()
        if len(text) <= limit:
            return text
        return "...\n" + text[-limit:]

    async def _run_report_script(self, script: str) -> "tuple[bool, str]":
        """Запускает read-only отчётный скрипт (без прогресс-стриминга,
        в отличие от backtest.py) и возвращает (успех, вывод/ошибка)."""
        import asyncio as _asyncio
        import os

        env = {**os.environ, "PYTHONPATH": "/app"}
        proc = await _asyncio.create_subprocess_exec(
            "python",
            script,
            env=env,
            stdout=_asyncio.subprocess.PIPE,
            stderr=_asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            return False, stderr.decode(errors="replace")[-300:]
        return True, stdout.decode(errors="replace")

    async def _run_alpha_report(self) -> None:
        """Запускает paper_alpha_report.py — значимость альфы на реальных
        закрытых paper-сделках (bootstrap Sharpe + Wilcoxon + Bonferroni)."""
        try:
            ok, out = await self._run_report_script("paper_alpha_report.py")
            if not ok:
                await self._notifier.notify(
                    f"❌ *Отчёт значимости альфы упал*\n```{out}```"
                )
                return
            await self._notifier.notify(
                "📈 *Значимость альфы (paper-сделки)*\n```"
                f"{self._truncate_report(out)}```"
            )
        except Exception as e:
            await self._notifier.notify(
                f"❌ Отчёт значимости альфы: непредвиденная ошибка\n`{e}`"
            )
        finally:
            self._alpha_report_running = False
            self._rc.end_job("alpha_report")

    async def _run_live_divergence(self) -> None:
        """Запускает backtest_live_divergence.py — расхождение реальной цены
        исполнения с моделью market impact на исторических данных."""
        try:
            ok, out = await self._run_report_script("backtest_live_divergence.py")
            if not ok:
                await self._notifier.notify(
                    f"❌ *Отчёт Live vs Backtest упал*\n```{out}```"
                )
                return
            await self._notifier.notify(
                "📐 *Live vs Backtest (расхождение исполнения)*\n```"
                f"{self._truncate_report(out)}```"
            )
        except Exception as e:
            await self._notifier.notify(
                f"❌ Отчёт Live vs Backtest: непредвиденная ошибка\n`{e}`"
            )
        finally:
            self._live_divergence_running = False
            self._rc.end_job("live_divergence")

    async def _run_ai_isolation_backtest(self) -> None:
        """Запускает ai_signal_isolation_backtest.py — изолирует вклад AI(LLM)
        от голых TA-индикаторов. В отличие от остальных отчётов в этом меню,
        делает реальные (хоть и ограниченные, см. AIB_SAMPLES) вызовы AI-провайдера
        — расходует его лимиты/квоту."""
        try:
            ok, out = await self._run_report_script("ai_signal_isolation_backtest.py")
            if not ok:
                await self._notifier.notify(f"❌ *AI-изоляция упала*\n```{out}```")
                return
            await self._notifier.notify(
                "🤖 *AI-изоляция (вклад AI-слоя отдельно от индикаторов)*\n```"
                f"{self._truncate_report(out)}```"
            )
        except Exception as e:
            await self._notifier.notify(f"❌ AI-изоляция: непредвиденная ошибка\n`{e}`")
        finally:
            self._ai_isolation_running = False
            self._rc.end_job("ai_isolation")

    async def _run_sac_training(self) -> None:
        """Запускает train_sac.py как subprocess, не блокируя event loop."""
        import asyncio as _asyncio
        import os

        model_path = self._rc.get_sac_model_path()
        profile = self._rc.get_market_profile()
        profile_labels = {
            "bluechip": "🔵 Блючипы",
            "altcoin": "🟡 Альткоины",
            "meme": "🚀 Мемкоины",
        }
        profile_note = (
            f" ({profile_labels[profile]})" if profile in profile_labels else ""
        )

        try:
            profile_cfg = self._rc.get_market_profile_config(profile)
            train_top_n = str(profile_cfg.get("train_top_n", 20))
            bt_timeframe = str(profile_cfg.get("timeframe", "15m"))
            env = {
                **os.environ,
                "PYTHONPATH": "/app",
                "SAC_MODEL_PATH": model_path,
                "TRAIN_TOP_N": train_top_n,
                "BT_TIMEFRAME": bt_timeframe,
                "SAC_PROFILE": profile,
                "EXPERIENCES_PATH": (
                    "data/experiences_altcoin.jsonl"
                    if profile == "altcoin"
                    else (
                        "data/experiences_meme.jsonl"
                        if profile == "meme"
                        else "data/experiences.jsonl"
                    )
                ),
            }
            proc = await _asyncio.create_subprocess_exec(
                "python",
                "reinforcement_learning/train_sac.py",
                env=env,
                stdout=_asyncio.subprocess.PIPE,
                stderr=_asyncio.subprocess.PIPE,
            )
            self._rc.clear_train_progress()
            await self._notifier.notify(
                f"🧠 *Обучение SAC запущено{profile_note}*\n\n"
                "Процесс идёт в фоне, бот продолжает торговать.\n"
                f"📁 Сохранение в: `{model_path}`\n"
                "⏱ Ожидаемое время: ~60 мин на CPU.\n"
                "_Пришлю уведомление когда модель будет готова._\n\n"
                "Нажми *Прогресс обучения* в Настройках чтобы проверить статус.",
            )
            stdout_lines: list[str] = []
            stderr_buf: bytes = b""

            async def _read_stdout() -> None:
                assert proc.stdout is not None
                while True:
                    raw = await proc.stdout.readline()
                    if not raw:
                        break
                    line = raw.decode(errors="replace").strip()
                    stdout_lines.append(line)
                    if line.startswith("TRAIN_PROGRESS:"):
                        try:
                            import json as _json

                            data = _json.loads(line[len("TRAIN_PROGRESS:") :])
                            self._rc.set_train_progress(data)
                        except Exception:
                            pass

            async def _read_stderr() -> None:
                nonlocal stderr_buf
                assert proc.stderr is not None
                stderr_buf = await proc.stderr.read()

            await _asyncio.gather(_read_stdout(), _read_stderr(), proc.wait())

            import os as _os

            stdout_bytes = "\n".join(stdout_lines).encode()
            model_exists = _os.path.exists(model_path)
            if proc.returncode == 0 and model_exists:
                train_res = _parse_train_result(stdout_bytes.decode())
                if train_res and train_res.get("backup"):
                    self._rc.set_sac_backup_path(train_res["backup"])
                worse = _is_worse(train_res)
                result_str = _format_train_result(train_res)
                verdict = (
                    "\n\n⚠️ *Модель показала результат хуже рынка.*"
                    " Рекомендуется откат или бэктест для анализа."
                    if worse
                    else "\n\n✅ Модель превзошла Buy\\&Hold на тестовой выборке."
                )
                await self._notifier.notify(
                    f"✅ *SAC модель обучена{profile_note}!*\n\n"
                    f"Модель сохранена в `{model_path}`.\n"
                    f"{result_str}{verdict}\n\n"
                    "Запустить бэктест чтобы проверить результат?",
                    reply_markup=_kb_after_training(worse=worse),
                )
            else:
                err = (
                    stderr_buf.decode(errors="replace")[-300:]
                    if stderr_buf
                    else "нет деталей"
                )
                await self._notifier.notify(
                    f"❌ *Ошибка обучения SAC*\n\n`{err}`\n\nПроверьте: `make logs`",
                    reply_markup=_kb_main(),
                )
        except Exception as e:
            await self._notifier.notify(f"❌ SAC обучение упало: {e}")
        finally:
            self._sac_training = False
            self._rc.end_job("sac")
            self._rc.clear_train_progress()

    async def _run_sac_tune_and_train(self) -> None:
        """Запускает tune_sac.py, затем train_sac.py последовательно."""
        import asyncio as _asyncio
        import json as _json
        import os

        model_path = self._rc.get_sac_model_path()
        env = {**os.environ, "PYTHONPATH": "/app", "SAC_MODEL_PATH": model_path}
        try:
            # ── Этап 1: тюнинг ───────────────────────────────────────────────
            proc = await _asyncio.create_subprocess_exec(
                "python",
                "reinforcement_learning/tune_sac.py",
                env=env,
                stdout=_asyncio.subprocess.PIPE,
                stderr=_asyncio.subprocess.PIPE,
            )

            stderr_lines: list = []
            last_notify_pct = -26

            async def _read_tune_stdout() -> None:
                nonlocal last_notify_pct
                assert proc.stdout
                async for raw in proc.stdout:
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("TUNE_PROGRESS:"):
                        continue
                    try:
                        d = _json.loads(line[len("TUNE_PROGRESS:") :])
                    except Exception:
                        continue
                    pct = d.get("pct", 0)
                    if pct - last_notify_pct < 25 and pct < 100:
                        continue
                    self._rc.set_tune_progress(d)
                    last_notify_pct = pct
                    trial = d.get("trial", 0)
                    total = d.get("total", 0)
                    best_pnl = d.get("best_pnl")
                    eta = d.get("eta_min")
                    pnl_str = f"${best_pnl:,.0f}" if best_pnl is not None else "—"
                    eta_str = f"{eta} мин" if eta is not None else "?"
                    await self._notifier.notify(
                        f"🔬 *Тюнинг SAC* — {pct:.0f}%\n\n"
                        f"Попытка: {trial}/{total}\n"
                        f"Лучший PnL: {pnl_str}\n"
                        f"Осталось: ~{eta_str}"
                    )

            async def _read_tune_stderr() -> None:
                assert proc.stderr
                async for raw in proc.stderr:
                    stderr_lines.append(raw.decode(errors="replace"))

            await _asyncio.gather(_read_tune_stdout(), _read_tune_stderr())
            await proc.wait()

            if proc.returncode != 0:
                err = "".join(stderr_lines)[-400:]
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
                "python",
                "reinforcement_learning/train_sac.py",
                env=env,
                stdout=_asyncio.subprocess.PIPE,
                stderr=_asyncio.subprocess.PIPE,
            )
            stdout2, stderr2 = await proc2.communicate()
            if proc2.returncode == 0:
                train_res2 = _parse_train_result(stdout2.decode())
                if train_res2 and train_res2.get("backup"):
                    self._rc.set_sac_backup_path(train_res2["backup"])
                worse2 = _is_worse(train_res2)
                result_str2 = _format_train_result(train_res2)
                verdict2 = (
                    "\n\n⚠️ *Модель показала результат хуже рынка.*"
                    " Рекомендуется откат или бэктест для анализа."
                    if worse2
                    else "\n\n✅ Модель превзошла Buy\\&Hold на тестовой выборке."
                )
                await self._notifier.notify(
                    "✅ *Модель обновлена с лучшими параметрами!*\n\n"
                    "SAC обучен на результатах тюнинга.\n"
                    f"{result_str2}{verdict2}\n\n"
                    "Запустить бэктест чтобы проверить результат?",
                    reply_markup=_kb_after_training(worse=worse2),
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
            self._rc.end_job("sac")
            self._rc.clear_tune_progress()

    async def _handle_text(
        self, update: Update, _context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat_id = str(getattr(update.effective_chat, "id", ""))
        if chat_id != str(self._notifier._chat_id):
            return
        if not update.message:
            return
        text = (update.message.text or "").strip()

        alert_name = self._rc.get_awaiting_alert_cooldown()
        if alert_name:
            self._rc.clear_awaiting_alert_cooldown()
            try:
                mins = int(text)
                if mins < 1 or mins > 1440:
                    raise ValueError
            except ValueError:
                await self._notifier.notify("❌ Введите число минут от 1 до 1440.")
                return
            self._rc.set_alert_cooldown_mins(alert_name, mins)
            label = self._rc._ALERT_DEFAULTS.get(alert_name, {}).get(
                "label", alert_name
            )
            await self._notifier.notify(
                f"✅ Cooldown для *{label}* установлен: *{mins} мин*",
                parse_mode="Markdown",
            )
            return

        if not self._rc.is_awaiting_mode_pin():
            await self._handle_phrase(update, text)
            return
        self._rc.clear_awaiting_mode_pin()
        if text != Config.TRADING_MODE_PIN:
            await self._notifier.notify(
                "❌ Неверное кодовое слово. Переключение отменено."
            )
            return
        import os as _os

        new_paper = not Config.PAPER_TRADING
        Config.PAPER_TRADING = new_paper
        _os.environ["PAPER_TRADING"] = "true" if new_paper else "false"
        self._rc.set_paper_trading_override(new_paper)
        mode_name = (
            "📄 PAPER (симуляция)" if new_paper else "💰 LIVE (реальная торговля)"
        )
        detail = (
            "⚠️ Бот теперь торгует на реальные деньги."
            if not new_paper
            else "✅ Бот переведён в режим симуляции. Реальных сделок нет."
        )
        await self._notifier.notify(
            f"🚨 *ВАЖНО! РЕЖИМ ТОРГОВЛИ ИЗМЕНЁН*\n\n"
            f"Текущий режим: *{mode_name}*\n\n"
            f"{detail}\n\n"
            f"Изменение сохранено и восстановится после перезапуска.",
            parse_mode="Markdown",
        )

    async def _handle_phrase(self, update: Update, text: str) -> None:
        """Обычные слова владельца → существующие команды (см. parse_phrase)."""
        parsed = parse_phrase(text)
        if parsed is None:
            await self._reply(update, "Не понял. " + _PHRASE_HELP)
            return
        intent, arg = parsed
        ctx: Any = SimpleNamespace(args=[arg] if arg else [])
        simple = {
            "pause": self._cmd_pause,
            "resume": self._cmd_resume,
            "status": self._cmd_status,
            "pnl": self._cmd_pnl,
            "pos": self._cmd_pos,
            "closeall": self._cmd_closeall,  # спросит подтверждение кнопкой
            "add": self._cmd_add,
            "remove": self._cmd_remove,
        }
        if intent in simple:
            await simple[intent](update, ctx)
        elif intent == "help":
            await self._reply(update, _PHRASE_HELP)
        elif intent == "mode":
            ctx.args = [_MODE_WORDS.get(arg, arg)]
            await self._cmd_mode(update, ctx)
        elif intent == "risk":
            pct = float(arg.replace(",", "."))
            if 0.1 <= pct <= 10 and self._rc.set_risk_per_trade(round(pct / 100, 4)):
                await self._reply(update, f"✅ Риск на сделку: *{pct:g}%*")
            else:
                await self._reply(update, "❌ Риск на сделку — от 0.1% до 10%.")
        elif intent == "slots":
            if self._rc.set_max_positions(int(arg)):
                await self._reply(update, f"✅ Сделок одновременно: *{int(arg)}*")
            else:
                await self._reply(update, "❌ Сделок одновременно — от 1 до 20.")
        else:
            await self._phrase_strategy(update, arg, on=intent == "strat_on")

    async def _phrase_strategy(self, update: Update, word: str, on: bool) -> None:
        from src.strategies import STRATEGY_REGISTRY  # noqa: PLC0415

        found = find_strategy(word, list(STRATEGY_REGISTRY))
        if len(found) != 1:
            hint = ", ".join(f"`{n}`" for n in found[:6]) or "нет такой"
            await self._reply(update, f"Какую стратегию? {hint}. Список: /strategies")
            return
        name = found[0]
        if on:
            self._rc.enable_strategy(name)
        else:
            self._rc.disable_strategy(name)
        state = "включена" if on else "отключена"
        await self._reply(update, f"{'✅' if on else '❌'} Стратегия `{name}` {state}.")

    async def _handle_callback(
        self, update: Update, _context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        try:
            await query.answer()
        except Exception:
            # Кнопка устарела (>10 мин после перезапуска бота) — молча игнорируем
            return

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

        elif data == "closeall":
            positions = self._get_state().get("positions", [])
            if not positions:
                await self._edit(query, "📭 Нет открытых позиций.", _kb_pos())
                return
            count = len(positions)
            paper = self._get_state().get("paper_trading", True)
            mode_note = " *(paper режим)*" if paper else " ⚠️ *РЕАЛЬНЫЕ ДЕНЬГИ*"
            await self._edit(
                query,
                f"🗑 *Закрыть все позиции?*{mode_note}\n\n"
                f"Будет закрыто: *{count}* позиций\n\n"
                f"_Это действие нельзя отменить._",
                InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "✅ Закрыть все", callback_data="closeall_confirm"
                            ),
                            InlineKeyboardButton("❌ Отмена", callback_data="pos"),
                        ]
                    ]
                ),
            )

        elif data == "closeall_confirm":
            if not self._on_close_all:
                await self._edit(query, "❌ Функция закрытия недоступна.", _kb_pos())
                return
            await self._edit(
                query,
                "⏳ *Закрываю позиции...*",
                InlineKeyboardMarkup([]),
            )
            try:
                closed = await self._on_close_all()
            except Exception as _e:
                await self._edit(
                    query, f"❌ *Ошибка при закрытии позиций:*\n`{_e}`", _kb_pos()
                )
                return
            if closed:
                await self._edit(
                    query,
                    f"✅ *Закрыто {closed} позиций*\n\n"
                    f"Redis обновлён. Бот продолжит торговлю.",
                    _kb_pos(),
                )
            else:
                await self._edit(
                    query, "📭 Позиций для закрытия не оказалось.", _kb_pos()
                )

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

        elif data == "restart_bot":
            await self._edit(
                query,
                "🔄 *Перезапуск бота...*\n\n"
                "Бот остановится и Docker поднимет его заново.\n"
                "Это займёт ~10 секунд.",
                InlineKeyboardMarkup([]),
            )
            if self._on_restart:
                _t = asyncio.create_task(self._on_restart())
                self._background_tasks.add(_t)
                _t.add_done_callback(self._background_tasks.discard)

        elif data == "toggle_auto_exec":
            self._rc.set_auto_execute(not self._rc.get_auto_execute())
            text, kb = self._build_settings()
            await self._edit(query, text, kb)

        elif data == "mode_menu":
            mode = self._rc.get_mode()
            mode_label = self._MODE_DISPLAY.get(mode, mode.upper())
            await self._edit(
                query, f"Текущий режим: `{mode_label}`\nВыбери новый:", _kb_mode_menu()
            )

        elif data == "scan_menu":
            n = self._rc.get_scan_top_n()
            await self._edit(query, f"Сейчас: `{n}` символов\nВыбери:", _kb_scan_menu())

        elif data.startswith("mode:"):
            mode = data.split(":", 1)[1]
            if self._rc.set_mode(mode):
                if mode == "hybrid":
                    await self._edit(
                        query,
                        "🔀 *Режим переключён: Hybrid*\n\n"
                        "Включить *усиленный контроль*?\n"
                        "SAC + LLM + Chronos — тройное подтверждение.\n"
                        "Chronos мягко корректирует confidence "
                        "(не отменяет сделку жёстко) и работает "
                        "в paper и live одинаково.",
                        _kb_chronos_prompt(),
                    )
                elif mode == "local":
                    strat_text, strat_kb = self._build_strategies()
                    warning = (
                        "✅ *Режим переключён: Local*\n\n"
                        "⚠️ *Внимание!* Только математические стратегии — без AI.\n"
                        "Бот не анализирует новости, макро-события и сентимент.\n"
                        "🚫 *Не рекомендуется для реальной торговли.* "
                        "Используй на свой страх и риск.\n\n"
                        "💡 *Совет:* В режиме AI эти же стратегии работают как "
                        "контекстное меню — передаются в промпт и помогают "
                        "нейросети принять более точное решение.\n\n"
                        "─────────────\n\n"
                    )
                    await self._edit(query, warning + strat_text, strat_kb)
                else:
                    text, kb = self._build_settings()
                    await self._edit(query, f"✅ Режим → `{mode}`\n\n" + text, kb)
            else:
                await self._edit(query, "❌ Неверный режим", _kb_mode_menu())

        elif data == "toggle_season_mode":
            current = self._rc.get_season_switch_mode()
            new_mode = "auto" if current == "alert" else "alert"
            self._rc.set_season_switch_mode(new_mode)
            if new_mode == "auto":
                text, kb = self._build_settings()
                await self._edit(
                    query,
                    "⚠️ *Авто-переключение сезона включено*\n\n"
                    "Бот будет самостоятельно менять профиль рынка\n"
                    "(Блючипы ↔ Альткоины) когда CoinGecko фиксирует смену сезона.\n\n"
                    "*Возможные риски:*\n"
                    "— Ложные сигналы могут переключить профиль в неподходящий момент\n"
                    "— SAC модель обучена на конкретном профиле — после смены\n"
                    "  торговля идёт на AI/Local до переобучения модели\n"
                    "— Рекомендуется иметь обученные модели для обоих профилей\n\n"
                    "_Для отключения нажми кнопку ещё раз._\n\n" + text,
                    kb,
                )
            else:
                text, kb = self._build_settings()
                await self._edit(
                    query,
                    "✅ *Режим алерт* — бот будет присылать уведомление\n"
                    "с кнопками когда обнаружит смену сезона.\n\n" + text,
                    kb,
                )

        elif data in ("chronos_on", "chronos_off", "toggle_chronos"):
            if data == "toggle_chronos":
                enabled = not self._rc.get_chronos_enabled()
            else:
                enabled = data == "chronos_on"
            self._rc.set_chronos_enabled(enabled)
            text, kb = self._build_settings()
            status = "🛡 Усиленный контроль ВКЛ" if enabled else "⚡ Стандартный режим"
            await self._edit(query, f"✅ {status}\n\n" + text, kb)

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
            kb = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "✅ Да, сбросить", callback_data="reset_defaults_confirm"
                        )
                    ],
                    [InlineKeyboardButton("❌ Отмена", callback_data="settings")],
                ]
            )
            await self._edit(
                query,
                "⚠️ *Сброс настроек*\n\n"
                "Все runtime\\-настройки вернутся к значениям из `.env`\\.\n"
                "Режим, пауза, символы, риск — всё будет сброшено\\.\n\n"
                "Вы уверены?",
                kb,
            )

        elif data == "reset_defaults_confirm":
            self._rc.reset_to_defaults()
            text, kb = self._build_settings()
            await self._edit(
                query,
                "🔄 *Настройки сброшены к значениям по умолчанию*\n\n" + text,
                kb,
            )

        # ── Переключение Paper ↔ Live ─────────────────────────────────────────
        elif data == "switch_trading_mode":
            target = (
                "💰 LIVE (реальная торговля)"
                if Config.PAPER_TRADING
                else "📄 PAPER (симуляция)"
            )
            warning = (
                "⚠️ *ВНИМАНИЕ!* Это включит реальную торговлю с реальными деньгами!"
                if Config.PAPER_TRADING
                else "✅ Бот перейдёт в режим симуляции."
            )
            if Config.TRADING_MODE_PIN:
                self._rc.set_awaiting_mode_pin(ttl=120)
                await self._edit(
                    query,
                    f"🔐 *Подтверждение смены режима*\n\n"
                    f"Вы собираетесь переключиться на: *{target}*\n\n"
                    f"{warning}\n\n"
                    f"Введите кодовое слово в чат (у вас есть *2 минуты*):",
                    InlineKeyboardMarkup(
                        [[InlineKeyboardButton("❌ Отмена", callback_data="settings")]]
                    ),
                )
            else:
                await self._edit(
                    query,
                    f"🔄 *Смена режима торговли*\n\n"
                    f"Вы собираетесь переключиться на: *{target}*\n\n"
                    f"{warning}\n\n"
                    f"Подтвердите переключение:",
                    InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton(
                                    "✅ Подтвердить",
                                    callback_data="confirm_mode_switch",
                                ),
                                InlineKeyboardButton(
                                    "❌ Отмена", callback_data="settings"
                                ),
                            ]
                        ]
                    ),
                )

        elif data == "confirm_mode_switch":
            new_paper = not Config.PAPER_TRADING
            self._rc.apply_paper_trading(new_paper)
            # Очищаем paper-позиции и синхронизируем с биржей
            if self._on_mode_switch:
                try:
                    await self._on_mode_switch()
                except Exception as _e:
                    logger.warning("on_mode_switch callback failed: %s", _e)
            mode_name = (
                "📄 PAPER (симуляция)" if new_paper else "💰 LIVE (реальная торговля)"
            )
            detail = (
                "⚠️ Бот теперь торгует на реальные деньги."
                if not new_paper
                else "✅ Бот переведён в режим симуляции. Реальных сделок нет."
            )
            await self._edit(
                query,
                f"🚨 *ВАЖНО! РЕЖИМ ТОРГОВЛИ ИЗМЕНЁН*\n\n"
                f"Текущий режим: *{mode_name}*\n\n"
                f"{detail}\n\n"
                f"Позиции синхронизированы. Нажми *Перезапустить* чтобы"
                f" применить все изменения чисто.",
                InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "🔄 Перезапустить бота", callback_data="restart_bot"
                            )
                        ],
                        [InlineKeyboardButton("« Главная", callback_data="main")],
                    ]
                ),
            )

        # ── Профиль рынка ────────────────────────────────────────────────────
        elif data == "market_profile_menu":
            import os as _os

            current = self._rc.get_market_profile()
            bc_has = _os.path.exists("models/sac_model.zip")
            alt_has = _os.path.exists("models/sac_model_altcoin.zip")
            meme_has = _os.path.exists("models/sac_model_meme.zip")
            model_legend = "🧠 — модель есть  |  ⚠️ — нужно обучить"
            await self._edit(
                query,
                "🎯 *Профиль рынка*\n\n"
                "🔵 *Блючипы* — BTC, ETH, SOL и другие топ-монеты.\n"
                "Таймфрейм 15m · топ-20 · риск 2% · Hybrid · цикл 15 мин\n\n"
                "🟡 *Альткоины* — середина рынка, топ-100 по объёму.\n"
                "Таймфрейм 5m · топ-100 (30 лучших → AI)"
                " · риск 1% · Hybrid · цикл 5 мин\n\n"
                "🚀 *Мемкоины* — памп-монеты, топ-200 по объёму.\n"
                "Таймфрейм 3m · топ-200 · риск 0.5% · Hybrid · цикл 90 сек\n"
                "SL 12% · TP 72% · сентимент Twitter+Telegram\n\n"
                f"_{model_legend}_\n"
                "_Профиль применяется мгновенно."
                " Интервал цикла — со следующего цикла._",
                _kb_market_profile_menu(current, bc_has, alt_has, meme_has),
            )

        elif data == "season_dismiss":
            await self._edit(
                query,
                "👌 Профиль оставлен без изменений.\n\n"
                "_Следующее уведомление — не раньше чем через 24ч,_\n"
                "_если сигнал сохранится._",
                _kb_main(),
            )

        elif data.startswith("market_profile:"):
            import os as _os

            profile_name = data.split(":", 1)[1]
            if self._rc.apply_market_profile(profile_name):
                labels = {
                    "bluechip": "🔵 Блючипы",
                    "altcoin": "🟡 Альткоины",
                    "meme": "🚀 Мемкоины",
                }
                label = labels.get(profile_name, profile_name)
                profile_cfg = self._rc.get_market_profile_config(profile_name)
                profile_mode = profile_cfg.get("mode", "hybrid")
                model_path = self._rc.get_sac_model_path()
                model_exists = _os.path.exists(model_path)
                # local-mode profiles don't use SAC — skip the missing-model warning
                if profile_mode == "local":
                    model_note = ""
                elif not model_exists:
                    model_note = (
                        f"\n\n⚠️ SAC модель для этого профиля не найдена.\n"
                        f"📁 `{model_path}`\n"
                        "_Используй кнопку обучения в настройках._"
                    )
                else:
                    model_note = ""
                # предупреждение уже показано inline — фоновый check не нужен
                if not model_exists and profile_mode != "local":
                    self._rc.set_sac_prompted(profile_name)
                text, kb = self._build_settings()
                await self._edit(
                    query,
                    f"✅ Профиль применён: *{label}*{model_note}\n\n" + text,
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

        elif data == "toggle_paper_guards":
            if not Config.PAPER_TRADING:  # в Live переключатель недоступен
                await query.answer("Только в Paper", show_alert=True)
                return
            self._rc.set_paper_guards_off(not self._rc.get_paper_guards_off())
            text, kb = self._build_risk()
            await self._edit(query, text, kb)

        elif data == "toggle_drawdown":
            current = self._rc.get_drawdown_scale_enabled()
            self._rc.set_drawdown_scale_enabled(not current)
            text, kb = self._build_risk()
            icon = "✅" if not current else "❌"
            await self._edit(query, f"{icon} Защита просадки изменена\n\n" + text, kb)

        elif data in ("dd_confirm_more", "dd_confirm_less", "dd_confirm_noop"):
            if data == "dd_confirm_noop":
                await query.answer()
                return
            cur = self._rc.get_drawdown_confirm_cycles()
            new_val = min(cur + 1, 10) if data == "dd_confirm_more" else max(cur - 1, 1)
            self._rc.set_drawdown_confirm_cycles(new_val)
            text, kb = self._build_risk()
            await self._edit(
                query,
                f"🛡 Flash-защита: `{new_val}` цикл. × 30с = `{new_val * 30}с`\n\n"
                + text,
                kb,
            )

        elif data in ("pos_more", "pos_less", "pos_paper_max", "pos_noop"):
            if data == "pos_noop":
                await query.answer()
                return
            cur = self._rc.get_max_positions()
            if data == "pos_more":
                new_val = min(cur + 1, 20)
            elif data == "pos_less":
                new_val = max(cur - 1, 1)
            else:
                new_val = 15
            self._rc.set_max_positions(new_val)
            text, kb = self._build_risk()
            await self._edit(query, f"✅ Макс. позиций: `{new_val}`\n\n" + text, kb)

        # ── Порог сигнала (внутри риск-меню) ────────────────────────────────
        elif data in (
            "conf_live_up",
            "conf_live_down",
            "conf_paper_up",
            "conf_paper_down",
        ):
            paper = data.startswith("conf_paper")
            delta = 0.05 if data.endswith("_up") else -0.05
            current = self._rc.get_signal_confidence(paper=paper)
            ok = self._rc.set_signal_confidence(round(current + delta, 2), paper=paper)
            text, kb = self._build_risk()
            note = "⚠️ Значение за пределами 0.40–0.95\n\n" if not ok else ""
            await self._edit(query, note + text, kb)

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
            dd = self._rc.get_leverage_dd_threshold()
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
            dd = self._rc.get_leverage_dd_threshold()
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
            dd = self._rc.get_leverage_dd_threshold()
            await self._edit(
                query,
                f"✅ Риск на ATR: `{val_f*100:.1f}%`\n\n_Выбери режим или параметр:_",
                _kb_lev_menu(mode, val_f, dd),
            )

        elif data.startswith("lev:drawdown:"):
            val_f = float(data.split(":", 2)[2])
            self._rc.set_leverage_dd_threshold(val_f)
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
            dd = self._rc.get_leverage_dd_threshold()
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

        # ── Тип рынка (spot / futures) ──────────────────────────────────────
        elif data == "market_type_menu":
            current = self._rc.get_market_type()
            label = "💵 Spot" if current == "spot" else "📐 Futures"
            positions = self._get_state().get("positions", [])
            _warn = (
                f"\n\n⚠️ Открыто позиций: {len(positions)} — переключение "
                f"заблокировано, пока они не закроются."
                if positions
                else ""
            )
            await self._edit(
                query,
                f"💱 *Тип рынка*\n\nТекущий: *{label}*\n\n"
                f"Spot — только лонги, без плеча и ликвидации.\n"
                f"Futures — лонг/шорт, плечо, funding rate.{_warn}",
                _kb_market_type_menu(current),
            )

        elif data.startswith("market_type:"):
            val = data.split(":", 1)[1]
            positions = self._get_state().get("positions", [])
            if positions:
                await self._edit(
                    query,
                    f"❌ Нельзя сменить тип рынка при {len(positions)} "
                    f"открытых позициях — закрой их сначала.",
                    _kb_market_type_menu(self._rc.get_market_type()),
                )
                return
            ok = self._rc.set_market_type(val)
            if not ok:
                await self._edit(
                    query,
                    f"❌ Неизвестный тип рынка: *{val}*",
                    _kb_market_type_menu(self._rc.get_market_type()),
                )
                return
            label = "💵 Spot" if val == "spot" else "📐 Futures"
            await self._edit(
                query,
                f"✅ Тип рынка: *{label}*\n\n_Применится к следующему циклу._",
                _kb_market_type_menu(val),
            )

        # ── AI-провайдер ─────────────────────────────────────────────────────
        elif data == "provider_menu":
            import time as _time
            from datetime import datetime, timezone

            current = self._rc.get_ai_provider()
            last = self._rc.get_last_ai_provider()
            _settings = self._rc.get_ai_chain()
            _keyed = list(
                dict.fromkeys(
                    [
                        p.id
                        for p in ai_providers.PROVIDERS
                        if p.key_env and os.getenv(p.key_env)
                    ]
                    + list(_settings.get("keyless", []))
                )
            )
            _chain = ai_providers.chain(_settings, _keyed, pinned=current)
            _mode = {
                "free": "только бесплатные",
                "auto": "сначала бесплатные",
                "paid": "только платные",
            }[_settings.get("mode", ai_providers.DEFAULT_MODE)]
            _names = " → ".join(ai_providers.BY_ID[p].name for p, _ in _chain)
            desc = (
                f"{_names or 'нет доступных провайдеров'} ({_mode})"
                if current == "auto"
                else ai_providers.BY_ID[current].note
            )

            _utc = datetime.now(timezone.utc)
            _secs = 86400 - _utc.hour * 3600 - _utc.minute * 60 - _utc.second
            _h, _m = divmod(_secs // 60, 60)
            _reset = f"{_h}ч {_m}мин"

            _STATUS_ICON = {
                "ok": "✅",
                "no_balance": "💸",
                "rate_limit": "⏳",
                "bad_key": "🔑",
                "bad_model": "❔",
                "unknown": "❓",
            }
            _STATUS_DESC = {
                "ok": "работает",
                "no_balance": "нет баланса — пополни",
                "rate_limit": f"лимит — сброс через {_reset}",
                "bad_key": "ключ не принят",
                "bad_model": "модель не найдена — выбери в приложении",
                "bad_format": "ответ не в нужном формате",
                "bad_request": "отклонил запрос",
                "error": "ошибка связи",
                "unknown": "ещё не использовался",
            }
            lines = []
            for _p in _keyed:
                _st = self._rc.get_provider_status(_p)
                _icon = _STATUS_ICON.get(_st, "⚠️")
                _sdesc = _STATUS_DESC.get(_st, _st)
                _extra = ""
                if _p == "deepseek":
                    _bal = self._rc.get_deepseek_balance()
                    if _bal:
                        _extra = f" — баланс ${_bal['balance']:.2f}"
                else:
                    _used = self._rc.get_token_usage_today(_p)
                    if _used:
                        _extra = f" — сегодня {_used:,} токенов".replace(",", " ")
                _until = self._rc.get_provider_cooldown(_p)
                if _until > _time.time():
                    _at = datetime.fromtimestamp(_until).strftime("%H:%M")
                    _extra += f" — пауза до {_at}"
                _name = ai_providers.BY_ID[_p].name
                lines.append(f"{_icon} *{_name}* — {_sdesc}{_extra}")
            _missing = len(ai_providers.PROVIDERS) - len(_keyed)
            if _missing:
                lines.append(
                    f"➕ Ещё {_missing} провайдеров без ключа — "
                    "добавьте в приложении: Настройки → AI"
                )
            providers_text = "\n".join(lines)
            _last_str = f"Активен сейчас: `{last}`\n\n" if last else ""
            await self._edit(
                query,
                f"🤖 *AI-провайдеры*\n\n"
                f"{providers_text}\n\n"
                f"{_last_str}"
                f"Режим: *{current}* — {desc}\n\n"
                + (
                    "_Один провайдер: решения принимает только он, без запасных; "
                    "недоступен — работают локальные стратегии._"
                    if _settings.get("single")
                    else "_Цепочка: при сбое — автоматически следующий провайдер._"
                ),
                _kb_provider_menu(current, bool(_settings.get("single"))),
            )

        elif data.startswith("ai_single:"):
            single = data.endswith(":1")
            self._rc.set_ai_chain({"single": single})
            current = self._rc.get_ai_provider()
            await self._edit(
                query,
                (
                    "🎯 *Один провайдер*: решения всегда принимает "
                    f"`{current if current != 'auto' else 'первый в цепочке'}`, "
                    "без запасных. Недоступен — работают локальные стратегии."
                    if single
                    else "🔗 *Цепочка*: при сбое или лимите — сразу следующий "
                    "провайдер."
                )
                + "\n\n_Применится к следующему циклу анализа._",
                _kb_provider_menu(current, single),
            )

        elif data.startswith("provider:"):
            val = data.split(":", 1)[1]
            ok = self._rc.set_ai_provider(val)
            if not ok:
                await self._edit(
                    query,
                    f"❌ Неизвестный провайдер: *{val}*",
                    _kb_provider_menu(
                        self._rc.get_ai_provider(),
                        bool(self._rc.get_ai_chain().get("single")),
                    ),
                )
                return
            desc = (
                "цепочка — порядок и модели настраиваются в приложении"
                if val == "auto"
                else ai_providers.BY_ID[val].note
            )
            await self._edit(
                query,
                f"✅ AI-провайдер: *{val}*\n{desc}\n\n"
                f"_Применится к следующему циклу анализа._",
                _kb_provider_menu(val, bool(self._rc.get_ai_chain().get("single"))),
            )

        # ── Стратегии ────────────────────────────────────────────────────────
        elif data == "strategies":
            text, kb = self._build_strategies()
            await self._edit(query, text, kb)

        elif data == "strat_reset":
            self._rc.reset_strategies()
            text, kb = self._build_strategies()
            await self._edit(
                query,
                "🔁 *Основные стратегии включены*\n"
                "_Экспериментальные остаются выключенными._\n\n" + text,
                kb,
            )

        elif data.startswith("strat_exp:"):
            try:
                page = int(data.split(":", 1)[1])
            except ValueError:
                page = 0
            text, kb = self._build_strategies(page)
            await self._edit(query, text, kb)

        elif data.startswith("strat:"):
            parts = data.split(":")
            name = parts[1]
            now_enabled = self._rc.toggle_strategy(name)
            icon = "✅" if now_enabled else "❌"
            page = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
            text, kb = self._build_strategies(page)
            short = _STRAT_SHORT.get(name, name)
            state = "включена" if now_enabled else "отключена"
            await self._edit(query, f"{icon} `{short}` {state}\n\n" + text, kb)

        # ── Бэктест ───────────────────────────────────────────────────────────
        elif data == "backtest_menu":
            await self._edit(
                query,
                "📊 *Бэктест и отчёты*\n\n"
                "▶ *Бэктест* — проверяет стратегии на 3-6 мес. исторических данных.\n"
                "📌 Когда: перед реальными деньгами, после тюнинга SAC. ~5-10 мин.\n\n"
                "📈 *Значимость альфы* — bootstrap Sharpe + Wilcoxon на реально\n"
                "закрытых paper-сделках (не симуляция). Секунды.\n\n"
                "📐 *Live vs Backtest* — совпадает ли реальная цена исполнения\n"
                "с тем, что предсказывала модель импакта. Секунды.\n\n"
                "🤖 *AI-изоляция* — даёт ли AI-слой эдж отдельно от индикаторов.\n"
                "⚠️ Тратит реальные лимиты AI-провайдера. Несколько минут.\n\n"
                "Бот продолжает торговать во всех случаях.",
                _kb_backtest_menu(),
            )

        elif data == "backtest_now":
            if self._backtesting or not self._rc.try_start_job(
                "backtest", "telegram", 3 * 3600
            ):
                await query.answer("⏳ Бэктест уже выполняется...", show_alert=True)
                return
            self._backtesting = True
            await self._edit(
                query,
                "⏳ *Бэктест запущен*\n\n"
                "Бот продолжает торговать.\nПришлю результаты когда готово.",
                _kb_after_action(),
            )
            import asyncio as _asyncio

            task = _asyncio.ensure_future(self._run_backtest())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        elif data == "alpha_report_now":
            if self._alpha_report_running or not self._rc.try_start_job(
                "alpha_report", "telegram", 3600
            ):
                await query.answer("⏳ Отчёт уже выполняется...", show_alert=True)
                return
            self._alpha_report_running = True
            await self._edit(
                query,
                "⏳ *Отчёт значимости альфы запущен*\n\n"
                "Bootstrap Sharpe + Wilcoxon на реальных paper-сделках.\n"
                "Пришлю результат через несколько секунд.",
                _kb_after_action(),
            )
            import asyncio as _asyncio

            task = _asyncio.ensure_future(self._run_alpha_report())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        elif data == "live_divergence_now":
            if self._live_divergence_running or not self._rc.try_start_job(
                "live_divergence", "telegram", 3600
            ):
                await query.answer("⏳ Отчёт уже выполняется...", show_alert=True)
                return
            self._live_divergence_running = True
            await self._edit(
                query,
                "⏳ *Отчёт Live vs Backtest запущен*\n\n"
                "Сравниваю реальные сделки с моделью market impact.\n"
                "Пришлю результат через несколько секунд.",
                _kb_after_action(),
            )
            import asyncio as _asyncio

            task = _asyncio.ensure_future(self._run_live_divergence())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        elif data == "ai_isolation_menu":
            await self._edit(
                query,
                "🤖 *AI-изоляция (вклад AI-слоя отдельно от индикаторов)*\n\n"
                "Прогоняет отдельный бэктест, который вызывает настоящего\n"
                "AI-провайдера (~150 вызовов по умолчанию), чтобы понять,\n"
                "даёт ли AI-слой эдж, которого нет у голых индикаторов.\n\n"
                "⚠️ *В отличие от остальных отчётов в этом меню — тратит\n"
                "реальные лимиты/квоту твоего AI-провайдера.*\n\n"
                "⏱ Обычно несколько минут, зависит от провайдера.\n"
                "Бот продолжает торговать.",
                _kb_ai_isolation_confirm(),
            )

        elif data == "ai_isolation_now":
            if self._ai_isolation_running or not self._rc.try_start_job(
                "ai_isolation", "telegram", 3600
            ):
                await query.answer("⏳ Уже выполняется...", show_alert=True)
                return
            self._ai_isolation_running = True
            await self._edit(
                query,
                "⏳ *AI-изоляция запущена*\n\n"
                "Расходует лимиты AI-провайдера. Бот продолжает торговать.\n"
                "Пришлю результат когда готово.",
                _kb_after_action(),
            )
            import asyncio as _asyncio

            task = _asyncio.ensure_future(self._run_ai_isolation_backtest())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        elif data == "ai_isolation_skip":
            await self._edit(
                query,
                "⏭ *AI-изоляция отменена*\n\n" "Вернуться можно в 📊 Бэктест и отчёты.",
                _kb_backtest_menu(),
            )

        # ── Таймаут подтверждения ─────────────────────────────────────────────
        elif data == "timeout_menu":
            cur = self._rc.get_confirm_timeout()
            auto_exec = self._rc.get_auto_execute()
            on_timeout = self._rc.get_confirm_on_timeout()
            if auto_exec:
                cur_label = "🤖 Авто"
                cur_hint = "нашёл сигнал — сразу открывает сделку"
            else:
                cur_label = f"{cur}с"
                _what = "открыть" if on_timeout == "execute" else "пропустить"
                cur_hint = f"{cur}с на решение, без ответа — {_what}"
            await self._edit(
                query,
                f"⏱ *Подтверждение сделок*\n\n"
                f"Сейчас: *{cur_label}* — {cur_hint}\n\n"
                f"🤖 *Авто* — без подтверждения: нашёл сигнал → сделка\n"
                f"*15–120с* — время на решение (дольше нельзя: цена уходит)\n"
                f"*Без ответа* — открыть сделку или пропустить\n\n"
                f"_В Paper бот не спрашивает — сделки виртуальные._",
                _kb_timeout_menu(cur, auto_exec, on_timeout),
            )

        elif data.startswith("timeout:"):
            raw = data.split(":", 1)[1]
            on_timeout = self._rc.get_confirm_on_timeout()
            if raw == "auto":
                self._rc.set_auto_execute(True)
                await self._edit(
                    query,
                    "✅ *Авто-режим* — нашёл сигнал, сразу открывает сделку.\n"
                    "Кнопки Trade/Skip не показываются.\n\n"
                    "_Применится к следующей сделке._",
                    _kb_timeout_menu(self._rc.get_confirm_timeout(), True, on_timeout),
                )
                return
            try:
                sec = int(raw)
            except (ValueError, IndexError):
                return
            if self._rc.set_confirm_timeout(sec):
                self._rc.set_auto_execute(False)
                _what = "открою" if on_timeout == "execute" else "пропущу"
                await self._edit(
                    query,
                    f"✅ *{sec}с* на решение. Без ответа — {_what} сделку.\n\n"
                    f"_Применится к следующей сделке._",
                    _kb_timeout_menu(sec, False, on_timeout),
                )

        elif data.startswith("on_timeout:"):
            value = data.split(":", 1)[1]
            if self._rc.set_confirm_on_timeout(value):
                self._rc.set_auto_execute(False)
                cur = self._rc.get_confirm_timeout()
                _what = "открою сделку" if value == "execute" else "пропущу сделку"
                await self._edit(
                    query,
                    f"✅ Без ответа за {cur}с — {_what}.\n\n"
                    f"_Применится к следующей сделке._",
                    _kb_timeout_menu(cur, False, value),
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
            if self._sac_training or not self._rc.try_start_job(
                "sac", "telegram", 8 * 3600
            ):
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
                "⏭ *Тюнинг отменён*\n\n" "Вернуться к тюнингу можно в ⚙️ Настройки.",
                _kb_main(),
            )

        # ── Прогресс обучения (SAC + бэктест) ────────────────────────────────
        elif data in ("train_progress", "backtest_progress"):
            # Авто-сброс если все фоновые задачи завершились
            if self._sac_training and all(t.done() for t in self._background_tasks):
                self._sac_training = False
                self._rc.end_job("sac")
                self._rc.clear_train_progress()
                self._rc.clear_tune_progress()

            tune_prog = self._rc.get_tune_progress()
            sac_prog = self._rc.get_train_progress()
            bt_prog = self._rc.get_backtest_progress()
            anything_running = self._sac_training or self._backtesting
            sections: list[str] = []

            if self._sac_training:
                # Показываем тюнинг если он активен
                if tune_prog:
                    trial = tune_prog.get("trial", 0)
                    total = tune_prog.get("total", 0)
                    pct = tune_prog.get("pct", 0.0)
                    bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
                    best_pnl = tune_prog.get("best_pnl")
                    eta = tune_prog.get("eta_min")
                    pnl_str = f"${best_pnl:,.0f}" if best_pnl is not None else "—"
                    eta_str = f"~{eta} мин" if eta is not None else "..."
                    sections.append(
                        f"🔬 *Тюнинг SAC*\n"
                        f"`[{bar}]` *{pct:.1f}%*\n"
                        f"Попытка: `{trial}` / `{total}`\n"
                        f"Лучший PnL: *{pnl_str}*\n"
                        f"Осталось: *{eta_str}*"
                    )
                elif sac_prog:
                    pct = sac_prog.get("pct", 0)
                    bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
                    eta_min = sac_prog.get("eta_min")
                    eta_str = f"~{eta_min} мин" if eta_min is not None else "..."
                    step = sac_prog.get("step", 0)
                    total = sac_prog.get("total", 0)
                    sections.append(
                        f"🧠 *Обучение SAC*\n"
                        f"`[{bar}]` *{pct:.1f}%*\n"
                        f"Шаги: `{step:,}` / `{total:,}`\n"
                        f"Осталось: *{eta_str}*"
                    )
                else:
                    sections.append(
                        "🔬 *Тюнинг / Обучение SAC*\n"
                        "_Запущено, прогресс появится через минуту..._"
                    )

            if self._backtesting:
                if bt_prog:
                    pct = bt_prog.get("pct", 0)
                    bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
                    sections.append(
                        f"📊 *Бэктест*\n"
                        f"`[{bar}]` *{pct:.1f}%*\n"
                        f"Символ: `{bt_prog.get('symbol', '...')}`\n"
                        f"Стратегия: `{bt_prog.get('strategy', '...')}`\n"
                        f"Шаг: `{bt_prog.get('step', 0)}` / `{bt_prog.get('total', 0)}`"
                    )
                else:
                    sections.append(
                        "📊 *Бэктест*\n_Запущено, прогресс появится через момент..._"
                    )

            if not anything_running:
                sections.append(
                    "Нет активных процессов.\n\n"
                    "_Запусти обучение SAC или бэктест из Настроек._"
                )

            text = "📈 *Прогресс обучения*\n\n" + "\n\n─────────────\n\n".join(sections)
            kb_rows = []
            if anything_running:
                kb_rows.append(
                    [
                        InlineKeyboardButton(
                            "🔄 Обновить", callback_data="train_progress"
                        )
                    ]
                )
            kb_rows.append(
                [InlineKeyboardButton("« Настройки", callback_data="settings")]
            )
            await self._edit(query, text, InlineKeyboardMarkup(kb_rows))

        elif data == "train_sac_now":
            if self._sac_training or not self._rc.try_start_job(
                "sac", "telegram", 4 * 3600
            ):
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

        # ── Уведомления ──────────────────────────────────────────────────────
        elif data == "alerts_menu":
            await self._edit(query, _ALERTS_MENU_TEXT, _kb_alerts(self._rc))

        elif data.startswith("alert_toggle:"):
            name = data[len("alert_toggle:") :]
            if name in self._rc._ALERT_DEFAULTS:
                current = self._rc.get_alert_enabled(name)
                self._rc.set_alert_enabled(name, not current)
            await self._edit(query, _ALERTS_MENU_TEXT, _kb_alerts(self._rc))

        elif data.startswith("alert_cooldown:"):
            name = data[len("alert_cooldown:") :]
            if name in self._rc._ALERT_DEFAULTS:
                self._rc.set_awaiting_alert_cooldown(name)
                label = self._rc._ALERT_DEFAULTS[name]["label"]
                current = self._rc.get_alert_cooldown_mins(name)
                await self._edit(
                    query,
                    f"⏱ *Cooldown для {label}*\n\n"
                    f"Текущее: *{current} мин*\n\n"
                    "Введите новое значение в минутах (1–1440):",
                    InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton(
                                    "« Отмена", callback_data="alerts_menu"
                                )
                            ]
                        ]
                    ),
                )

        elif data == "rollback_sac":
            import os as _os
            import shutil as _shutil

            backup = self._rc.get_sac_backup_path()
            model = self._rc.get_sac_model_path()
            if not backup or not _os.path.exists(backup):
                await self._edit(
                    query,
                    "❌ *Бэкап не найден*\n\nФайл резервной копии отсутствует.",
                    _kb_after_action(),
                )
            else:
                _shutil.copy2(backup, model)
                await self._edit(
                    query,
                    f"✅ *Модель откатана*\n\n"
                    f"Восстановлена из: `{backup}`\n"
                    "Бот подхватит старую модель в следующем цикле.",
                    _kb_after_action(),
                )

        else:
            await query.answer("Неизвестная команда")
