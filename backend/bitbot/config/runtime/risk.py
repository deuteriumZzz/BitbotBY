"""Риск: размер риска на сделку, пресеты, просадка, плечо, настройки тревог.
"""

from __future__ import annotations

import logging
from typing import Optional

from bitbot.config import keys
from bitbot.config.runtime.presets import _LEVERAGE_MODES, RISK_PRESETS, _hour_in
from bitbot.config.runtime.trading import TradingSettings
from bitbot.config.settings import Config

logger = logging.getLogger(__name__)


class RiskSettings(TradingSettings):
    # ── Alert settings ────────────────────────────────────────────────────────
    _ALERT_DEFAULTS: dict = {
        # level: critical — приходит всегда, даже в тихие часы; trade и info — нет.
        "atr_spike": {"label": "⚡ ATR Spike", "cooldown": 10, "level": "critical"},
        "drawdown": {"label": "📉 Drawdown", "cooldown": 5, "level": "critical"},
        "daily_limit": {"label": "🔒 Daily limit", "cooldown": 60, "level": "critical"},
        "bot_stalled": {"label": "🔇 BotStalled", "cooldown": 20, "level": "critical"},
        "new_signal": {"label": "📊 Новые сигналы", "cooldown": 0, "level": "info"},
        "position_open": {
            "label": "📈 Открытие позиции",
            "cooldown": 0,
            "level": "trade",
        },
        "position_close": {
            "label": "💰 Закрытие позиции",
            "cooldown": 0,
            "level": "trade",
        },
    }

    # ── Risk management ───────────────────────────────────────────────────────
    def get_max_positions(self) -> int:
        val = self._get(keys.MAX_POSITIONS)
        try:
            return max(1, min(int(val), 20)) if val else Config.MAX_POSITIONS
        except (ValueError, TypeError):
            return Config.MAX_POSITIONS

    def set_max_positions(self, n: int) -> bool:
        if not (1 <= n <= 20):
            return False
        self._set(keys.MAX_POSITIONS, str(n))
        logger.info("Runtime: max_positions → %d", n)
        return True

    def get_risk_per_trade(self) -> float:
        val = self._get(keys.RISK_PER_TRADE)
        try:
            v = float(val) if val else Config.RISK_PER_TRADE
            return max(0.001, min(v, 0.5))
        except (ValueError, TypeError):
            return Config.RISK_PER_TRADE

    def set_risk_per_trade(self, v: float) -> bool:
        if not (0.001 <= v <= 0.5):
            return False
        self._set(keys.RISK_PER_TRADE, str(v))
        logger.info("Runtime: risk_per_trade → %.1f%%", v * 100)
        return True

    def get_sl_percent(self) -> float:
        """SL % для текущего профиля (bluechip=5%, altcoin=4%)."""
        try:
            val = self._get(keys.SL_PERCENT)
            return float(val) if val else Config.STOP_LOSS_PERCENT
        except (ValueError, TypeError):
            return Config.STOP_LOSS_PERCENT

    def get_tp_multiplier(self) -> float:
        """Множитель TP относительно SL (bluechip=2x, altcoin=4x)."""
        try:
            val = self._get(keys.TP_MULTIPLIER)
            return float(val) if val else 2.0
        except (ValueError, TypeError):
            return 2.0

    def get_trading_interval(self) -> int:
        """Интервал торгового цикла в секундах (из профиля или умолчание)."""
        try:
            val = self._get(keys.TRADING_INTERVAL)
            return int(val) if val else Config.TRADING_INTERVAL
        except (ValueError, TypeError):
            return Config.TRADING_INTERVAL

    def get_signal_confidence(self, paper: bool) -> float:
        key = keys.SIGNAL_CONFIDENCE_PAPER if paper else keys.SIGNAL_CONFIDENCE_LIVE
        default = (
            Config.MIN_SIGNAL_CONFIDENCE_PAPER
            if paper
            else Config.MIN_SIGNAL_CONFIDENCE
        )
        val = self._get(key)
        try:
            v = float(val) if val else default
            return max(0.40, min(v, 0.95))
        except (ValueError, TypeError):
            return default

    def set_signal_confidence(self, v: float, paper: bool) -> bool:
        if not (0.40 <= v <= 0.95):
            return False
        key = keys.SIGNAL_CONFIDENCE_PAPER if paper else keys.SIGNAL_CONFIDENCE_LIVE
        self._set(key, str(round(v, 2)))
        mode = "paper" if paper else "live"
        logger.info("Runtime: signal_confidence[%s] → %.2f", mode, v)
        return True

    def get_drawdown_scale_enabled(self) -> bool:
        val = self._get(keys.DRAWDOWN_SCALE_ENABLED)
        if val is None:
            return Config.DRAWDOWN_SCALE_ENABLED
        return val == "1"

    def set_drawdown_scale_enabled(self, enabled: bool) -> None:
        self._set(keys.DRAWDOWN_SCALE_ENABLED, "1" if enabled else "0")
        logger.info("Runtime: drawdown_scale → %s", "ВКЛ" if enabled else "ВЫКЛ")

    def apply_risk_preset(self, name: str) -> bool:
        """Применяет пресет риск-профиля. Возвращает False если имя неизвестно."""
        preset = RISK_PRESETS.get(name)
        if not preset:
            return False
        self.set_max_positions(int(preset["MAX_POSITIONS"]))
        self.set_risk_per_trade(float(preset["RISK_PER_TRADE"]))
        self.set_drawdown_scale_enabled(bool(preset["DRAWDOWN_SCALE_ENABLED"]))
        self.set_leverage_mode(str(preset["LEVERAGE_MODE"]))
        logger.info("Runtime: применён риск-профиль '%s'", name)
        return True

    def get_risk_summary(self) -> dict:
        return {
            "max_positions": self.get_max_positions(),
            "risk_per_trade": self.get_risk_per_trade(),
            "drawdown_scale_enabled": self.get_drawdown_scale_enabled(),
            "drawdown_confirm_cycles": self.get_drawdown_confirm_cycles(),
        }

    def get_drawdown_confirm_cycles(self) -> int:
        """Кол-во циклов подряд для подтверждения просадки перед halt (1–10)."""
        val = self._get(keys.DRAWDOWN_CONFIRM_CYCLES)
        try:
            return (
                max(1, min(int(val), 10))
                if val
                else int(getattr(Config, "DRAWDOWN_CONFIRM_CYCLES", 3))
            )
        except (ValueError, TypeError):
            return 3

    def set_drawdown_confirm_cycles(self, n: int) -> bool:
        if not (1 <= n <= 10):
            return False
        self._set(keys.DRAWDOWN_CONFIRM_CYCLES, str(n))
        logger.info("Runtime: drawdown_confirm_cycles → %d", n)
        return True

    # ── Leverage ──────────────────────────────────────────────────────────────
    def get_leverage_dd_threshold(self) -> float:
        """Порог просадки для режима плеча full (0.05–0.5).

        Не hard drawdown halt: тот читает Config.MAX_DRAWDOWN_PERCENT напрямую.
        """
        default = float(getattr(Config, "LEVERAGE_DD_THRESHOLD", 0.15))
        val = self._get(keys.LEVERAGE_DD_THRESHOLD)
        try:
            return max(0.05, min(float(val), 0.5)) if val else default
        except (ValueError, TypeError):
            return default

    def get_leverage_mode(self) -> str:
        """Режим плеча: fixed | volatility | full."""
        val = self._get(keys.LEVERAGE_MODE)
        if val and val in _LEVERAGE_MODES:
            return val
        return getattr(Config, "LEVERAGE_MODE", "volatility")

    def set_leverage_mode(self, mode: str) -> bool:
        if mode not in _LEVERAGE_MODES:
            return False
        self._set(keys.LEVERAGE_MODE, mode)
        logger.info("Runtime: leverage_mode → %s", mode)
        return True

    def get_leverage_target_risk(self) -> float:
        """Целевой риск на ATR-движение для режимов volatility/full."""
        val = self._get(keys.LEVERAGE_TARGET_RISK)
        try:
            return (
                max(0.001, min(float(val), 0.1))
                if val
                else float(getattr(Config, "LEVERAGE_TARGET_RISK", 0.01))
            )
        except (ValueError, TypeError):
            return 0.01

    def set_leverage_target_risk(self, risk: float) -> bool:
        if not (0.001 <= risk <= 0.1):
            return False
        self._set(keys.LEVERAGE_TARGET_RISK, str(risk))
        logger.info("Runtime: leverage_target_risk → %.3f", risk)
        return True

    def get_alert_level(self, name: str) -> str:
        return str(self._ALERT_DEFAULTS.get(name, {}).get("level", "info"))

    def get_quiet_hours(self) -> str:
        """Тихие часы «HH-HH» UTC: только критичные уведомления. Пусто — нет."""
        return self._get(keys.QUIET_HOURS) or ""

    def set_quiet_hours(self, hours: str) -> bool:
        if hours not in ("", "0") and _hour_in(hours, 0) is None:
            return False
        self._set(keys.QUIET_HOURS, "" if hours == "0" else hours)
        return True

    def should_notify(self, name: str, hour: Optional[int] = None) -> bool:
        """Отправлять ли уведомление этого вида сейчас: тумблер, потом тихие часы."""
        if not self.get_alert_enabled(name):
            return False
        if self.get_alert_level(name) == "critical":
            return True
        import datetime as dt

        hour = dt.datetime.utcnow().hour if hour is None else hour
        return not _hour_in(self.get_quiet_hours(), hour)

    def get_alert_enabled(self, name: str) -> bool:
        val = self._get(f"{keys.ALERT_ENABLED_PREFIX}{name}")
        return val != "0"  # включено по умолчанию

    def set_alert_enabled(self, name: str, enabled: bool) -> None:
        self._set(f"{keys.ALERT_ENABLED_PREFIX}{name}", "1" if enabled else "0")

    def get_alert_cooldown_mins(self, name: str) -> int:
        val = self._get(f"{keys.ALERT_COOLDOWN_PREFIX}{name}")
        if val is not None:
            try:
                return max(0, int(val))
            except ValueError:
                pass
        return self._ALERT_DEFAULTS.get(name, {}).get("cooldown", 5)

    def set_alert_cooldown_mins(self, name: str, minutes: int) -> None:
        self._set(f"{keys.ALERT_COOLDOWN_PREFIX}{name}", str(max(0, minutes)))
