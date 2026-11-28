// Настройки, у которых в приложении есть свой понятный раздел. В общем списке
// «Параметры» они не дублируются: там остаётся ссылка на то место, где ими удобнее управлять.

import type { BotNode, Tab } from "./types";

export type SettingsPane = "main" | "acct" | "all";

export interface Home {
  tab: Tab;
  pane?: SettingsPane;
  /** Узел конвейера во вкладке «Бот». */
  node?: BotNode;
  /** Как это место называется для человека. */
  where: string;
}

const RISK: Home = { tab: "bots", node: "risk", where: "Стратегия → Риск → Как рискуем" };
const RISK_MANUAL: Home = { tab: "bots", node: "risk", where: "Стратегия → Риск → Как рискуем → Настроить вручную" };
const RULES: Home = { tab: "bots", node: "risk", where: "Стратегия → Риск → Правила защиты" };
const CONFIRM: Home = { tab: "bots", node: "order", where: "Стратегия → Ордер → Открытие сделок" };
const WHAT: Home = { tab: "bots", node: "scan", where: "Стратегия → Сканер → Что торговать" };
const FORECAST: Home = { tab: "bots", node: "ai", where: "Стратегия → Решение → значок прогноза" };
const SIGNALS: Home = { tab: "bots", node: "ai", where: "Стратегия → AI → Сигналы для решения" };

export const OWN_HOME: Record<string, Home> = {
  RISK_PER_TRADE: RISK,
  MAX_POSITIONS: RISK,
  DRAWDOWN_SCALE_ENABLED: RULES,
  LEVERAGE_MODE: RISK,
  AUTO_EXECUTE: CONFIRM,
  TELEGRAM_CONFIRM_TIMEOUT: CONFIRM,
  CONFIRM_ON_TIMEOUT: CONFIRM,
  MODE: { tab: "bots", where: "Стратегия → Кто принимает решение" },
  CHRONOS_ENABLED: FORECAST,
  FORECASTER: FORECAST,
  MARKET_TYPE: WHAT,
  SEASON_SWITCH_MODE: WHAT,
  PAPER_GUARDS_OFF: RULES,
  LOCAL_FALLBACK: { tab: "bots", node: "strat", where: "Стратегия → Стратегии" },
  TRADING_HOURS: { tab: "bots", node: "scan", where: "Стратегия → Сканер → Часы торговли" },
  TESTNET: { tab: "settings", pane: "acct", where: "Настройки → Счёт и бот → Bybit" },
  QUIET_HOURS: { tab: "bots", where: "Стратегия → Telegram → Бот и уведомления" },
  AI_PROVIDER: { tab: "bots", node: "ai", where: "Стратегия → AI" },
  MIN_SIGNAL_CONFIDENCE: RISK_MANUAL,
  MIN_SIGNAL_CONFIDENCE_PAPER: RISK_MANUAL,
  SENTIMENT_BUY_THRESHOLD: SIGNALS,
  SENTIMENT_SELL_THRESHOLD: SIGNALS,
  LEVERAGE: RISK_MANUAL,
  LEVERAGE_MIN: RISK_MANUAL,
  LEVERAGE_MAX: RISK_MANUAL,
  LEVERAGE_TARGET_RISK: RISK_MANUAL,
  LEVERAGE_DD_THRESHOLD: RISK_MANUAL,
  MAX_CORRELATION: RISK_MANUAL,
  MAX_CORRELATION_PAPER: RISK_MANUAL,
  CORRELATION_WINDOW: RISK_MANUAL,
  MAX_SPREAD_PCT: RISK_MANUAL,
  MAX_POSITION_SIZE: RISK_MANUAL,
  LIVE_ALPHA_KILL_ENABLED: RISK_MANUAL,
  LIVE_ALPHA_KILL_MIN_TRADES: RISK_MANUAL,
  LIVE_ALPHA_KILL_EV_THRESHOLD: RISK_MANUAL,
};
