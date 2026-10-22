// Настройки, у которых в приложении есть свой понятный раздел. В общем списке
// «Параметры» они не дублируются: там остаётся ссылка на то место, где ими удобнее управлять.

import type { BotNode, Tab } from "./types";

export type SettingsPane = "main" | "ai" | "conn" | "all";

export interface Home {
  tab: Tab;
  pane?: SettingsPane;
  /** Узел конвейера во вкладке «Бот». */
  node?: BotNode;
  /** Как это место называется для человека. */
  where: string;
}

const RISK: Home = { tab: "bots", node: "risk", where: "Бот → Риск → Режим риска" };
const CONFIRM: Home = { tab: "bots", node: "order", where: "Бот → Ордер → Открытие сделок" };
const WHAT: Home = { tab: "bots", node: "scan", where: "Бот → Сканер → Что торговать" };

export const OWN_HOME: Record<string, Home> = {
  RISK_PER_TRADE: RISK,
  MAX_POSITIONS: RISK,
  DRAWDOWN_SCALE_ENABLED: RISK,
  LEVERAGE_MODE: RISK,
  AUTO_EXECUTE: CONFIRM,
  TELEGRAM_CONFIRM_TIMEOUT: CONFIRM,
  CONFIRM_ON_TIMEOUT: CONFIRM,
  MODE: { tab: "bots", where: "Бот → Кто принимает решение" },
  CHRONOS_ENABLED: WHAT,
  MARKET_TYPE: WHAT,
  SEASON_SWITCH_MODE: WHAT,
  PAPER_GUARDS_OFF: { tab: "bots", node: "risk", where: "Бот → Риск → Защиты портфеля" },
  LOCAL_FALLBACK: { tab: "bots", node: "strat", where: "Бот → Стратегии" },
  TRADING_HOURS: { tab: "settings", pane: "main", where: "Настройки → Основное → Часы торговли" },
  TESTNET: { tab: "settings", pane: "conn", where: "Настройки → Подключения → Bybit" },
  AI_PROVIDER: { tab: "settings", pane: "ai", where: "Настройки → AI" },
};
