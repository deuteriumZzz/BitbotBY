// Типы ответов API и событий WebSocket — по docs/design/CONTRACT.md.

export type Side = "long" | "short";
export type Period = "24h" | "7d" | "30d" | "all";
export type Tab = "terminal" | "bots" | "settings";

export interface Position {
  id: string;
  trade_id: number | string | null;
  symbol: string;
  side: Side;
  entry: number;
  qty: number;
  sl: number;
  tp: number;
  pnl_pct?: number | null;
  partial_tp_done?: boolean;
  trailing?: boolean;
  price?: number;
  /** Из сигнала радара, если приложение видело его до открытия. */
  strategy?: string;
  confidence?: number;
}

export interface BotState {
  bot: { alive: boolean; paused: boolean; paper: boolean; mode: string; profile: string | null };
  balance: number;
  slots: { used: number; max: number };
  positions: Position[];
  timestamp: string;
}

export interface Metrics {
  period: Period;
  n: number;
  net: number;
  fees: number;
  realized: number;
  win_rate: number | null;
  profit_factor: number | null;
  avg_win: number | null;
  avg_loss: number | null;
  rr: number | null;
  max_drawdown_pct: number;
  sharpe: number | null;
}

export interface DailyRow {
  date: string;
  net: number;
  fees: number;
  trades: number;
}

export interface Strategy {
  name: string;
  enabled: boolean;
  backtest: { win_rate: number; trades: number; sharpe: number } | null;
  live: { win_rate: number; trades: number; ev: number } | null;
}

export interface SettingDef {
  key: string;
  group: string;
  label: string;
  type: "int" | "float" | "pct" | "bool" | "choice" | "str";
  default: unknown;
  min: number | null;
  max: number | null;
  step: number | null;
  unit: string;
  options: { value: string; label: string }[];
  hint: string;
  readonly: boolean;
  hot: boolean;
}

export type ConfigValues = Record<string, { value: unknown; source: string }>;

export interface Alert {
  kind: string;
  label: string;
  enabled: boolean;
  cooldown_min: number;
}

export interface Symbols {
  forced: string[];
  excluded: string[];
}

export interface SecretField {
  env: string;
  label: string;
  secret: boolean;
  optional: boolean;
  set: boolean;
  last4: string | null;
  value: string | null;
}

export interface SecretItem {
  service: string;
  name: string;
  group: string;
  fields: SecretField[];
  model: { env: string; value: string | null } | null;
  status: "ok" | "warn" | "bad" | "unset" | "unchecked";
  message: string;
  checked_at: number | null;
}

export interface PendingSignal {
  id: string;
  symbol: string;
  side: Side;
  confidence: number;
  strategy: string;
  entry: number;
  stop_loss: number;
  take_profit: number;
  expires_at: number;
  on_timeout_execute: boolean;
  backtest: { win_rate: number; trades: number; ev: number };
  live: { win_rate: number; trades: number; ev: number };
}

export interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface Guard {
  kind: string;
  active: boolean;
  detail?: string;
  until?: number;
}

export interface BotEvent {
  type: string;
  ts: number;
  data: Record<string, unknown>;
}

export interface CommandResult {
  ok: boolean;
  error?: { code: string; message: string };
  [k: string]: unknown;
}
