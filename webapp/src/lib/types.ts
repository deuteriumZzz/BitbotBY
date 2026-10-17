// Типы ответов API и событий WebSocket — по docs/design/CONTRACT.md.

export type Side = "long" | "short";
export type Period = "24h" | "7d" | "30d" | "all";
export type Tab = "terminal" | "bots" | "learn" | "journal" | "settings";

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
  /** Почему бот открыл позицию: стратегия, уверенность, объяснение. */
  strategy?: string;
  confidence?: number;
  reasoning?: string;
  /** Когда открыта (unix, с); нет у позиций, открытых до обновления. */
  opened_at?: number | null;
}

export interface BotState {
  bot: {
    alive: boolean;
    paused: boolean;
    paper: boolean;
    mode: string;
    profile: string | null;
    /** Провайдер AI, который ответил последним. */
    ai?: string | null;
  };
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
  /** Где применима: spot — только лонг, futures — нужны шорты, both — везде. */
  markets: "spot" | "futures" | "both";
  /** Добавлена без доказанного преимущества; выключена по умолчанию. */
  experimental?: boolean;
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
  /** Видно в «Все параметры» без «Показать расширенные». */
  basic: boolean;
}

export type ConfigValues = Record<string, { value: unknown; source: string }>;

/** Фоновая задача бота. Прогресс приходит событиями job.progress, итог — job.done. */
export type JobId = "backtest" | "train" | "tune" | "alpha_report" | "live_divergence" | "ai_isolation";

export interface JobDone {
  ok: boolean;
  error?: string;
  // бэктест
  n?: number;
  months?: number;
  timeframe?: string;
  best?: { strategy: string; sharpe: number; trades: number } | null;
  // обучение и тюнинг SAC
  sac_pct?: number | null;
  bh_pct?: number | null;
  worse?: boolean;
  // отчёты
  output?: string;
}

export interface JobState {
  running: boolean;
  /** Последнее событие прогресса: pct, eta_min, step/total, trial, symbol, strategy… */
  info: Record<string, unknown>;
  done?: JobDone;
}

/** `GET /api/jobs`: занятые группы (sac, backtest, …), прогресс и наличие резервной копии модели. */
export interface JobsStatus {
  running: Record<string, string>;
  progress: Partial<Record<"backtest" | "train" | "tune", Record<string, unknown> | null>>;
  rollback_available: boolean;
}

/** Запись журнала изменений настроек (`GET /api/config/history`). */
export interface ConfigHistoryItem {
  ts: number;
  by: string;
  changes: { key: string; label: string; old: unknown; new: unknown }[];
}

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
  /** Подсказка под названием поля: формат, где взять, предупреждения. */
  hint?: string;
  placeholder?: string;
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
  /** Только у X/Twitter: сохранённые аккаунты для обхода официального API (без паролей). */
  accounts?: { login: string; email: string }[];
  status: "ok" | "warn" | "bad" | "unset" | "unchecked";
  message: string;
  checked_at: number | null;
}

/** Что уже подключено (`GET /api/setup`): ключи из приложения или .env. */
export interface SetupInfo {
  /** Можно ли сохранять ключи из приложения (задан BITBOT_SECRETS_KEY). */
  keys_in_app: boolean;
  bybit: boolean;
  telegram: boolean;
}

export type SetupStep = "bybit" | "telegram" | "ai" | "risk" | "mode";

/** Закрытая сделка из журнала (`GET /api/journal`). */
export interface JournalTrade {
  id: number;
  symbol: string;
  side: Side;
  strategy: string;
  confidence: number;
  entry: number;
  exit: number;
  qty: number;
  /** Чистый результат — уже за вычетом комиссий. */
  pnl: number;
  /** Доля от маржи. */
  pnl_pct: number;
  fees: number;
  opened: string;
  closed: string;
  /** Код причины выхода: tp, sl, trail, manual_app, … */
  reason: string;
  reasoning: string;
}

export interface Journal {
  period: Period;
  limit: number;
  trades: JournalTrade[];
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
  reasoning?: string;
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

// ── AI-провайдеры (GET /api/ai) ───────────────────────────────────────────────

export type AiMode = "free" | "auto" | "paid";

export interface AiModel {
  id: string;
  free: boolean;
  strong: boolean;
  recommended: boolean;
}

export interface AiProvider {
  id: string;
  name: string;
  note: string;
  signup: string;
  free_tier: boolean;
  key_optional: boolean;
  url_label: string;
  key_set: boolean;
  url_ok: boolean;
  usable: boolean;
  keyless_on: boolean;
  disabled: boolean;
  model: string;
  model_source: "app" | "env" | "catalog";
  free: boolean;
  strong: boolean;
  models: AiModel[];
  verified: string;
  status: string;
  cooldown_until: number | null;
  calls_today: number;
  tokens_today: number;
}

export interface AiState {
  mode: AiMode;
  /** Один провайдер без запасных — иначе цепочка. */
  single: boolean;
  pinned: string;
  last: string;
  providers: AiProvider[];
  chain: { provider: string; model: string }[];
}

export interface AiChainChanges {
  mode?: AiMode;
  single?: boolean;
  order?: string[];
  disabled?: string[];
  keyless?: string[];
  models?: Record<string, string>;
}
