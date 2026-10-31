// Типы ответов API и событий WebSocket — по docs/design/CONTRACT.md.

export type Side = "long" | "short";
export type Period = "24h" | "7d" | "30d" | "all";
export type Tab = "terminal" | "bots" | "learn" | "journal" | "settings";
/** Узел конвейера решений во вкладке «Бот». */
export type BotNode = "scan" | "strat" | "ai" | "sac" | "risk" | "order" | "track";

/** Шаг решения на узле (src/decision_trace.py): true — пропустил дальше, false — остановил, null — пропущен. */
export interface TraceStep {
  node: BotNode;
  ok: boolean | null;
  text: string;
}

/** Итог последнего цикла по узлам конвейера. */
export interface CycleTrace {
  at: number;
  mode: string;
  state: string;
  nodes: Partial<Record<BotNode, TraceStep>>;
}

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
  /** Шаги решения по узлам конвейера; нет у позиций, открытых до этапа 2. */
  trace?: TraceStep[];
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
  /** Сезон рынка и индекс Страха/Жадности; null, пока бот их не загрузил. */
  market?: { season: Season | null; fear_greed: FearGreed | null };
  /** Итог последнего цикла; null — бот ещё не прошёл цикл после обновления. */
  cycle?: CycleTrace | null;
  timestamp: string;
}

export interface Season {
  /** altcoin — альтсезон, bluechip — сезон BTC, null — нейтрально. */
  signal: "altcoin" | "bluechip" | null;
  altcoin_index: number | null;
  btc_dominance: number | null;
  btc_30d: number | null;
  checked_at: number;
}

export interface FearGreed {
  value: number;
  label: string;
}

/** Профиль рынка во вкладке «Обучение»: у каждого своя модель, таймфрейм и число монет. */
export interface ProfileInfo {
  id: string;
  label: string;
  timeframe: string;
  train_top_n: number;
  model_path: string;
  model_exists: boolean;
  trained_at: string | null;
  active: boolean;
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

export interface ExecutionMetrics {
  period: Period;
  /** Paper: цена и время исполнения симулированы. */
  simulated: boolean;
  n: number;
  min_n: number;
  /** null, пока n < min_n. */
  latency_ms_avg: number | null;
  /** %, плюс — хуже для сделки. */
  slippage_pct_avg: number | null;
  slippage_pct_worst: number | null;
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
  /** Тип логики: trending — тренд, ranging — возврат к средней, volatile — пробой и всплески. */
  type?: string;
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

/** Пресет риска из бота (GET /api/risk/presets) — те же значения, что у Telegram /risk. */
export interface RiskPreset {
  MAX_POSITIONS: number;
  RISK_PER_TRADE: number;
  DRAWDOWN_SCALE_ENABLED: boolean;
  LEVERAGE_MODE: "fixed" | "volatility" | "full";
}

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
  profiles: ProfileInfo[];
  /** Чью модель вернёт откат (последнее обучение). */
  backup_profile: string | null;
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
  /** critical — приходит всегда, даже в тихие часы. */
  level?: "critical" | "trade" | "info";
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
  /** Только у памп-каналов Telegram: вход по коду выполнен, сессия лежит в хранилище. */
  signed_in?: boolean;
  status: "ok" | "warn" | "bad" | "unset" | "unchecked";
  message: string;
  checked_at: number | null;
}

/** Шаг входа в Telegram для чтения каналов (`POST /api/secrets/telethon/*`). */
export type TgStep = { step: "code"; via: "app" | "sms" } | { step: "password" } | ({ step: "done" } & SecretItem);

/** Что бот делает с каналами Telegram (`GET /api/secrets/telethon/status`). */
export interface TgStatus {
  state: "ok" | "off" | "no_session" | "revoked" | "error" | "stopped" | "unknown";
  message: string;
  at?: number;
  channels?: string[];
  /** Каналы, которые не читаются, и почему. */
  failed?: Record<string, string>;
  window_min?: number;
  /** Постов в окне и сколько из них упоминают монеты текущего скана (0 — нет данных, а не «нейтрально»). */
  posts?: number;
  mentions?: Record<string, number>;
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
  /** Шаги решения; пусто у сделок до этапа 2. */
  trace?: TraceStep[];
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
