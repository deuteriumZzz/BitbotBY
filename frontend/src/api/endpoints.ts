// Все адреса API бэкенда в одном месте — контракт с backend/bitbot/api/routers.
// Параметры в пути и запросе кодируются здесь, а не в каждом месте вызова.

const q = encodeURIComponent;

export const EP = {
  state: "/api/state",
  setup: "/api/setup",
  profile: "/api/profile",
  accountMode: "/api/account/mode",
  launcher: "/api/launcher",

  // вход
  authTelegram: "/api/auth/telegram",
  authLocal: "/api/auth/local",
  authDesktopRequest: "/api/auth/desktop/request",
  authDesktopVerify: "/api/auth/desktop/verify",
  authLogout: "/api/auth/logout",

  // бот и команды
  botPause: "/api/bot/pause",
  botResume: "/api/bot/resume",
  botRestart: "/api/bot/restart",
  botPanic: "/api/bot/panic",
  command: (requestId: string) => `/api/commands/${requestId}`,
  emergency: "/api/emergency",

  // торговля
  symbols: "/api/symbols",
  candles: (symbol: string) => `/api/candles?symbol=${q(symbol)}&tf=15m&limit=200`,
  position: (tradeId: number | string | null) => `/api/positions/${tradeId}`,
  positionClose: (tradeId: number | string | null) => `/api/positions/${tradeId}/close`,
  signalsPending: "/api/signals/pending",
  signal: (id: string, confirm: boolean) => `/api/signals/${id}/${confirm ? "confirm" : "reject"}`,
  strategies: "/api/strategies",
  strategy: (name: string) => `/api/strategies/${name}`,
  strategiesReset: "/api/strategies/reset",
  riskPresets: "/api/risk/presets",

  // результаты
  journal: (period: string) => `/api/journal?period=${period}`,
  pnlDaily: (days: number) => `/api/pnl/daily?days=${days}`,
  metrics: (period: string) => `/api/metrics?period=${period}`,
  metricsExecution: (period: string) => `/api/metrics/execution?period=${period}`,
  reportsDigest: (hours: number) => `/api/reports/digest?hours=${hours}`,
  reportsAudit: (id: string) => `/api/reports/audit/${q(id)}`,
  reportsDbcheck: "/api/reports/dbcheck",

  // задачи и модели
  jobs: "/api/jobs",
  job: (id: string) => `/api/jobs/${id}`,
  modelRollback: "/api/model/rollback",
  modelStatus: "/api/model/status",
  modelApprove: "/api/model/approve",
  metaStatus: "/api/meta/status",

  // AI
  ai: "/api/ai",
  aiChain: "/api/ai/chain",
  aiScoreboard: "/api/ai/scoreboard",

  // настройки и уведомления
  config: "/api/config",
  configSchema: "/api/config/schema",
  configHistory: "/api/config/history",
  configReset: "/api/config/reset",
  alerts: "/api/alerts",
  alert: (kind: string) => `/api/alerts/${kind}`,

  // ключи и подключения
  secrets: "/api/secrets",
  secret: (service: string) => `/api/secrets/${service}`,
  secretCheck: (service: string) => `/api/secrets/${service}/check`,
  telegramChat: "/api/secrets/telegram/chat",
  telethonStatus: "/api/secrets/telethon/status",
  telethon: (path: string) => `/api/secrets/telethon/${path}`,
  twitterAccounts: "/api/secrets/twitter/accounts",
  twitterAccount: (login: string) => `/api/secrets/twitter/accounts/${q(login)}`,
} as const;
