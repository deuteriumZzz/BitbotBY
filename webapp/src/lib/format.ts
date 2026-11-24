import type { OpsStatus } from "./types";

// Числа в стиле терминала: узкий неразрывный пробел между разрядами, точка в дробной части.

export const fmtN = (n: number, d = 2): string =>
  Math.abs(n)
    .toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d })
    .replace(/,/g, " ");

export const sign = (n: number, withPlus = true): string =>
  n < 0 ? "−" : withPlus && n > 0 ? "+" : "";

export const usd = (n: number, withPlus = true): string => `${sign(n, withPlus)}$${fmtN(n)}`;

/** Знаков после точки по величине цены: 64210.00, 182.40, 3.842, 0.00001020. */
export const priceDecimals = (p: number): number =>
  p > 0 ? Math.min(10, Math.max(2, 3 - Math.floor(Math.log10(p)))) : 2;

export const price = (p: number): string => fmtN(p, priceDecimals(p));

export const pct = (n: number, d = 2): string => `${sign(n)}${fmtN(n, d)}%`;

export const arrow = (n: number): string => (n > 0 ? "▲" : n < 0 ? "▼" : "•");

export const tone = (n: number): "up" | "down" | "" => (n > 0 ? "up" : n < 0 ? "down" : "");

/** Русское множественное число: plural(3, ["сделка", "сделки", "сделок"]). */
export function plural(n: number, forms: [string, string, string]): string {
  const a = Math.abs(n) % 100;
  const b = a % 10;
  if (a > 10 && a < 20) return forms[2];
  if (b > 1 && b < 5) return forms[1];
  if (b === 1) return forms[0];
  return forms[2];
}

export const hhmm = (d: Date = new Date()): string => d.toTimeString().slice(0, 5);

export const base = (symbol: string): string => symbol.split("/")[0];

// Причины выхода из сделки (коды бота → по-русски). Динамический выход приходит
// текстом вида «signal_reversal SELL conf=0.72» — узнаём по первому слову.
const EXIT_REASON: Record<string, string> = {
  tp: "по тейк-профиту",
  sl: "по стоп-лоссу",
  trail: "по трейлинг-стопу",
  manual_app: "вручную",
  telegram_closeall: "командой «закрыть всё»",
  emergency: "аварийно",
  hard_drawdown: "защита от просадки",
  atr_spike: "всплеск волатильности",
  dynamic_exit: "по сигналу на выход",
  signal_reversal: "сигнал развернулся",
  regime_flip: "рынок развернулся против позиции",
  funding: "перегрев по фандингу",
  fear_greed: "крайний страх или жадность на рынке",
  long_liquidation: "каскад ликвидаций",
  short_squeeze: "каскад ликвидаций",
};

/** Почему режим не принимает решений (коды из src/core/trading_modes.py). */
const READINESS_REASON: Record<string, string> = {
  sac_model_missing: "модель SAC не загружена — пока её нет, новых сделок в этом режиме не будет",
  sac_not_approved:
    "модель SAC загружена, но не допущена реестром (в Paper нужна стадия paper и выше, в Live — только production)",
  ai_unavailable: "AI недоступен — ни один провайдер цепочки не отвечает",
  unknown_mode: "неизвестный режим",
};

/** Подписи функций защиты и наблюдения: [ключ флага, название, что делает]. */
const OPS_FLAG_INFO: Array<[string, string, string]> = [
  ["WS_SHADOW_ENABLED", "WS-наблюдение за рынком", "Публичный поток Bybit параллельно REST: только метрики и тревоги, решений не принимает."],
  ["AI_SANITIZE_PROMPT", "Очистка новостей для AI", "Заголовки новостей очищаются от управляющих символов и разметки ролей и помечаются как недоверенный текст."],
  ["AI_STRICT_VALIDATION", "Строгая проверка ответов AI", "Ответ с лишними полями, выдуманными ценами или SL/TP не с той стороны отбрасывается."],
  ["AI_RAG_CONTEXT", "Память решений в промпте", "В запрос к AI добавляется, чем заканчивались похожие решения бота."],
  ["RISK_GATE_SHADOW", "Единый Risk Gate (тень)", "Вердикт общего риск-шлюза по каждому входу пишется в журнал, но сделки не блокирует."],
  ["DB_BACKUP_ENABLED", "Ежедневная копия базы сделок", "Проверенная копия базы раз в сутки, хранятся последние 7."],
  ["FUNDING_IN_PNL", "Funding в результате сделки", "Выплаты funding входят в итог сделки."],
  ["SAC_V2_ENABLED", "SAC v2", "Новая версия нейросети SAC; торгует только после допуска в реестре."],
  ["EVENT_STREAM_ENABLED", "Шина событий", "Копии записей журнала как отметки в Redis Stream."],
  ["DATA_QUALITY_GATE", "Блокировка входа при плохих данных", "Нет входа по снимкам со статусом STALE/INVALID."],
];

export interface OpsLine {
  key: string;
  label: string;
  hint: string;
  on: boolean;
}

/** Готовые к показу строки карточки «Защита и наблюдение». Неизвестные боту флаги пропускаются. */
export function opsSummary(ops: OpsStatus): {
  flags: OpsLine[];
  ws: string | null;
  wsAlarm: boolean;
  backup: string | null;
  backupAlarm: boolean;
  emergency: number;
} {
  const flags = OPS_FLAG_INFO.filter(([k]) => k in ops.flags).map(([key, label, hint]) => ({ key, label, hint, on: !!ops.flags[key] }));
  const w = ops.ws;
  const ws = w
    ? w.silent.length
      ? `молчат ${w.silent.length} из ${w.streams} потоков`
      : `${w.streams} потоков в норме, p99 задержки ${w.lag_p99_ms === null ? "—" : `${Math.round(w.lag_p99_ms)} мс`}, переподключений ${w.reconnects}, пропусков ${w.gaps}`
    : null;
  const b = ops.backup;
  const backup = b ? (b.ok ? `последняя копия ${b.at.slice(0, 16).replace("T", " ")} UTC` : `копия не создана: ${b.error ?? "ошибка"}`) : null;
  return { flags, ws, wsAlarm: !!w && w.silent.length > 0, backup, backupAlarm: !!b && !b.ok, emergency: ops.emergency };
}

export const readinessReason = (code?: string | null): string =>
  code ? (READINESS_REASON[code] ?? code) : "";

export const exitReason = (code?: string | null): string =>
  code ? (EXIT_REASON[code] ?? EXIT_REASON[code.split(/[\s=]/)[0]] ?? code) : "";
