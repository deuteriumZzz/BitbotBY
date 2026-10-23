// Мок-бот для разработки и демо (?mock=1): те же ответы REST и события WS,
// что у настоящего сервера, но рынок и сделки выдуманы.

import { ApiError } from "../lib/api";
import { plural } from "../lib/format";
import type {
  AiState,
  Alert,
  BotEvent,
  BotNode,
  Candle,
  CycleTrace,
  ConfigValues,
  DailyRow,
  JournalTrade,
  ExecutionMetrics,
  Metrics,
  PendingSignal,
  Position,
  SecretItem,
  SettingDef,
  Side,
  Strategy,
  TraceStep,
} from "../lib/types";
import fixtures from "./fixtures.json";

// Тело запроса приходит нетипизированным — как JSON на сервере.
type Body = Record<string, any>;

// ── Детерминированный сид: одинаковая история при каждом запуске ──────────────
function mulberry32(a: number): () => number {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const seeded = mulberry32(42);

const PRICES: Record<string, number> = {
  "BTC/USDT": 64210,
  "ETH/USDT": 3120,
  "SOL/USDT": 182.4,
  "XRP/USDT": 0.6123,
  "DOGE/USDT": 0.1542,
  "LINK/USDT": 14.82,
  "AAVE/USDT": 142.3,
  "SUI/USDT": 1.842,
  "NEAR/USDT": 5.412,
  "HYPE/USDT": 28.4,
  "ENA/USDT": 0.4821,
  "UNI/USDT": 9.62,
  "WLD/USDT": 2.214,
  "ONDO/USDT": 1.124,
  "MNT/USDT": 0.8812,
  "ARB/USDT": 0.7421,
};
const SYMBOLS = Object.keys(PRICES);
// Объяснения в том виде, как их пишет бот (AI + SAC в signal_combiner).
const WHY = [
  "RSI перепродан (28), цена у нижней полосы Боллинджера. [SAC 71%, regime=sideways]",
  "EMA 9 пересекла EMA 21 снизу, объём выше среднего в 1.8 раза. [SAC 64%, regime=trending_up]",
  "Пробой локального максимума на растущем объёме, фандинг нейтральный.",
  "AI+SAC agree, SAC 77%, regime=trending_down",
];

const REJECTS = [
  "сильная корреляция с открытой позицией",
  "мало ликвидности или широкий спред",
  "достигнут лимит нагрузки портфеля",
  "стратегия отключена kill-switch",
];

// ── Состояние ─────────────────────────────────────────────────────────────────
const schema = fixtures.schema as SettingDef[];
const config = structuredClone(fixtures.config) as ConfigValues;
const strategies = structuredClone(fixtures.strategies) as Strategy[];
const secrets = structuredClone(fixtures.secrets) as SecretItem[];
for (const s of secrets) {
  if (!["bybit", "groq", "gemini", "telegram"].includes(s.service)) continue;
  Object.assign(s, { status: "ok", message: "Ключ проверен.", checked_at: Date.now() / 1000 - 3600 });
  for (const f of s.fields)
    if (!f.optional) Object.assign(f, { set: true, last4: f.secret ? "a9F2" : null, value: f.secret ? null : "1200450" });
}
// ── AI: цепочка провайдеров (как GET /api/ai) ─────────────────────────────────
const ai = structuredClone(fixtures.ai) as AiState;

/** Те же правила, что ai_providers.chain на сервере. */
function recomputeAi(): void {
  const ok = (p: AiState["providers"][number]) =>
    p.id === ai.pinned || (ai.mode === "free" ? p.free : ai.mode === "paid" ? !p.free : true);
  let chain = ai.providers.filter((p) => p.usable && (!p.disabled || p.id === ai.pinned) && ok(p));
  if (ai.mode === "auto") chain = [...chain.filter((p) => p.free), ...chain.filter((p) => !p.free)];
  chain = [...chain.filter((p) => p.id === ai.pinned), ...chain.filter((p) => p.id !== ai.pinned)];
  if (ai.single) chain = ai.pinned !== "auto" ? chain.filter((p) => p.id === ai.pinned) : chain.slice(0, 1);
  ai.chain = chain.map((p) => ({ provider: p.id, model: p.model }));
}

function applyAi(b: Body): AiState {
  if (b.mode !== undefined && !["free", "auto", "paid"].includes(b.mode)) fail(400, "bad_request", "Режим — free, auto или paid.");
  if (b.mode) ai.mode = b.mode;
  if (typeof b.single === "boolean") ai.single = b.single;
  if (Array.isArray(b.order)) {
    const rank = (id: string) => (b.order.includes(id) ? b.order.indexOf(id) : 1e3);
    ai.providers.sort((x, y) => rank(x.id) - rank(y.id));
  }
  if (Array.isArray(b.disabled)) ai.providers.forEach((p) => (p.disabled = b.disabled.includes(p.id)));
  if (Array.isArray(b.keyless))
    ai.providers.forEach((p) => {
      if (!p.key_optional) return;
      p.keyless_on = b.keyless.includes(p.id);
      p.usable = p.key_set || p.keyless_on;
    });
  if (b.models && typeof b.models === "object")
    ai.providers.forEach((p) => {
      const m = b.models[p.id];
      if (!m) return;
      const known = p.models.find((x) => x.id === m);
      Object.assign(p, { model: m, model_source: "app", free: known ? known.free : p.free_tier, strong: known ? known.strong : true });
    });
  recomputeAi();
  emit("config.changed", { keys: ["AI_CHAIN"] });
  return ai;
}

/** Цикл зовёт AI: иногда первый провайдер упирается в лимит и уходит на паузу. */
// Кто из AI ответил в этом цикле и кто до него не смог — для следа решения.
let aiAnswer = { who: "", failed: [] as string[] };

function simulateAi(): void {
  aiAnswer = { who: "", failed: [] };
  const live = ai.chain.filter(({ provider }) => {
    const p = ai.providers.find((x) => x.id === provider)!;
    return !(p.cooldown_until && p.cooldown_until * 1000 > Date.now());
  });
  for (const [i, { provider, model }] of live.entries()) {
    const p = ai.providers.find((x) => x.id === provider)!;
    p.calls_today += 1;
    if (i < live.length - 1 && Math.random() < 0.2) {
      Object.assign(p, { status: "rate_limit", cooldown_until: Date.now() / 1000 + 420 });
      emit("ai.status", { provider, model, status: "rate_limit", until: p.cooldown_until });
      aiAnswer.failed.push(`${p.name} — лимит`);
      continue;
    }
    Object.assign(p, { status: "ok", cooldown_until: null, tokens_today: p.tokens_today + 3200 });
    ai.last = provider;
    aiAnswer.who = p.name;
    emit("ai.status", { provider, model, status: "ok", until: null });
    return;
  }
}

const alerts: Alert[] = [
  ["atr_spike", "Всплеск волатильности"],
  ["drawdown", "Просадка"],
  ["daily_limit", "Дневной лимит убытка"],
  ["bot_stalled", "Бот перестал работать"],
  ["new_signal", "Новые сигналы"],
  ["position_open", "Открытие позиции"],
  ["position_close", "Закрытие позиции"],
].map(([kind, label]) => ({
  kind,
  label,
  enabled: kind !== "new_signal",
  cooldown_min: kind === "atr_spike" ? 30 : 0,
  level: kind === "new_signal" ? "info" : kind.startsWith("position") ? "trade" : "critical",
}));
const lists = { forced: ["SOL/USDT"], excluded: ["PUMP/USDT"] };

const bot = { paused: false, paper: true, mode: "ai", profile: "altcoin" as string | null };
let balance = 10_000;
let tradeSeq = 1000;
type Live = Position & { price: number };
const positions: Record<string, Live> = {};
const pending: Record<string, PendingSignal> = {};
const results: Record<string, unknown> = {};

interface Closed {
  exit: number; // мс
  net: number;
  fees: number;
  j: JournalTrade;
}
const closed: Closed[] = [];

// ── След решения: то же, что src/decision_trace.py, на данных симуляции ──────
let lastCycle: CycleTrace | null = null;
const step = (node: BotNode, ok: boolean | null, text: string): TraceStep => ({ node, ok, text });
const modeNow = (): string => String(config.MODE?.value ?? bot.mode);
const coins = (syms: string[], sides: Record<string, Side>): string =>
  syms.map((s) => `${s.split("/")[0]} ${sides[s] === "long" ? "▲" : "▼"}`).join(", ");
const nSignals = (n: number): string => `${n} ${plural(n, ["сигнал", "сигнала", "сигналов"])}`;
const nHeld = (n: number): string => `сопровождает ${n} ${plural(n, ["позицию", "позиции", "позиций"])}`;

function saveCycle(state: string, nodes: CycleTrace["nodes"]): void {
  lastCycle = { at: Date.now() / 1000, mode: modeNow(), state, nodes };
  emit("trace.cycle", lastCycle as unknown as Record<string, unknown>);
}

function cycleNodes(
  scanN: number,
  pre: string[],
  sides: Record<string, Side>,
  rejected: Record<string, string>,
  opened: string[],
  used: number,
): CycleTrace["nodes"] {
  const mode = modeNow();
  const aiOn = mode === "ai" || mode === "hybrid";
  const sigs = Object.keys(sides);
  const rej = Object.values(rejected);
  const top = [...rej].sort((a, b) => rej.filter((x) => x === b).length - rej.filter((x) => x === a).length)[0] ?? "";
  const held = used + opened.length;
  const failed = aiAnswer.failed.length ? ` после сбоя: ${aiAnswer.failed.join(", ")}` : "";
  return {
    scan: step("scan", true, `${scanN} ${plural(scanN, ["монета", "монеты", "монет"])} · ${pre.length} в кандидаты`),
    strat:
      mode === "local"
        ? sigs.length
          ? step("strat", true, `сигналы: ${coins(sigs, sides)}`)
          : step("strat", null, "сигналов нет")
        : step("strat", null, aiOn ? "подсказка для AI" : "только запасной путь"),
    ai: !aiOn
      ? step("ai", null, "не участвует")
      : aiAnswer.who
        ? step("ai", true, `ответил ${aiAnswer.who}${failed} · ${nSignals(sigs.length)}`)
        : step("ai", false, "не ответил никто"),
    sac: mode === "dqn" || mode === "hybrid" ? step("sac", null, "сигналов нет") : step("sac", null, "не участвует"),
    risk: !sigs.length
      ? step("risk", null, "нечего проверять")
      : opened.length
        ? step("risk", true, `прошли ${opened.length}${rej.length ? ` · отсеяно ${rej.length}: ${top}` : ""}`)
        : step("risk", false, `отсеяно ${rej.length}: ${top}`),
    order: opened.length ? step("order", true, `открыта: ${coins(opened, sides)}`) : step("order", null, "сделок нет"),
    track: held ? step("track", true, nHeld(held)) : step("track", null, "позиций нет"),
  };
}

function tradeSteps(side: Side, why: Partial<Position>, scanN: number, inPre: boolean, entry: number): TraceStep[] {
  const mode = modeNow();
  const aiOn = mode === "ai" || mode === "hybrid";
  const word = side === "long" ? "LONG" : "SHORT";
  const conf = why.confidence !== undefined ? ` · ${why.confidence.toFixed(2)}` : "";
  const failed = aiAnswer.failed.length ? ` (после сбоя: ${aiAnswer.failed.join(", ")})` : "";
  return [
    step("scan", true, `в выборке из ${scanN} монет${inPre ? " · прошла предфильтр" : ""}`),
    aiOn
      ? step("strat", null, `подсказка: AI отнёс сигнал к «${why.strategy ?? "—"}»`)
      : step("strat", true, `стратегия «${why.strategy ?? "—"}»`),
    aiOn ? step("ai", true, `${aiAnswer.who || "AI"}: ${word}${conf}${failed}`) : step("ai", null, "не участвует"),
    step("sac", null, mode === "dqn" || mode === "hybrid" ? "сигнала не дал" : "не участвует"),
    step("risk", true, "проверки пройдены"),
    step("order", true, `открыта ${word} по ${+entry.toPrecision(6)}`),
  ];
}

function journalRow(p: Position, exitPrice: number, qty: number, net: number, fees: number, exit: number, opened: number, reason: string): Closed {
  const j: JournalTrade = {
    id: Number(p.trade_id),
    symbol: p.symbol,
    side: p.side,
    strategy: p.strategy ?? "ema_crossover",
    confidence: p.confidence ?? 0.6,
    entry: p.entry,
    exit: exitPrice,
    qty,
    pnl: net,
    pnl_pct: net / (p.entry * qty),
    fees,
    opened: new Date(opened).toISOString(),
    closed: new Date(exit).toISOString(),
    reason,
    reasoning: p.reasoning ?? "",
    trace: p.trace,
  };
  return { exit, net, fees, j };
}

// 35 дней истории: 1–5 сделок в день, слабый плюс — как у честного paper-бота.
{
  const now = Date.now();
  const at = <T,>(a: readonly T[]): T => a[Math.floor(seeded() * a.length)];
  for (let day = 35; day >= 1; day--) {
    const n = 1 + Math.floor(seeded() * 5);
    for (let i = 0; i < n; i++) {
      const size = 20 + seeded() * 120;
      const win = seeded() < 0.47;
      const net = win ? size * 1.6 : -size;
      const fees = 1 + seeded() * 3;
      const symbol = at(SYMBOLS);
      const side: Side = seeded() < 0.6 ? "long" : "short";
      const entry = PRICES[symbol] * (0.95 + seeded() * 0.1);
      const qty = (400 + seeded() * 1600) / entry;
      const exitPrice = entry + ((side === "long" ? 1 : -1) * (net + fees)) / qty;
      const exit = now - day * 86_400_000 + i * 3_600_000;
      const p = { trade_id: -closed.length - 1, symbol, side, entry, strategy: at(strategies.slice(0, 6)).name, reasoning: at(WHY) };
      const reason = win ? at(["tp", "tp", "trail"]) : at(["sl", "sl", "signal_reversal SELL conf=0.71", "hard_drawdown"]);
      closed.push(journalRow(p as Position, exitPrice, qty, net, fees, exit, exit - (20 + seeded() * 600) * 60_000, reason));
      balance += net;
    }
  }
}

// ── Качество исполнения — та же логика, что compute_execution на сервере ─────
const EXEC_MIN_TRADES = 20;

function execution(period: string): ExecutionMetrics {
  const since = PERIOD_MS[period] ? Date.now() - PERIOD_MS[period] : 0;
  // Детерминированно от номера сделки: слегка в минус для сделки, как у paper с импактом.
  const rows = closed
    .map((c, k) => ({ c, lat: 120 + ((k * 53) % 260), slip: 0.004 + ((k * 29) % 40) / 2000 - 0.004 }))
    .filter((r) => r.c.exit >= since);
  const n = rows.length;
  const ok = n >= EXEC_MIN_TRADES;
  return {
    period: period as ExecutionMetrics["period"],
    simulated: true,
    n,
    min_n: EXEC_MIN_TRADES,
    latency_ms_avg: ok ? +(rows.reduce((s, r) => s + r.lat, 0) / n).toFixed(1) : null,
    slippage_pct_avg: ok ? +(rows.reduce((s, r) => s + r.slip, 0) / n).toFixed(4) : null,
    slippage_pct_worst: ok ? +Math.max(...rows.map((r) => r.slip)).toFixed(4) : null,
  };
}

const publicPos = (p: Position) => ({
  symbol: p.symbol,
  trade_id: p.trade_id,
  side: p.side,
  entry: p.entry,
  qty: p.qty,
  sl: p.sl,
  tp: p.tp,
  trailing: p.trailing,
  opened_at: p.opened_at,
});

function open(symbol: string, side: Side, why: Partial<Position> = {}): Live {
  const price = PRICES[symbol];
  const dist = price * (0.012 + Math.random() * 0.01);
  const dir = side === "long" ? 1 : -1;
  const id = ++tradeSeq;
  const p: Live = {
    id: String(id),
    trade_id: id,
    symbol,
    side,
    entry: price,
    qty: +((balance * 0.02) / dist).toPrecision(4),
    sl: price - dir * dist,
    tp: price + dir * dist * 2,
    price,
    trailing: true,
    opened_at: Date.now() / 1000,
    strategy: "ema_crossover",
    confidence: 0.68,
    reasoning: WHY[id % WHY.length],
    ...why,
  };
  positions[symbol] = p;
  return p;
}

// Демо-позиции, чтобы терминал не был пустым при первом открытии.
for (const [sym, side, move, hours] of [["SOL/USDT", "long", 1.006, 7], ["ETH/USDT", "short", 0.997, 3]] as const) {
  const p = open(sym, side);
  p.trace = tradeSteps(side, p, SYMBOLS.length, true, p.entry);
  p.price = PRICES[sym] = p.entry * move;
  p.opened_at = Date.now() / 1000 - hours * 3600;
}

function closePos(symbol: string, fraction = 1, reason = "manual_app"): void {
  const p = positions[symbol];
  if (!p) return;
  const dir = p.side === "long" ? 1 : -1;
  const qty = fraction >= 1 ? p.qty : p.qty * fraction;
  const fees = p.price * qty * 0.0011;
  const net = (p.price - p.entry) * qty * dir - fees;
  balance += net;
  closed.push(journalRow(p, p.price, qty, net, fees, Date.now(), (p.opened_at ?? Date.now() / 1000) * 1000, reason));
  if (fraction >= 1) {
    delete positions[symbol];
    emit("position.closed", { symbol, reason, price: p.price });
  } else {
    p.qty = +(p.qty - qty).toPrecision(4);
    emit("position.updated", publicPos(p));
  }
}

const maxPositions = (): number => Number(config.MAX_POSITIONS?.value ?? 3);

function state() {
  return {
    bot: { alive: true, ...bot, ai: ai.last || null },
    balance: +balance.toFixed(2),
    slots: { used: Object.keys(positions).length, max: maxPositions() },
    market: {
      season: { signal: null, altcoin_index: 41, btc_dominance: 58.3, btc_30d: 4.2, checked_at: Date.now() / 1000 },
      fear_greed: { value: 38, label: "Fear" },
    },
    positions: Object.values(positions).map((p) => ({
      ...publicPos(p),
      id: p.id,
      pnl_pct: null,
      partial_tp_done: false,
      strategy: p.strategy,
      confidence: p.confidence,
      reasoning: p.reasoning,
      trace: p.trace,
    })),
    cycle: lastCycle,
    timestamp: new Date().toISOString(),
  };
}

// ── Метрики — та же логика, что compute_metrics на сервере ───────────────────
const DAY = 86_400_000;
const PERIOD_MS: Record<string, number> = { "24h": DAY, "7d": 7 * DAY, "30d": 30 * DAY };

function metrics(period: string): Metrics {
  const since = PERIOD_MS[period] ? Date.now() - PERIOD_MS[period] : 0;
  const t = closed.filter((c) => c.exit >= since);
  const sum = (a: Closed[]) => a.reduce((s, c) => s + c.net, 0);
  const wins = t.filter((c) => c.net > 0);
  const losses = t.filter((c) => c.net <= 0);
  const net = sum(t);
  const fees = t.reduce((s, c) => s + c.fees, 0);
  const avgWin = wins.length ? sum(wins) / wins.length : null;
  const avgLoss = losses.length ? sum(losses) / losses.length : null;
  let eq = 0;
  let peak = 0;
  let dd = 0;
  for (const c of t) {
    eq += c.net;
    peak = Math.max(peak, eq);
    dd = Math.max(dd, (peak - eq) / (10_000 + peak));
  }
  const rets = t.map((c) => c.net / 10_000);
  const mean = rets.reduce((a, b) => a + b, 0) / (rets.length || 1);
  const sd = Math.sqrt(rets.reduce((a, b) => a + (b - mean) ** 2, 0) / (rets.length || 1));
  return {
    period: period as Metrics["period"],
    n: t.length,
    net: +net.toFixed(2),
    fees: +fees.toFixed(2),
    realized: +(net + fees).toFixed(2),
    win_rate: t.length ? wins.length / t.length : null,
    profit_factor: losses.length ? sum(wins) / -sum(losses) : null,
    avg_win: avgWin,
    avg_loss: avgLoss,
    rr: avgWin !== null && avgLoss ? avgWin / -avgLoss : null,
    max_drawdown_pct: +(dd * 100).toFixed(2),
    sharpe: t.length >= 50 && sd > 0 ? +((mean / sd) * Math.sqrt(365)).toFixed(2) : null,
  };
}

function daily(days: number): DailyRow[] {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  return Array.from({ length: days }, (_, k) => {
    const start = today.getTime() - (days - 1 - k) * DAY;
    const t = closed.filter((c) => c.exit >= start && c.exit < start + DAY);
    return {
      date: new Date(start).toISOString().slice(0, 10),
      net: +t.reduce((s, c) => s + c.net, 0).toFixed(2),
      fees: +t.reduce((s, c) => s + c.fees, 0).toFixed(2),
      trades: t.length,
    };
  });
}

function candles(symbol: string, limit: number): Candle[] {
  const step = 900;
  const now = Math.floor(Date.now() / 1000 / step) * step;
  let c = PRICES[symbol] ?? 100;
  const out: Candle[] = [];
  // Идём назад от текущей цены, чтобы последняя свеча совпала с тиками.
  for (let i = 0; i < limit; i++) {
    const o = c * (1 + (Math.random() - 0.5) * 0.008);
    const high = Math.max(o, c) * (1 + Math.random() * 0.003);
    const low = Math.min(o, c) * (1 - Math.random() * 0.003);
    out.push({ time: now - i * step, open: o, high, low, close: c });
    c = o;
  }
  return out.reverse();
}

// ── События ───────────────────────────────────────────────────────────────────
const listeners = new Set<(e: BotEvent) => void>();
function emit(type: string, data: Record<string, unknown>): void {
  const e = { type, ts: Date.now() / 1000, data };
  listeners.forEach((l) => l(e));
}

let timers: ReturnType<typeof setInterval>[] = [];
let cycleNo = 0;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const pick = <T,>(a: T[]): T => a[Math.floor(Math.random() * a.length)];

function tick(): void {
  for (const s of SYMBOLS) PRICES[s] *= 1 + (Math.random() - 0.495) * 0.0016;
  for (const p of Object.values(positions)) {
    p.price = PRICES[p.symbol];
    const dir = p.side === "long" ? 1 : -1;
    // Трейлинг: SL тянется за ценой на исходной дистанции, только в сторону прибыли.
    const dist = Math.abs(p.entry - p.sl);
    const next = p.price - dir * dist;
    if (p.trailing && (p.price - p.entry) * dir > dist && (next - p.sl) * dir > 0) p.sl = next;
    emit("tick", { ...publicPos(p), price: p.price });
    if ((p.price - p.tp) * dir >= 0) closePos(p.symbol, 1, "tp");
    else if ((p.price - p.sl) * dir <= 0) closePos(p.symbol, 1, (p.sl - p.entry) * dir > 0 ? "trail" : "sl");
  }
}

async function cycle(): Promise<void> {
  if (bot.paused) {
    emit("cycle", { state: "paused" });
    const nodes: CycleTrace["nodes"] = {};
    for (const n of ["scan", "strat", "ai", "sac", "risk", "order"] as BotNode[]) nodes[n] = step(n, null, "пауза — новые входы остановлены");
    nodes.track = step("track", null, "пауза: открытые позиции сопровождаются");
    saveCycle("paused", nodes);
    return;
  }
  cycleNo++;
  const used = Object.keys(positions).length;
  const max = maxPositions();
  if (used >= max) {
    emit("cycle", { state: "monitor", used, max });
    const skip = "пропущено: все слоты заняты";
    saveCycle("monitor", {
      scan: step("scan", null, skip),
      strat: step("strat", null, skip),
      ai: step("ai", null, skip),
      sac: step("sac", null, skip),
      risk: step("risk", false, `все слоты заняты (${used}/${max})`),
      order: step("order", null, "сделок нет"),
      track: step("track", true, nHeld(used)),
    });
    return;
  }
  const scan = SYMBOLS.filter((s) => !lists.excluded.includes(s));
  emit("radar.stage", { stage: "scan", symbols: scan });
  await sleep(900);
  emit("cycle", { state: "scan", used, max });
  const pre = scan.filter(() => Math.random() < 0.45).slice(0, 6);
  emit("radar.stage", { stage: "prescore", symbols: pre });
  simulateAi();
  await sleep(1100);
  const enabled = strategies.filter((s) => s.enabled).map((s) => s.name);
  const picks = pre.filter(() => Math.random() < 0.5).slice(0, 3);
  const sides: Record<string, Side> = {};
  const whys: Record<string, Partial<Position>> = {};
  for (const sym of picks) {
    sides[sym] = config.MARKET_TYPE?.value === "spot" || Math.random() < 0.6 ? "long" : "short";
    whys[sym] = {
      confidence: +(0.55 + Math.random() * 0.3).toFixed(2),
      strategy: enabled.length ? pick(enabled) : "ema_crossover",
      reasoning: pick(WHY),
    };
    emit("radar.stage", { stage: "signal", symbol: sym, side: sides[sym], ...whys[sym] });
    await sleep(500);
  }
  const rejected: Record<string, string> = {};
  const opened: string[] = [];
  for (const sym of picks) {
    emit("radar.stage", { stage: "guards", symbol: sym });
    await sleep(700);
    const reject = (reason: string) => {
      rejected[sym] = reason;
      emit("radar.stage", { stage: "rejected", symbol: sym, reason });
    };
    if (Object.keys(positions).length >= maxPositions()) reject("все слоты заняты");
    else if (positions[sym]) reject("по монете уже есть позиция");
    else if (Math.random() < 0.55) reject(pick(REJECTS));
    // Как у бота: спрашивает только в Live без авто-сделок, в Paper — сразу сделка.
    else if (!config.AUTO_EXECUTE?.value && !bot.paper) ask(sym, sides[sym], whys[sym]);
    else {
      openAndEmit(sym, sides[sym], whys[sym], scan.length, pre.includes(sym));
      opened.push(sym);
    }
  }
  saveCycle("scan", cycleNodes(scan.length, pre, sides, rejected, opened, used));
}

function openAndEmit(sym: string, side: Side, why: Partial<Position> = {}, scanN = SYMBOLS.length, inPre = true): void {
  const p = open(sym, side, why);
  p.trace = tradeSteps(side, why, scanN, inPre, p.entry);
  emit("radar.stage", { stage: "open", symbol: sym });
  emit("position.opened", {
    symbol: sym,
    trade_id: p.trade_id,
    side: side === "long" ? "buy" : "sell",
    entry: p.entry,
    qty: p.qty,
    stop_loss: p.sl,
    take_profit: p.tp,
    strategy: p.strategy,
    confidence: p.confidence,
    reasoning: p.reasoning,
    opened_at: p.opened_at,
    trace: p.trace,
  });
}

const decisionSec = (): number => Number(config.TELEGRAM_CONFIRM_TIMEOUT?.value ?? 60);

function ask(sym: string, side: Side, why: Partial<Position> = {}): void {
  const price = PRICES[sym];
  const dir = side === "long" ? 1 : -1;
  const id = `sig${++tradeSeq}`;
  const sig: PendingSignal = {
    id,
    symbol: sym,
    side,
    confidence: why.confidence ?? 0.68,
    strategy: why.strategy ?? "ema_crossover",
    reasoning: why.reasoning ?? pick(WHY),
    entry: price,
    stop_loss: price * (1 - dir * 0.015),
    take_profit: price * (1 + dir * 0.03),
    expires_at: Date.now() / 1000 + decisionSec(),
    on_timeout_execute: config.CONFIRM_ON_TIMEOUT?.value !== "skip",
    backtest: { win_rate: 0.34, trades: 2931, ev: -0.12 },
    live: { win_rate: 0.41, trades: 17, ev: 0.08 },
  };
  pending[id] = sig;
  emit("signal.pending", { ...sig });
  setTimeout(() => resolve(id, sig.on_timeout_execute, "timeout"), decisionSec() * 1000);
}

function resolve(id: string, confirmed: boolean, by: string): boolean {
  const sig = pending[id];
  if (!sig) return false;
  delete pending[id];
  emit("signal.resolved", { id, symbol: sig.symbol, confirmed, by });
  const reason = !confirmed
    ? "сделку отклонили"
    : positions[sig.symbol]
      ? "по монете уже есть позиция"
      : Object.keys(positions).length >= maxPositions()
        ? "все слоты заняты"
        : "";
  if (reason) emit("radar.stage", { stage: "rejected", symbol: sig.symbol, reason });
  else {
    const { strategy, confidence, reasoning } = sig;
    openAndEmit(sig.symbol, sig.side, { strategy, confidence, reasoning });
  }
  return true;
}

function start(): void {
  if (timers.length) return;
  timers = [
    setInterval(tick, 1000),
    setInterval(() => void cycle(), 7000),
    setInterval(() => emit("heartbeat", { alive: true }), 10_000),
  ];
  setTimeout(() => void cycle(), 600);
}

function stop(): void {
  timers.forEach(clearInterval);
  timers = [];
}

// ── REST ──────────────────────────────────────────────────────────────────────
function fail(status: number, code: string, message: string, fields?: Record<string, string>): never {
  throw new ApiError(status, code, message, fields);
}

/** Команда исполнилась сразу — как будто бот ответил в пределах 8 с. */
function done(body: Body, result: Record<string, unknown>) {
  const id = String(body.request_id ?? Math.random());
  return (results[id] = { request_id: id, status: "done", result });
}
const err = (code: string, message: string) => ({ ok: false, error: { code, message } });
const findPos = (tradeId: string) => Object.values(positions).find((p) => String(p.trade_id) === tradeId);
const NUMERIC = ["int", "float", "pct"];

const configHistory: { ts: number; by: string; changes: { key: string; label: string; old: unknown; new: unknown }[] }[] = [];

const jobGroup: Record<string, string> = { backtest: "backtest", train: "sac", tune: "sac", alpha_report: "alpha_report", live_divergence: "live_divergence", ai_isolation: "ai_isolation" };
const busyGroups = new Set<string>();
let modelBackup = false;
let backupProfile: string | null = null;
const trainedAt: Record<string, string | null> = { bluechip: "2026-09-12T10:00:00Z", altcoin: "2026-10-02T08:30:00Z", meme: null };
const PROFILE_INFO = [
  { id: "bluechip", label: "Блючипы", timeframe: "15m", train_top_n: 20 },
  { id: "altcoin", label: "Альткоины", timeframe: "5m", train_top_n: 100 },
  { id: "meme", label: "Мемкоины", timeframe: "3m", train_top_n: 50 },
];

/** Имитация фоновых задач бота: прогресс событиями, как job.progress/job.done. */
function startJob(job: string, b: Body) {
  const group = jobGroup[job];
  const profile = String(b.profile || bot.profile || "bluechip");
  if (!group) return fail(404, "unknown_job", "Такой задачи нет.");
  if (busyGroups.has(group)) return done(b, err("busy", "Эта задача уже выполняется — дождитесь конца."));
  busyGroups.add(group);
  const steps = job === "backtest" ? strategies.length : job === "train" || job === "tune" ? 20 : 3;
  let i = 0;
  const tick = setInterval(() => {
    i++;
    if (i <= steps) {
      const pct = Math.round((i / steps) * 100);
      const extra =
        job === "backtest"
          ? { symbol: "BTC/USDT", strategy: strategies[i - 1].name }
          : job === "train"
            ? { step: i * 5000, total: steps * 5000, eta_min: Math.round((steps - i) * 3) }
            : job === "tune"
              ? { trial: Math.ceil((i / steps) * 30), total: 30, best_pnl: 400 + i * 20, eta_min: Math.round((steps - i) * 9) }
              : {};
      emit("job.progress", { job, pct, ...extra });
      return;
    }
    clearInterval(tick);
    busyGroups.delete(group);
    if (job === "train" || job === "tune") {
      modelBackup = true;
      backupProfile = profile;
      trainedAt[profile] = new Date().toISOString();
    }
    const result: Record<string, unknown> =
      job === "backtest"
        ? { n: strategies.length, months: 12, timeframe: "15m", best: { strategy: "liquidity_sweep", sharpe: -0.05, trades: 1771 } }
        : job === "train" || job === "tune"
          ? { sac_pct: -4.2, bh_pct: 18.6, worse: true }
          : { output: job === "alpha_report" ? "strategy      n   sharpe   p-value\nema_crossover 120 -0.12    0.81\n0/20 значимы после поправки Bonferroni" : "расхождение исполнения: +0.04% на сделку" };
    emit("job.done", { job, ok: true, ...result });
    emit("config.changed", { keys: [] });
  }, 300);
  return done(b, { ok: true, started: true, job });
}

function xAccount(b: Body) {
  const tw = secrets.find((x) => x.service === "twitter")!;
  const email = String(b.email ?? "");
  if (!b.login || !b.password || !email.includes("@") || !b.email_password) fail(422, "invalid", "Проверьте поля.");
  tw.accounts = [...(tw.accounts ?? []).filter((a) => a.login !== b.login), { login: String(b.login), email: `${email[0]}***@${email.split("@")[1]}` }];
  tw.status = "warn";
  return tw;
}

function patchConfig(changes: Body) {
  const fields: Record<string, string> = {};
  for (const [key, raw] of Object.entries(changes)) {
    const def = schema.find((s) => s.key === key);
    const num = Number(raw);
    if (!def || def.readonly) fields[key] = "Эту настройку нельзя менять из приложения.";
    else if (!NUMERIC.includes(def.type)) continue;
    else if (!Number.isFinite(num)) fields[key] = "Нужно число.";
    else if (def.min !== null && num < def.min) fields[key] = `Не меньше ${def.min}.`;
    else if (def.max !== null && num > def.max) fields[key] = `Не больше ${def.max}.`;
  }
  if (Object.keys(fields).length) fail(400, "invalid", "Проверьте значения.", fields);
  const out = { applied_hot: [] as string[], pending_restart: [] as string[], failed: {} };
  const did: { key: string; label: string; old: unknown; new: unknown }[] = [];
  for (const [key, value] of Object.entries(changes)) {
    const def = schema.find((s) => s.key === key)!;
    const next = NUMERIC.includes(def.type) ? Number(value) : value;
    if (config[key]?.value !== next) did.push({ key, label: def.label, old: config[key]?.value, new: next });
    config[key] = { value: NUMERIC.includes(def.type) ? Number(value) : value, source: def.hot ? "live" : "ui" };
    (def.hot ? out.applied_hot : out.pending_restart).push(key);
  }
  if (did.length) configHistory.unshift({ ts: Date.now() / 1000, by: "app", changes: did });
  if (typeof changes.MODE === "string") bot.mode = changes.MODE;
  if (typeof changes.AI_PROVIDER === "string") {
    ai.pinned = changes.AI_PROVIDER;
    recomputeAi();
  }
  emit("config.changed", { keys: Object.keys(changes).sort() });
  return out;
}

function putSymbols(b: Body) {
  const norm = (l: unknown) =>
    (Array.isArray(l) ? l : [])
      .map((s) => String(s).toUpperCase().replace(/[^A-Z0-9/]/g, ""))
      .filter(Boolean)
      .map((s) => (s.includes("/") ? s : `${s}/USDT`));
  const forced = norm(b.forced);
  const excluded = norm(b.excluded);
  if (forced.some((s) => excluded.includes(s))) fail(400, "conflict", "Монета не может быть сразу в обоих списках.");
  lists.forced = [...new Set(forced)].sort();
  lists.excluded = [...new Set(excluded)].sort();
  emit("config.changed", { keys: ["SYMBOLS"] });
  return lists;
}

function secret(method: string, id: string, b: Body) {
  const s = secrets.find((x) => x.service === id) ?? fail(404, "unknown_service", "Нет такого сервиса.");
  if (method === "DELETE") {
    s.fields.forEach((f) => Object.assign(f, { set: false, last4: null, value: null }));
    Object.assign(s, { status: "unset", message: "", checked_at: null });
  } else {
    const values: Record<string, string> = b.fields ?? {};
    for (const f of s.fields) {
      const v = values[f.env];
      if (v) Object.assign(f, { set: true, last4: f.secret ? v.slice(-4) : null, value: f.secret ? null : v });
    }
    if (s.model && b.model) s.model.value = b.model;
    Object.assign(s, { status: "ok", message: "Демо: ключ принят без проверки.", checked_at: Date.now() / 1000 });
  }
  const aiProvider = ai.providers.find((p) => p.id === s.service);
  if (aiProvider) {
    aiProvider.key_set = s.fields.some((f) => f.set);
    aiProvider.usable = aiProvider.key_set || aiProvider.keyless_on;
    recomputeAi();
  }
  emit("secret.changed", { service: s.service, status: s.status });
  return { ...s, restart_required: true };
}

// Как настоящая сеть: каждый ответ — новая копия, а не ссылка на состояние мока.
async function request(method: string, path: string, body: unknown): Promise<unknown> {
  await sleep(120 + Math.random() * 180);
  return structuredClone(route(method, path, body));
}

let launcher: Record<string, unknown> = { local: true, autostart: true, keep_awake: false, phone: true, helper: true, result: null };

function route(method: string, path: string, body: unknown): unknown {
  const b = (body ?? {}) as Body;
  const url = new URL(path, "http://mock");
  const p = url.pathname;
  const q = url.searchParams;
  const route = `${method} ${p}`;

  switch (route) {
    case "GET /api/state":
      return state();
    case "GET /api/positions":
      return state().positions;
    case "GET /api/metrics":
      return metrics(q.get("period") ?? "30d");
    case "GET /api/metrics/execution":
      return execution(q.get("period") ?? "30d");
    case "GET /api/journal": {
      const period = q.get("period") ?? "30d";
      const since = PERIOD_MS[period] ? Date.now() - PERIOD_MS[period] : 0;
      const trades = closed.filter((c) => c.exit >= since).map((c) => c.j);
      return { period, limit: 1000, trades: trades.reverse().slice(0, 1000) };
    }
    case "GET /api/pnl/daily":
      return daily(Number(q.get("days") ?? 35));
    case "GET /api/strategies":
      return strategies;
    case "GET /api/risk/presets":
      return {
        conservative: { MAX_POSITIONS: 2, RISK_PER_TRADE: 0.01, DRAWDOWN_SCALE_ENABLED: true, LEVERAGE_MODE: "fixed" },
        moderate: { MAX_POSITIONS: 3, RISK_PER_TRADE: 0.02, DRAWDOWN_SCALE_ENABLED: true, LEVERAGE_MODE: "volatility" },
        aggressive: { MAX_POSITIONS: 5, RISK_PER_TRADE: 0.04, DRAWDOWN_SCALE_ENABLED: false, LEVERAGE_MODE: "full" },
      };
    case "GET /api/config/schema":
      return schema;
    case "GET /api/config":
      return config;
    case "GET /api/config/history":
      return configHistory;
    case "GET /api/secrets":
      return secrets;
    case "POST /api/secrets/telegram/chat":
      return { chats: [{ id: "123456789", name: "Дмитрий" }] };
    case "GET /api/setup": {
      const has = (id: string) => Boolean(secrets.find((s) => s.service === id)?.fields.every((f) => f.set || f.optional));
      return { keys_in_app: true, bybit: has("bybit"), telegram: has("telegram") };
    }
    case "GET /api/signals/pending":
      return Object.values(pending);
    case "GET /api/symbols":
      return lists;
    case "GET /api/ai":
      return ai;
    case "PUT /api/ai/chain":
      return applyAi(b);
    case "GET /api/alerts":
      return alerts;
    case "GET /api/candles":
      return candles(q.get("symbol") ?? "BTC/USDT", Number(q.get("limit") ?? 200));

    case "POST /api/bot/pause":
    case "POST /api/bot/resume":
      bot.paused = route.endsWith("pause");
      emit("cycle", { state: bot.paused ? "paused" : "scan" });
      return done(b, { ok: true, paused: bot.paused });
    case "POST /api/bot/panic": {
      const syms = Object.keys(positions);
      syms.forEach((s) => closePos(s, 1, "telegram_closeall"));
      bot.paused = true;
      emit("cycle", { state: "paused" });
      return done(b, { ok: true, closed: syms.length, paused: true });
    }
    case "POST /api/config/reset":
      strategies.forEach((s) => (s.enabled = !s.experimental));
      emit("config.changed", { keys: ["STRATEGIES", "SYMBOLS", "MARKET_PROFILE", "MODE"] });
      return { ok: true };
    case "GET /api/jobs":
      return { running: Object.fromEntries([...busyGroups].map((g) => [g, "app"])), progress: {},
        rollback_available: modelBackup,
        backup_profile: backupProfile,
        profiles: PROFILE_INFO.map((p) => ({
          ...p,
          model_path: `models/sac_model_${p.id}.zip`,
          model_exists: trainedAt[p.id] !== null,
          trained_at: trainedAt[p.id],
          active: p.id === bot.profile,
        })),
      };
    case "POST /api/model/rollback":
      if (!modelBackup) return done(b, err("no_backup", "Резервная копия модели не найдена."));
      modelBackup = false;
      emit("model.rolled_back", {});
      return done(b, { ok: true, rolled_back: true });
    case "POST /api/bot/restart":
      return done(b, { ok: true, restarting: true });
    case "POST /api/account/mode":
      if (!b.paper && b.pin !== "1234") return done(b, err("bad_pin", "Неверный PIN. В демо PIN — 1234."));
      bot.paper = Boolean(b.paper);
      emit("account.mode", { paper: bot.paper });
      return done(b, { ok: true, paper: bot.paper });
    case "PATCH /api/config":
      return patchConfig(b);
    case "POST /api/profile":
      if (!["bluechip", "altcoin", "meme"].includes(b.name)) fail(400, "bad_profile", "Профиль — bluechip, altcoin или meme.");
      bot.profile = b.name;
      emit("config.changed", { keys: ["MARKET_PROFILE"], profile: b.name });
      return { profile: b.name };
    case "POST /api/strategies/reset":
      strategies.forEach((s) => (s.enabled = !s.experimental));
      emit("config.changed", { keys: ["STRATEGIES"] });
      return { ok: true };
    case "PUT /api/symbols":
      return putSymbols(b);
    case "POST /api/auth/logout":
      return { ok: true };
  }

  let m: RegExpMatchArray | null;
  if (method === "GET" && (m = p.match(/^\/api\/ai\/providers\/(.+)\/models$/))) {
    const prov = ai.providers.find((x) => x.id === m![1]) ?? fail(404, "unknown_provider", "Нет такого провайдера.");
    if (!prov.key_set && !prov.key_optional) fail(400, "no_key", "Сначала добавьте ключ.");
    return { provider: prov.id, models: [...prov.models.map((x) => x.id), `${prov.id}-preview-2026`].sort() };
  }
  if ((m = p.match(/^\/api\/commands\/(.+)$/))) return results[m[1]] ?? { request_id: m[1], status: "pending" };
  if (method === "POST" && (m = p.match(/^\/api\/jobs\/(.+)$/))) return startJob(m[1], b);
  if (method === "POST" && (m = p.match(/^\/api\/positions\/(.+)\/close$/))) {
    const pos = findPos(m[1]);
    if (!pos) return done(b, err("not_found", "Позиция уже закрыта."));
    const f = Number(b.fraction ?? 1);
    if (!(f === 1 || (f >= 0.1 && f <= 0.9))) return done(b, err("bad_fraction", "Доля — от 10% до 90% или 100%."));
    closePos(pos.symbol, f);
    return done(b, { ok: true });
  }
  if (method === "PATCH" && (m = p.match(/^\/api\/positions\/(.+)$/))) {
    const pos = findPos(m[1]);
    if (!pos) return done(b, err("not_found", "Позиция уже закрыта."));
    const sl = Number(b.sl ?? pos.sl);
    const tp = Number(b.tp ?? pos.tp);
    const ok = pos.side === "long" ? sl < pos.price && pos.price < tp : tp < pos.price && pos.price < sl;
    if (!ok) return done(b, err("bad_levels", pos.side === "long" ? "Для LONG: SL ниже цены, TP выше." : "Для SHORT: SL выше цены, TP ниже."));
    Object.assign(pos, { sl, tp, trailing: b.trailing ?? pos.trailing });
    emit("position.updated", publicPos(pos));
    return done(b, { ok: true });
  }
  if (method === "POST" && (m = p.match(/^\/api\/signals\/(.+)\/(confirm|reject)$/))) {
    return resolve(m[1], m[2] === "confirm", "app") ? done(b, { ok: true }) : done(b, err("not_pending", "Предложение уже неактуально."));
  }
  if (method === "PATCH" && (m = p.match(/^\/api\/strategies\/(.+)$/))) {
    const name = m[1];
    const s = strategies.find((x) => x.name === name) ?? fail(404, "unknown_strategy", "Нет такой стратегии.");
    s.enabled = Boolean(b.enabled);
    emit("config.changed", { keys: ["STRATEGIES"] });
    return s;
  }
  if (method === "PATCH" && (m = p.match(/^\/api\/alerts\/(.+)$/))) {
    const kind = m[1];
    const a = alerts.find((x) => x.kind === kind) ?? fail(404, "unknown_alert", "Нет такого уведомления.");
    if (b.cooldown_min !== undefined && !(b.cooldown_min >= 0 && b.cooldown_min <= 1440))
      fail(400, "bad_cooldown", "Пауза — от 0 до 1440 минут.");
    if (b.enabled !== undefined) a.enabled = Boolean(b.enabled);
    if (b.cooldown_min !== undefined) a.cooldown_min = Number(b.cooldown_min);
    return a;
  }
  if (method === "GET" && p === "/api/launcher") return { ...launcher, at: Date.now() / 1000 };
  if (method === "POST" && p === "/api/launcher") {
    const key = String(b.setting) as "autostart" | "keep_awake" | "phone";
    if (!["autostart", "keep_awake", "phone"].includes(key) || typeof b.on !== "boolean" || (key === "phone" && b.on))
      fail(400, "not_allowed", "Это можно сделать только командой на компьютере.");
    const id = Math.random().toString(16).slice(2, 10);
    launcher = { ...launcher, [key]: b.on, result: { id, ok: true, message: b.on ? "Включено." : "Выключено.", at: Date.now() / 1000 } };
    return { id };
  }
  if (method === "POST" && p === "/api/secrets/twitter/accounts") return xAccount(b);
  if (method === "DELETE" && (m = p.match(/^\/api\/secrets\/twitter\/accounts\/(.+)$/))) {
    const tw = secrets.find((x) => x.service === "twitter")!;
    tw.accounts = (tw.accounts ?? []).filter((a) => a.login !== decodeURIComponent(m![1]));
    return tw;
  }
  if ((m = p.match(/^\/api\/secrets\/(.+)$/)) && (method === "PUT" || method === "DELETE")) return secret(method, m[1], b);
  return fail(404, "not_found", `Демо не знает ${route}.`);
}

export const mock = {
  request,
  subscribe(onEvent: (e: BotEvent) => void): () => void {
    listeners.add(onEvent);
    onEvent({ type: "state", ts: Date.now() / 1000, data: state() });
    start();
    return () => {
      listeners.delete(onEvent);
      if (!listeners.size) stop();
    };
  },
};
