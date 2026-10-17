import { create } from "zustand";

import { ApiError, api, command, isMock, subscribe } from "./lib/api";
import { arrow, base, exitReason, plural, usd } from "./lib/format";
import * as platform from "./lib/platform";
import type {
  AiChainChanges,
  AiState,
  Alert,
  BotEvent,
  BotState,
  ConfigValues,
  DailyRow,
  Guard,
  Journal,
  Metrics,
  PendingSignal,
  Period,
  Position,
  SecretItem,
  SettingDef,
  SetupInfo,
  SetupStep,
  Side,
  Strategy,
  Symbols,
  Tab,
} from "./lib/types";

// Данные событий приходят нетипизированными — проверяем по месту.
type Data = Record<string, any>;

export type FeedKind = "open" | "win" | "loss" | "skip" | "guard" | "edit";
export interface FeedItem {
  id: number;
  at: number;
  kind: FeedKind;
  text: string;
}

export interface Radar {
  cycle: number;
  scan: string[];
  prescore: string[];
  signal: Record<string, { side: Side; confidence: number; strategy: string; reasoning?: string }>;
  guards: string[];
  /** lane — на каком этапе отсеяна: 0 скан, 1 отбор, 2 сигнал, 3 проверки. */
  rejected: Record<string, { reason: string; at: number; lane: number }>;
  opened: string[];
  note: string;
}

const emptyRadar = (cycle = 0): Radar => ({
  cycle,
  scan: [],
  prescore: [],
  signal: {},
  guards: [],
  rejected: {},
  opened: [],
  note: "",
});

export const GUARD_TEXT: Record<string, string> = {
  drawdown_halt: "Входы остановлены: предельная просадка",
  daily_limit: "Входы остановлены: дневной лимит убытка",
  atr_spike: "Резкий всплеск волатильности — позиции закрыты",
  macro: "Пауза вокруг макро-новостей",
  circuit_breaker: "Остановка после серии убытков",
};

export interface Store {
  ready: boolean;
  needLogin: boolean;
  wsUp: boolean;
  alive: boolean;
  bot: BotState["bot"] | null;
  balance: number;
  slots: { used: number; max: number };
  positions: Record<string, Position>;
  cycle: { state: string; used?: number; max?: number };
  /** Когда пришло последнее событие цикла (мс): по нему видно, жив ли цикл бота. */
  cycleAt: number | null;
  guards: Record<string, Guard>;
  radar: Radar;
  feed: FeedItem[];
  pending: PendingSignal[];
  period: Period;
  metrics: Partial<Record<Period, Metrics>>;
  daily: DailyRow[];
  strategies: Strategy[];
  schema: SettingDef[];
  config: ConfigValues;
  alerts: Alert[];
  symbols: Symbols;
  secrets: SecretItem[] | null;
  secretsNote: string;
  ai: AiState | null;
  journal: Journal | null;
  setupInfo: SetupInfo | null;
  /** Открытый шаг мастера первого запуска; null — мастер закрыт. */
  setup: SetupStep | null;
  tab: Tab;
  /** Поиск параметра по Cmd/Ctrl+K: null — не запрошен, строка — что подставить. */
  paramSearch: string | null;
  toast: { id: number; text: string } | null;

  boot(): Promise<void>;
  setTab(tab: Tab): void;
  openSetup(step: SetupStep | null): void;
  /** Открыть «Настройки → Параметры» с фокусом в поиске. */
  searchParams(q?: string): void;
  clearParamSearch(): void;
  setPeriod(p: Period): void;
  say(text: string): void;
  apply(e: BotEvent): void;
  /** Выполнить действие: успех — тост ok, ошибка — тост с текстом сервера. */
  act(fn: () => Promise<unknown>, ok?: string): Promise<boolean>;

  loadState(): Promise<void>;
  loadMetrics(p?: Period): Promise<void>;
  loadDaily(): Promise<void>;
  loadStrategies(): Promise<void>;
  loadConfig(): Promise<void>;
  loadAlerts(): Promise<void>;
  loadSymbols(): Promise<void>;
  loadSecrets(): Promise<void>;
  loadAi(): Promise<void>;
  loadJournal(p: Period): Promise<void>;
  loadSetup(): Promise<void>;

  requestCode(): Promise<boolean>;
  verifyCode(code: string): Promise<boolean>;
}

let feedSeq = 0;
let toastSeq = 0;
let unsubscribe: (() => void) | null = null;
let refreshTimer: ReturnType<typeof setTimeout> | undefined;

const MAX_FEED = 60;

const sideOf = (s: unknown): Side => (s === "sell" || s === "short" ? "short" : "long");

export const useStore = create<Store>((set, get) => {
  const feed = (kind: FeedKind, text: string): void =>
    set((s) => ({
      feed: [{ id: ++feedSeq, at: Date.now(), kind, text }, ...s.feed].slice(0, MAX_FEED),
    }));

  /** После закрытия сделки меняются баланс и метрики — подтягиваем разом. */
  const refreshSoon = (): void => {
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(() => {
      void get().loadState().catch(() => undefined);
      void get().loadMetrics().catch(() => undefined);
      void get().loadDaily().catch(() => undefined);
      const j = get().journal;
      if (j) void get().loadJournal(j.period).catch(() => undefined);
    }, 800);
  };

  // Снимок /api/state не знает последней цены и флага трейлинга — берём из тиков.
  const positionsFrom = (list: Position[]): Record<string, Position> => {
    const prev = get().positions;
    return Object.fromEntries(
      list.map((p) => {
        const old = prev[p.symbol];
        return [p.symbol, { ...old, ...p, price: old?.price ?? p.price, trailing: old?.trailing ?? true }];
      }),
    );
  };

  const applyState = (st: BotState): void =>
    set({
      bot: st.bot,
      alive: st.bot.alive,
      balance: st.balance,
      slots: st.slots,
      positions: positionsFrom(st.positions),
    });

  return {
    ready: false,
    needLogin: false,
    wsUp: false,
    alive: true,
    bot: null,
    balance: 0,
    slots: { used: 0, max: 0 },
    positions: {},
    cycle: { state: "scan" },
    cycleAt: null,
    guards: {},
    radar: emptyRadar(),
    feed: [],
    pending: [],
    period: "30d",
    metrics: {},
    daily: [],
    strategies: [],
    schema: [],
    config: {},
    alerts: [],
    symbols: { forced: [], excluded: [] },
    secrets: null,
    secretsNote: "",
    ai: null,
    journal: null,
    setupInfo: null,
    setup: null,
    paramSearch: null,
    tab: "terminal",
    toast: null,

    async boot() {
      platform.init();
      // Десктоп без адреса сервера: без этого запрос уйдёт в tauri://localhost/api/…
      if (!isMock && platform.kind === "desktop" && !platform.apiBase()) {
        set({ needLogin: true, ready: true });
        return;
      }
      if (!isMock && platform.kind === "miniapp" && !platform.getToken()) {
        try {
          const r = await api<{ token: string }>("POST", "/api/auth/telegram", {
            initData: platform.initData(),
          });
          platform.setToken(r.token);
        } catch (e) {
          get().say(e instanceof ApiError ? e.message : "Не удалось войти через Telegram.");
        }
      }
      try {
        await get().loadState();
      } catch (e) {
        // На десктопе любая ошибка — на вход: там можно исправить адрес сервера.
        if (e instanceof ApiError && (e.status === 401 || platform.kind === "desktop")) {
          if (e.status !== 401) get().say(e.message);
          set({ needLogin: true, ready: true });
          return;
        }
        get().say(e instanceof ApiError ? e.message : "Нет связи с сервером бота.");
      }
      set({ ready: true, needLogin: false });
      void Promise.allSettled([
        get().loadMetrics(),
        get().loadDaily(),
        get().loadStrategies(),
        get().loadConfig(),
        get().loadAlerts(),
        get().loadSymbols(),
        get().loadSecrets(),
        get().loadAi(),
        get().loadSetup(),
        api<PendingSignal[]>("GET", "/api/signals/pending").then((pending) => set({ pending })),
      ]);
      unsubscribe?.();
      unsubscribe = subscribe(
        (e) => get().apply(e),
        (up) => set({ wsUp: up }),
        () => {
          platform.setToken(null);
          set({ needLogin: true });
        },
      );
    },

    setTab: (tab) => set({ tab, setup: null }),
    openSetup: (setup) => set({ setup }),
    searchParams: (q = "") => set({ tab: "settings", setup: null, paramSearch: q }),
    clearParamSearch: () => set({ paramSearch: null }),
    setPeriod(period) {
      set({ period });
      if (!get().metrics[period]) void get().loadMetrics(period).catch(() => undefined);
    },
    say: (text) => set({ toast: { id: ++toastSeq, text } }),

    async act(fn, ok) {
      try {
        await fn();
        if (ok) get().say(ok);
        platform.haptic("success");
        return true;
      } catch (e) {
        platform.haptic("error");
        if (e instanceof ApiError && e.status === 401 && platform.kind !== "miniapp") {
          set({ needLogin: true });
        }
        get().say(e instanceof ApiError ? e.message : "Что-то пошло не так.");
        return false;
      }
    },

    apply(e) {
      const d = e.data as Data;
      switch (e.type) {
        case "state":
          applyState(d as unknown as BotState);
          break;
        case "heartbeat":
          set({ alive: Boolean(d.alive) });
          break;
        case "tick": {
          const cur = get().positions[d.symbol];
          // Тик несёт все поля позиции — карточка появится и без снимка.
          const p: Position = cur
            ? { ...cur }
            : { id: String(d.trade_id ?? d.symbol), trade_id: d.trade_id, symbol: d.symbol, side: sideOf(d.side), entry: d.entry, qty: d.qty, sl: d.sl, tp: d.tp };
          set({
            positions: {
              ...get().positions,
              [d.symbol]: {
                ...p,
                price: d.price,
                sl: d.sl ?? p.sl,
                tp: d.tp ?? p.tp,
                qty: d.qty ?? p.qty,
                trailing: d.trailing ?? p.trailing,
              },
            },
          });
          break;
        }
        case "position.opened": {
          const p: Position = {
            id: String(d.trade_id ?? d.symbol),
            trade_id: d.trade_id,
            symbol: d.symbol,
            side: sideOf(d.side),
            entry: d.entry,
            qty: d.qty,
            sl: d.stop_loss,
            tp: d.take_profit,
            price: d.entry,
            trailing: true,
            strategy: d.strategy ?? get().radar.signal[d.symbol]?.strategy,
            confidence: d.confidence ?? get().radar.signal[d.symbol]?.confidence,
            reasoning: d.reasoning || get().radar.signal[d.symbol]?.reasoning,
            opened_at: d.opened_at ?? Date.now() / 1000,
          };
          set((s) => ({ positions: { ...s.positions, [p.symbol]: p } }));
          feed("open", `Открыл ${base(p.symbol)} ${p.side === "long" ? "LONG" : "SHORT"}`);
          platform.haptic("success");
          refreshSoon();
          break;
        }
        case "position.closed": {
          const old = get().positions[d.symbol];
          const rest = { ...get().positions };
          delete rest[d.symbol];
          set({ positions: rest });
          const why = d.reason ? ` ${exitReason(d.reason)}` : "";
          const p = old && { ...old, price: d.price ?? old.price };
          if (p?.price !== undefined) {
            const { pnl } = positionPnl(p);
            feed(pnl >= 0 ? "win" : "loss", `Закрыл ${base(d.symbol)}${why}: ${arrow(pnl)} ${usd(pnl)}`);
          } else feed("edit", `Закрыл ${base(d.symbol)}${why}`);
          refreshSoon();
          break;
        }
        case "position.updated": {
          const cur = get().positions[d.symbol];
          if (cur)
            set({
              positions: {
                ...get().positions,
                [d.symbol]: { ...cur, qty: d.qty, sl: d.sl, tp: d.tp, trailing: d.trailing },
              },
            });
          break;
        }
        case "radar.stage": {
          const reachedGuards = get().radar.guards.includes(d.symbol);
          set({ radar: reduceRadar(get().radar, d) });
          if (d.stage === "rejected" && reachedGuards) {
            feed("skip", `Пропустил ${base(d.symbol)}: ${d.reason}`);
          }
          break;
        }
        case "cycle": {
          // Пауза могла прийти из Telegram — цикл бота знает о ней раньше снимка.
          const bot = get().bot;
          set({ cycle: d as Store["cycle"], cycleAt: Date.now(), bot: bot && { ...bot, paused: d.state === "paused" } });
          break;
        }
        case "guard":
          set((s) => ({ guards: { ...s.guards, [d.kind]: d as Guard } }));
          if (d.active) feed("guard", d.detail ? String(d.detail) : GUARD_TEXT[d.kind] ?? "Защита");
          break;
        case "signal.pending":
          set((s) => ({ pending: [...s.pending.filter((p) => p.id !== d.id), d as PendingSignal] }));
          feed("open", `Нашёл сделку ${base(d.symbol)} — жду подтверждения`);
          platform.haptic("warning");
          break;
        case "signal.resolved": {
          set((s) => ({ pending: s.pending.filter((p) => p.id !== d.id) }));
          const how = d.by === "timeout" ? "нет ответа" : d.by === "app" ? "в приложении" : "в Telegram";
          feed(
            d.confirmed ? "open" : "skip",
            `${base(d.symbol)}: ${d.confirmed ? "сделка подтверждена" : "сделка пропущена"} (${how})`,
          );
          break;
        }
        case "config.changed": {
          void get().loadConfig().catch(() => undefined);
          const keys: string[] = d.keys ?? [];
          if (keys.includes("STRATEGIES")) void get().loadStrategies().catch(() => undefined);
          if (keys.includes("SYMBOLS")) void get().loadSymbols().catch(() => undefined);
          if (keys.includes("MARKET_PROFILE") || keys.includes("MODE")) void get().loadState().catch(() => undefined);
          if (keys.includes("AI_CHAIN") || keys.includes("AI_PROVIDER")) void get().loadAi().catch(() => undefined);
          break;
        }
        case "ai.status": {
          // Живой статус провайдера: ответил, лимит, нет денег, пауза до…
          const ai = get().ai;
          const ok = d.status === "ok";
          if (ok) set((st) => ({ bot: st.bot && { ...st.bot, ai: d.provider } }));
          if (!ai) break;
          set({
            ai: {
              ...ai,
              last: ok ? d.provider : ai.last,
              providers: ai.providers.map((p) =>
                p.id === d.provider
                  ? { ...p, status: d.status, cooldown_until: d.until ?? null, calls_today: p.calls_today + 1 }
                  : p,
              ),
            },
          });
          break;
        }
        case "account.mode":
          void get().loadState().catch(() => undefined);
          feed("guard", d.paper ? "Счёт → PAPER" : "Счёт → LIVE: реальные деньги");
          break;
        case "secret.changed":
          void get().loadSecrets();
          void get().loadAi().catch(() => undefined);
          void get().loadSetup().catch(() => undefined);
          break;
        default:
          break;
      }
    },

    async loadState() {
      applyState(await api<BotState>("GET", "/api/state"));
    },
    async loadMetrics(p) {
      const period = p ?? get().period;
      const m = await api<Metrics>("GET", `/api/metrics?period=${period}`);
      set((s) => ({ metrics: { ...s.metrics, [period]: m } }));
    },
    async loadSetup() {
      set({ setupInfo: await api<SetupInfo>("GET", "/api/setup") });
    },
    async loadJournal(period) {
      set({ journal: await api<Journal>("GET", `/api/journal?period=${period}`) });
    },
    async loadDaily() {
      set({ daily: await api<DailyRow[]>("GET", "/api/pnl/daily?days=35") });
    },
    async loadStrategies() {
      set({ strategies: await api<Strategy[]>("GET", "/api/strategies") });
    },
    async loadConfig() {
      const [schema, config] = await Promise.all([
        get().schema.length ? get().schema : api<SettingDef[]>("GET", "/api/config/schema"),
        api<ConfigValues>("GET", "/api/config"),
      ]);
      set({ schema, config });
    },
    async loadAlerts() {
      set({ alerts: await api<Alert[]>("GET", "/api/alerts") });
    },
    async loadSymbols() {
      set({ symbols: await api<Symbols>("GET", "/api/symbols") });
    },
    async loadSecrets() {
      try {
        set({ secrets: await api<SecretItem[]>("GET", "/api/secrets"), secretsNote: "" });
      } catch (e) {
        set({ secrets: null, secretsNote: e instanceof ApiError ? e.message : "Ключи недоступны." });
      }
    },

    async loadAi() {
      set({ ai: await api<AiState>("GET", "/api/ai") });
    },

    requestCode() {
      return get().act(() => api("POST", "/api/auth/desktop/request"), "Код отправлен вам в Telegram");
    },
    async verifyCode(code) {
      try {
        const r = await api<{ token: string }>("POST", "/api/auth/desktop/verify", { code });
        platform.setToken(r.token);
        set({ needLogin: false });
        await get().boot();
        return true;
      } catch (e) {
        get().say(e instanceof ApiError ? e.message : "Не удалось войти.");
        return false;
      }
    },
  };
});

const n = (list: unknown): number => (Array.isArray(list) ? list.length : 0);

/** Конвейер радара: scan → prescore → signal → guards → rejected | open. */
export function reduceRadar(r: Radar, d: Data): Radar {
  const sym: string = d.symbol;
  switch (d.stage) {
    case "scan":
      return {
        ...emptyRadar(r.cycle + 1),
        scan: d.symbols ?? [],
        note: `Сканирую ${n(d.symbols)} ${plural(n(d.symbols), ["монету", "монеты", "монет"])}`,
      };
    case "prescore":
      return { ...r, prescore: d.symbols ?? [], note: `В отбор прошли ${n(d.symbols)}: объём, ATR и тренд` };
    case "signal":
      return {
        ...r,
        signal: {
          ...r.signal,
          [sym]: { side: d.side, confidence: d.confidence, strategy: d.strategy, reasoning: d.reasoning },
        },
        note: `Сигнал по ${base(sym)}: ${d.side === "long" ? "LONG" : "SHORT"}`,
      };
    case "guards":
      return {
        ...r,
        guards: r.guards.includes(sym) ? r.guards : [...r.guards, sym],
        note: `Проверяю риски по ${base(sym)}…`,
      };
    case "rejected": {
      const lane = r.guards.includes(sym) ? 3 : r.signal[sym] ? 2 : r.prescore.includes(sym) ? 1 : 0;
      return {
        ...r,
        guards: r.guards.filter((s) => s !== sym),
        rejected: { ...r.rejected, [sym]: { reason: d.reason, at: Date.now(), lane } },
        note: `Отсеял ${base(sym)}: ${d.reason}`,
      };
    }
    case "open":
      return {
        ...r,
        guards: r.guards.filter((s) => s !== sym),
        opened: [...r.opened, sym],
        note: `Открыл ${base(sym)}. SL и TP на бирже.`,
      };
    default:
      return r;
  }
}

/** P&L позиции по последней цене. */
export function positionPnl(p: Position): { pnl: number; move: number } {
  const price = p.price ?? p.entry;
  const dir = p.side === "long" ? 1 : -1;
  return { pnl: (price - p.entry) * p.qty * dir, move: (price / p.entry - 1) * 100 * dir };
}

// ── Действия (вызываются из компонентов) ──────────────────────────────────────

const st = () => useStore.getState();

export interface PatchResult {
  applied_hot: string[];
  pending_restart: string[];
}

export const actions = {
  pause: () => st().act(() => command("POST", "/api/bot/pause"), "Бот на паузе"),
  resume: () => st().act(() => command("POST", "/api/bot/resume"), "Бот снова работает"),
  panic: () =>
    st().act(async () => {
      const r = await command("POST", "/api/bot/panic");
      const c = Number(r.closed ?? 0);
      st().say(`Закрыто ${c} ${plural(c, ["позиция", "позиции", "позиций"])}. Бот на паузе.`);
    }),
  restart: () => st().act(() => command("POST", "/api/bot/restart"), "Бот перезапускается…"),
  closePosition: (p: Position, fraction = 1) =>
    st().act(
      () => command("POST", `/api/positions/${p.trade_id}/close`, { fraction }),
      fraction === 1 ? `${base(p.symbol)} закрыт по рынку` : `Закрыто ${Math.round(fraction * 100)}% ${base(p.symbol)}`,
    ),
  updatePosition: (p: Position, changes: { sl?: number; tp?: number; trailing?: boolean }) =>
    st().act(() => command("PATCH", `/api/positions/${p.trade_id}`, changes), "Сохранено"),
  answerSignal: (id: string, confirm: boolean) =>
    st().act(
      () => command("POST", `/api/signals/${id}/${confirm ? "confirm" : "reject"}`),
      confirm ? "Сделка подтверждена" : "Сделка пропущена",
    ),
  accountMode: (paper: boolean, pin = "") =>
    st().act(() => command("POST", "/api/account/mode", { paper, pin }), paper ? "Вернулись в Paper" : "Включён Live"),
  async patchConfig(changes: Record<string, unknown>): Promise<PatchResult | null> {
    let result: PatchResult | null = null;
    const ok = await st().act(async () => {
      result = await api<PatchResult>("PATCH", "/api/config", changes);
    });
    if (!ok || !result) return null;
    const r: PatchResult = result;
    st().say(
      r.pending_restart.length
        ? `Сразу: ${r.applied_hot.length}. После перезапуска: ${r.pending_restart.length}.`
        : "Применено — бот уже работает с новыми настройками",
    );
    await st().loadConfig().catch(() => undefined);
    return r;
  },
  setProfile: (name: string) =>
    st().act(() => api("POST", "/api/profile", { name }).then(() => st().loadState()), "Профиль рынка переключён"),
  setStrategy: (name: string, enabled: boolean) =>
    st().act(() => api("PATCH", `/api/strategies/${name}`, { enabled }).then(() => st().loadStrategies())),
  resetStrategies: () =>
    st().act(() => api("POST", "/api/strategies/reset").then(() => st().loadStrategies()), "Все стратегии включены"),
  putSymbols: (symbols: Symbols) =>
    st().act(() => api<Symbols>("PUT", "/api/symbols", symbols).then((v) => useStore.setState({ symbols: v }))),
  patchAlert: (kind: string, changes: Partial<Alert>) =>
    st().act(() => api("PATCH", `/api/alerts/${kind}`, changes).then(() => st().loadAlerts())),
  putSecret: (service: string, fields: Record<string, string>, model?: string) =>
    st().act(
      () => api("PUT", `/api/secrets/${service}`, { fields, ...(model ? { model } : {}) }).then(() => st().loadSecrets()),
      "Ключ сохранён. Заработает после перезапуска бота.",
    ),
  putAiChain: (changes: AiChainChanges, ok?: string) =>
    st().act(() => api<AiState>("PUT", "/api/ai/chain", changes).then((ai) => useStore.setState({ ai })), ok),
  /** Живой список моделей провайдера по сохранённому ключу. */
  aiModels: async (provider: string): Promise<string[]> => {
    try {
      return (await api<{ models: string[] }>("GET", `/api/ai/providers/${provider}/models`)).models;
    } catch (e) {
      st().say(e instanceof ApiError ? e.message : "Не удалось загрузить список моделей.");
      return [];
    }
  },
  deleteSecret: (service: string) =>
    st().act(() => api("DELETE", `/api/secrets/${service}`).then(() => st().loadSecrets()), "Ключ удалён"),
};
