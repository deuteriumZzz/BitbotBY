// Данные событий приходят нетипизированными — проверяем по месту.

import { create } from "zustand";

import { ApiError, api, isMock, subscribe } from "../lib/api";
import { arrow, base, exitReason, usd } from "../lib/format";
import * as platform from "../lib/platform";
import type {
  AiState,
  Alert,
  BotEvent,
  BotNode,
  BotState,
  CycleTrace,
  ConfigValues,
  DailyRow,
  Guard,
  JobDone,
  JobId,
  JobsStatus,
  JobState,
  ExecutionMetrics,
  Journal,
  Metrics,
  PendingSignal,
  Period,
  Position,
  ProfileInfo,
  SecretItem,
  RiskPreset,
  SettingDef,
  SetupInfo,
  SetupStep,
  Side,
  Strategy,
  Symbols,
  TrustedPost,
  Tab,
} from "../lib/types";

import { Data, Radar, emptyRadar, reduceRadar } from "./radar";
import { cycleLine } from "../lib/cycle";
import { EP } from "../api/endpoints";
import { REFRESH_DEBOUNCE_MS } from "../constants";

export type FeedKind = "open" | "win" | "loss" | "skip" | "guard" | "edit" | "cycle";

export interface FeedItem {
  id: number;
  at: number;
  kind: FeedKind;
  text: string;
}

export const JOB_TITLE: Record<string, string> = {
  backtest: "бэктест стратегий",
  train: "обучение SAC",
  tune: "тюнинг и обучение SAC",
  alpha_report: "значимость альфы",
  live_divergence: "Live vs бэктест",
  ai_isolation: "проверка AI",
};

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
  /** Итог последнего цикла по узлам конвейера (вкладка «Бот»). */
  lastCycle: CycleTrace | null;
  /** Готовность режимов решать (причина — код из бота); null, пока бот не публиковал. */
  modeReadiness: BotState["mode_readiness"];
  /** Что включено и состояние наблюдения (WS, копия БД, аварийный уровень); null, пока бот не публиковал. */
  opsStatus: BotState["ops_status"];
  /** Последний пост доверенного канала: вход по нему или причина отказа. */
  trusted: TrustedPost | null;
  balance: number;
  /** Опора дневного лимита убытка (капитал на 00:00 UTC); null — бот ещё не начал день. */
  dayRef: BotState["day_ref"];
  slots: { used: number; max: number };
  positions: Record<string, Position>;
  market: BotState["market"] | null;
  cycle: { state: string; used?: number; max?: number };
  /** Когда пришло последнее событие цикла (мс): по нему видно, жив ли цикл бота. */
  cycleAt: number | null;
  guards: Record<string, Guard>;
  radar: Radar;
  feed: FeedItem[];
  pending: PendingSignal[];
  period: Period;
  metrics: Partial<Record<Period, Metrics>>;
  execution: Partial<Record<Period, ExecutionMetrics>>;
  daily: DailyRow[];
  strategies: Strategy[];
  schema: SettingDef[];
  riskPresets: Record<string, RiskPreset>;
  config: ConfigValues;
  alerts: Alert[];
  symbols: Symbols;
  secrets: SecretItem[] | null;
  secretsNote: string;
  ai: AiState | null;
  journal: Journal | null;
  jobs: Partial<Record<JobId, JobState>>;
  /** Занятые группы задач: {sac: "telegram"} — запущено не из приложения, но тоже идёт. */
  jobGroups: Record<string, string>;
  rollbackAvailable: boolean;
  profiles: ProfileInfo[];
  backupProfile: string | null;
  setupInfo: SetupInfo | null;
  /** Окно справки открыто. */
  helpOpen: boolean;
  /** Окно уведомлений открыто и номер последнего просмотренного события. */
  notifOpen: boolean;
  paletteOpen: boolean;
  seenFeed: number;
  /** Открытый шаг мастера первого запуска; null — мастер закрыт. */
  setup: SetupStep | null;
  tab: Tab;
  /** Режим интерфейса на этом устройстве: «Простой» прячет детали, ничего не меняя в настройках. */
  ui: Ui;
  /** Выбранный узел конвейера во вкладке «Бот». */
  node: BotNode;
  /** Открыта ли панель настроек узла. Её открывает любой переход к узлу, в том числе из другой вкладки. */
  nodeOpen: boolean;
  /** Поиск параметра по Cmd/Ctrl+K: null — не запрошен, строка — что подставить. */
  paramSearch: string | null;
  /** Ключ, который открыть в «Настройки → Ключи и доступ» при переходе из другой вкладки. */
  keyJump: string | null;
  /** Раскрыть «Как рискуем → Настроить вручную» при переходе с плашки порога на «Цепочке». */
  riskManual: boolean;
  toast: { id: number; text: string; undo?: () => void } | null;

  boot(): Promise<void>;
  setTab(tab: Tab): void;
  setUi(ui: Ui): void;
  /** Из «Простого» в «Профи», когда человек идёт туда, чего в «Простом» нет. */
  needPro(): void;
  /** Открыть вкладку «Бот» на нужном узле конвейера с его панелью настроек. */
  openNode(node: BotNode): void;
  /** Открыть «Настройки» с панелью этого ключа. */
  openKey(service: string): void;
  closeNode(): void;
  openSetup(step: SetupStep | null): void;
  openHelp(open: boolean): void;
  openNotif(open: boolean): void;
  openPalette(open: boolean): void;
  /** Открыть «Настройки → Параметры» с фокусом в поиске. */
  searchParams(q?: string): void;
  clearParamSearch(): void;
  setPeriod(p: Period): void;
  say(text: string): void;
  /** Сообщение с кнопкой «Отменить»; живёт дольше обычного. */
  sayUndo(text: string, undo: () => void): void;
  apply(e: BotEvent): void;
  /** Выполнить действие: успех — тост ok, ошибка — тост с текстом сервера. */
  act(fn: () => Promise<unknown>, ok?: string): Promise<boolean>;

  loadState(): Promise<void>;
  loadMetrics(p?: Period): Promise<void>;
  loadExecution(p?: Period): Promise<void>;
  loadDaily(): Promise<void>;
  loadStrategies(): Promise<void>;
  loadConfig(): Promise<void>;
  loadAlerts(): Promise<void>;
  loadSymbols(): Promise<void>;
  loadSecrets(): Promise<void>;
  loadAi(): Promise<void>;
  loadJournal(p: Period): Promise<void>;
  loadSetup(): Promise<void>;
  loadJobs(): Promise<void>;

  requestCode(): Promise<boolean>;
  verifyCode(code: string): Promise<boolean>;
  /** Закрыть сессию на сервере и забыть токен на этом устройстве. */
  logout(): Promise<void>;
}

let feedSeq = 0;

let toastSeq = 0;

let unsubscribe: (() => void) | null = null;

let refreshTimer: ReturnType<typeof setTimeout> | undefined;

const MAX_FEED = 60;

const sideOf = (s: unknown): Side => (s === "sell" || s === "short" ? "short" : "long");

export type Ui = "simple" | "pro";

const UI_KEY = "bitbot.ui";

/** По умолчанию «Профи»: у тех, кто уже работал с приложением, ничего не меняется само. */
const savedUi = (): Ui => {
  try {
    return localStorage.getItem(UI_KEY) === "simple" ? "simple" : "pro";
  } catch {
    return "pro";
  }
};

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
      if (get().execution[get().period]) void get().loadExecution().catch(() => undefined);
      void get().loadDaily().catch(() => undefined);
      const j = get().journal;
      if (j) void get().loadJournal(j.period).catch(() => undefined);
    }, REFRESH_DEBOUNCE_MS);
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
      lastCycle: st.cycle ?? null,
      modeReadiness: st.mode_readiness ?? null,
      opsStatus: st.ops_status ?? null,
      trusted: st.trusted ?? null,
      alive: st.bot.alive,
      balance: st.balance,
      dayRef: st.day_ref ?? null,
      slots: st.slots,
      positions: positionsFrom(st.positions),
      market: st.market ?? null,
    });

  return {
    ready: false,
    needLogin: false,
    wsUp: false,
    alive: true,
    bot: null,
    lastCycle: null,
    modeReadiness: null,
    opsStatus: null,
    trusted: null,
    balance: 0,
    dayRef: null,
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
    execution: {},
    daily: [],
    strategies: [],
    schema: [],
    riskPresets: {},
    config: {},
    alerts: [],
    symbols: { forced: [], excluded: [] },
    secrets: null,
    secretsNote: "",
    ai: null,
    journal: null,
    jobs: {},
    jobGroups: {},
    rollbackAvailable: false,
    profiles: [],
    backupProfile: null,
    market: null,
    setupInfo: null,
    setup: null,
    helpOpen: false,
    notifOpen: false,
    paletteOpen: false,
    seenFeed: 0,
    paramSearch: null,
    keyJump: null,
    riskManual: false,
    tab: "terminal",
    ui: savedUi(),
    node: "scan",
    nodeOpen: false,
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
          const r = await api<{ token: string }>("POST", EP.authTelegram, {
            initData: platform.initData(),
          });
          platform.setToken(r.token);
        } catch (e) {
          get().say(e instanceof ApiError ? e.message : "Не удалось войти через Telegram.");
        }
      }
      // Ссылка лаунчера: обмениваем одноразовый токен на сессию (работает только на этом компьютере).
      const local = !isMock && platform.kind === "web" ? platform.takeLocalToken() : null;
      if (local) {
        try {
          const r = await api<{ token: string }>("POST", EP.authLocal, { token: local });
          platform.setToken(r.token);
        } catch (e) {
          get().say(e instanceof ApiError ? e.message : "Не удалось войти по ссылке запуска.");
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
        get().loadJobs(),
        api<PendingSignal[]>("GET", EP.signalsPending).then((pending) => set({ pending })),
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

    // Журнал живёт в нижней панели Терминала; старые ссылки на вкладку ведут туда.
    setTab: (tab) => set({ tab: tab === "journal" ? "terminal" : tab, setup: null, nodeOpen: false }),
    setUi: (ui) => {
      try {
        localStorage.setItem(UI_KEY, ui);
      } catch {
        /* нет хранилища — режим живёт до перезагрузки */
      }
      // «Обучения» в «Простом» нет — с него уходим на «Главную».
      set(ui === "simple" && (get().tab === "learn" || get().tab === "stats") ? { ui, tab: "terminal" } : { ui });
    },
    needPro: () => {
      if (get().ui !== "simple") return;
      get().setUi("pro");
      get().say("Открыто в режиме «Профи» — здесь все настройки");
    },
    // Узел цепочки есть только в «Профи»: из «Простого» переход открывает «Профи».
    openNode: (node) => {
      get().needPro();
      set({ tab: "bots", node, nodeOpen: true, setup: null });
    },
    openKey: (keyJump) => set({ tab: "settings", keyJump, nodeOpen: false, setup: null }),
    closeNode: () => set({ nodeOpen: false }),
    openSetup: (setup) => set({ setup }),
    openHelp: (helpOpen) => set({ helpOpen }),
    openPalette: (paletteOpen) => set({ paletteOpen }),
    openNotif: (notifOpen) => set((s) => (notifOpen ? { notifOpen, seenFeed: s.feed[0]?.id ?? s.seenFeed } : { notifOpen })),
    searchParams: (q = "") => {
      get().needPro();
      set({ tab: "settings", setup: null, paramSearch: q });
    },
    clearParamSearch: () => set({ paramSearch: null }),
    setPeriod(period) {
      set({ period });
      if (!get().metrics[period]) void get().loadMetrics(period).catch(() => undefined);
    },
    say: (text) => set({ toast: { id: ++toastSeq, text } }),
    sayUndo: (text, undo) => set({ toast: { id: ++toastSeq, text, undo } }),

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
            trace: d.trace,
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
        case "price": {
          // Живая цена с биржи (дашборд, WebSocket Bybit): только цена, уровни — из тиков бота.
          const cur = get().positions[d.symbol];
          if (cur && typeof d.price === "number") set({ positions: { ...get().positions, [d.symbol]: { ...cur, price: d.price } } });
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
        case "job.progress": {
          const id = d.job as JobId;
          set((s) => ({ jobs: { ...s.jobs, [id]: { running: true, info: d, done: undefined } } }));
          break;
        }
        case "job.done": {
          const id = d.job as JobId;
          set((s) => ({ jobs: { ...s.jobs, [id]: { running: false, info: {}, done: d as unknown as JobDone } } }));
          feed(d.ok ? "edit" : "guard", d.ok ? `Задача готова: ${JOB_TITLE[id] ?? id}` : `Задача не завершилась: ${JOB_TITLE[id] ?? id}`);
          if (id === "backtest") void get().loadStrategies().catch(() => undefined);
          void get().loadJobs().catch(() => undefined);
          break;
        }
        case "model.approved":
          feed("edit", "Модель SAC допущена к paper");
          break;
        case "model.rolled_back":
          void get().loadJobs().catch(() => undefined);
          feed("edit", "Модель SAC откатана к предыдущей");
          break;
        case "trusted.post": {
          const t = d as TrustedPost;
          set({ trusted: t });
          const what = t.symbol ? `${base(t.symbol)} ${t.side === "buy" ? "LONG" : "SHORT"}` : "пост";
          feed(t.ok ? "open" : "skip", t.ok ? `Канал @${t.channel}: ${what} — вход через ${t.latency_s} с` : `Канал @${t.channel}: ${what} пропущен — ${t.reason}`);
          break;
        }
        case "trace.cycle": {
          const c = d as unknown as CycleTrace;
          set({ lastCycle: c });
          feed("cycle", cycleLine(c));
          break;
        }
        case "cycle": {
          // Пауза могла прийти из другого окна — цикл бота знает о ней раньше снимка.
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
      applyState(await api<BotState>("GET", EP.state));
    },
    async loadMetrics(p) {
      const period = p ?? get().period;
      const m = await api<Metrics>("GET", EP.metrics(period));
      set((s) => ({ metrics: { ...s.metrics, [period]: m } }));
    },
    async loadExecution(p) {
      const period = p ?? get().period;
      const m = await api<ExecutionMetrics>("GET", EP.metricsExecution(period));
      set((s) => ({ execution: { ...s.execution, [period]: m } }));
    },
    async loadJobs() {
      const st = await api<JobsStatus>("GET", EP.jobs);
      set((s) => {
        const jobs = { ...s.jobs };
        for (const id of ["backtest", "train", "tune"] as const) {
          const info = st.progress[id];
          if (info) jobs[id] = { running: true, info, done: undefined };
        }
        return {
          jobs,
          jobGroups: st.running,
          rollbackAvailable: st.rollback_available,
          profiles: st.profiles ?? [],
          backupProfile: st.backup_profile ?? null,
        };
      });
    },
    async loadSetup() {
      set({ setupInfo: await api<SetupInfo>("GET", EP.setup) });
    },
    async loadJournal(period) {
      set({ journal: await api<Journal>("GET", EP.journal(period)) });
    },
    async loadDaily() {
      set({ daily: await api<DailyRow[]>("GET", EP.pnlDaily(35)) });
    },
    async loadStrategies() {
      set({ strategies: await api<Strategy[]>("GET", EP.strategies) });
    },
    async loadConfig() {
      const have = get().riskPresets;
      const [schema, config, riskPresets] = await Promise.all([
        get().schema.length ? get().schema : api<SettingDef[]>("GET", EP.configSchema),
        api<ConfigValues>("GET", EP.config),
        Object.keys(have).length ? have : api<Record<string, RiskPreset>>("GET", EP.riskPresets).catch(() => have),
      ]);
      set({ schema, config, riskPresets });
    },
    async loadAlerts() {
      set({ alerts: await api<Alert[]>("GET", EP.alerts) });
    },
    async loadSymbols() {
      set({ symbols: await api<Symbols>("GET", EP.symbols) });
    },
    async loadSecrets() {
      try {
        set({ secrets: await api<SecretItem[]>("GET", EP.secrets), secretsNote: "" });
      } catch (e) {
        set({ secrets: null, secretsNote: e instanceof ApiError ? e.message : "Ключи недоступны." });
      }
    },

    async loadAi() {
      set({ ai: await api<AiState>("GET", EP.ai) });
    },

    requestCode() {
      return get().act(() => api("POST", EP.authDesktopRequest), "Код отправлен вам в Telegram");
    },
    async logout() {
      await api("POST", EP.authLogout).catch(() => undefined);
      unsubscribe?.();
      unsubscribe = null;
      platform.setToken(null);
      set({ needLogin: true, wsUp: false });
    },
    async verifyCode(code) {
      try {
        const r = await api<{ token: string }>("POST", EP.authDesktopVerify, { code });
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

/** P&L позиции по последней цене. */
export function positionPnl(p: Position): { pnl: number; move: number } {
  const price = p.price ?? p.entry;
  const dir = p.side === "long" ? 1 : -1;
  return { pnl: (price - p.entry) * p.qty * dir, move: (price / p.entry - 1) * 100 * dir };
}
