// Вкладка «Бот» как конвейер решений: полотно с узлами слева направо и мини-цепочкой AI,
// под ним — сводка выбранного узла и его карточки (те же, что были во вкладке раньше).
// Итогов последнего цикла на узлах пока нет: бот не сохраняет шаги решения (этап 2).

import { useEffect, useRef, useState } from "react";

import type { BotNode, SecretItem, SettingDef } from "../lib/types";
import { actions } from "../store/actions";
import { positionPnl, useStore } from "../store";
import { ArrowsClockwise, ArrowsDownUp, Receipt, Clock, HandTap, Lifebuoy, Lightning, LockSimple, ShieldSlash, Target, CurrencyBtc, Fire, Key, Rocket, Brain, ChartBar, Coins, FlagCheckered, Handshake, MagnifyingGlass, ShieldCheck, Sparkle, WarningCircle } from "@phosphor-icons/react";
import { aiLogo } from "../components/brand";
import { plural, usd } from "../lib/format";
import { inHours } from "../lib/idle";

import { Tiles } from "../components/irow";
import { Q } from "../components/tooltip";
import { AiPane, aiLine } from "./ai";
import { KeyLine } from "./settings/keys";
import { TelegramPanel } from "./settings/tgpanel";
import { PRESET_META, applyPreset, ConfirmCard, RiskCard, StrategiesCard, TrackCard, WhatCard, cfg, levText, presetName, presetOf, profileName } from "./bots";
import { EmergencyCard } from "./emergency";
import { RulesCard } from "./rules";
import { UiSwitchRow } from "./uimode";
import { HoursCard, SymbolsCard } from "./settings/trading";
import { ForecastCard, PipeCanvas } from "./pipecanvas";
import { SignalsCard } from "./signals";
import { mark } from "./trace";
import { cycleKind } from "./terminal/radar";

export const NODES: readonly { id: BotNode; title: string; icon: JSX.Element }[] = [
  { id: "scan", title: "Сканер", icon: <MagnifyingGlass /> },
  { id: "strat", title: "Стратегии", icon: <ChartBar /> },
  { id: "ai", title: "AI", icon: <Sparkle /> },
  { id: "sac", title: "SAC", icon: <Brain /> },
  { id: "risk", title: "Риск", icon: <ShieldCheck /> },
  { id: "order", title: "Ордер", icon: <Receipt /> },
  { id: "track", title: "Выход", icon: <FlagCheckered /> },
];

const MODES = [
  ["local", "Стратегии", <ChartBar />],
  ["ai", "AI", <Sparkle />],
  ["dqn", "Нейросеть", <Brain />],
  ["hybrid", "AI + нейросеть", <Handshake />],
] as const;

/** Чего не хватает режиму, чтобы он заработал; "" — всё готово. Видно на плитке ещё до переключения. */
function useModeIssues(): Record<string, string> {
  const ai = useStore((s) => s.ai);
  const ready = useStore((s) => s.modeReadiness);
  const model = useStore((s) => s.profiles.find((p) => p.active)?.model_exists);
  const noKey = Boolean(ai && !ai.chain.length);
  // Все ключи цепочки в ошибке (не принят, нет денег, ломает ответ…) — AI фактически не работает.
  const keyed = (ai?.chain ?? []).map((c) => ai?.providers.find((p) => p.id === c.provider)?.status ?? "unknown");
  const aiDown = ready?.ai?.reason === "ai_unavailable" || (keyed.length > 0 && keyed.every((s) => !["ok", "unknown", "rate_limit"].includes(s)));
  const sac = model === false ? "модель не обучена" : ready?.dqn?.reason === "sac_not_approved" ? "модель не допущена" : "";
  const aiIssue = noKey ? "нужен ключ AI" : aiDown ? "AI не отвечает" : "";
  return { local: "", ai: aiIssue, dqn: sac, hybrid: aiIssue || sac };
}

/** «Кто принимает решение» — одно место для MODE; в «Простом» два понятных варианта. */
export function ModeSeg({ simple = false, onBlocked }: { simple?: boolean; onBlocked?: (mode: string) => void }) {
  const mode = useStore((s) => cfg<string>(s.config, "MODE", "ai"));
  const setTab = useStore((s) => s.setTab);
  const issues = useModeIssues();
  // Без обученной модели сервер не включит «Нейросеть» — не шлём запрос, а показываем, что сделать.
  const [blocked, setBlocked] = useState<{ v: string; n: number } | null>(null);
  const sacIssue = issues.dqn;
  const pick = (v: string) => {
    if ((v === "dqn" || v === "hybrid") && sacIssue) {
      setBlocked({ v, n: Date.now() });
      onBlocked?.(v);
      return;
    }
    setBlocked(null);
    void actions.patchConfig({ MODE: v });
  };
  const own = simple && mode !== "local" && mode !== "ai";
  const opts = (simple ? MODES.slice(0, 2) : MODES).map(([v, name, icon]) => [v, name, icon, issues[v] || undefined] as const);
  return (
    <>
      <div className="mode-tiles issues">
        <Tiles value={own ? null : mode} options={opts} onChange={pick} label="Кто принимает решение" shake={blocked} />
      </div>
      {blocked && (
        <div className="pipe-fix" key={blocked.n} role="alert">
          <WarningCircle weight="fill" />
          <span>
            «{MODES.find(([m]) => m === blocked.v)?.[1]}» не включится: {sacIssue}.
          </span>
          <button type="button" className="pipe-fix-go" onClick={() => setTab("learn")}>
            <Brain />
            Обучить модель →
          </button>
        </div>
      )}
      {own && <p className="note-soon">Сейчас своё, из «Профи»: {MODES.find(([m]) => m === mode)?.[1] ?? mode}. Выберите вариант выше, чтобы сменить.</p>}
    </>
  );
}


/** Участвует ли узел в решении при этом MODE (как в SignalCombiner): local — без AI и SAC, ai — без SAC, dqn — без AI. */
export function takesPart(mode: string, node: BotNode): boolean {
  if (node === "ai") return mode === "ai" || mode === "hybrid";
  if (node === "sac") return mode === "dqn" || mode === "hybrid";
  return true;
}

/** Значение параметра так же, как его показывает «Все параметры»: проценты, вкл/выкл, подписи вариантов, единицы. */
export function formatSetting(d: SettingDef | undefined, v: unknown): string {
  if (v === undefined || v === null || v === "") return "—";
  if (!d) return String(v);
  if (d.type === "bool") return v ? "вкл" : "выкл";
  if (d.type === "pct" && typeof v === "number") return `${+(v * 100).toFixed(2)}%`;
  if (d.type === "choice") return d.options.find((o) => o.value === String(v))?.label ?? String(v);
  return d.unit ? `${String(v)} ${d.unit}` : String(v);
}

/** Состояние ключа подключения (Bybit, Telegram и т. д.) — как в Настройки → Подключения. */
export function keyState(it?: SecretItem): string {
  if (!it) return "—";
  const last4 = it.fields.find((f) => f.secret && f.last4)?.last4;
  const tail = last4 ? ` · ••••${last4}` : "";
  if (it.status === "ok") return `▲ работает${tail}`;
  if (it.status === "bad") return `▼ не работает${tail}`;
  if (it.status === "warn") return `есть проблема${tail}`;
  return "не задан";
}

const pct = (v: number) => `${Math.round(v * 100)}%`;
const day = (iso: string) => new Date(iso).toLocaleDateString("ru-RU", { day: "numeric", month: "short" });

const PROFILE_ICON: Record<string, JSX.Element> = { bluechip: <CurrencyBtc />, altcoin: <Rocket />, meme: <Fire /> };

/** «15m» → «15 мин», «1h» → «1 ч», «1d» → «1 д». */
export const tfText = (tf: string): string => {
  const m = /^(\d+)([mhdw])$/.exec(tf);
  return m ? `${m[1]} ${({ m: "мин", h: "ч", d: "д", w: "нед" } as Record<string, string>)[m[2]]}` : tf;
};

/** Свеча — своего значка в библиотеке нет: две свечи с фитилями. */
const Candle = () => (
  <svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" aria-hidden="true">
    <path d="M5 1.5v2.5M5 11v3.5M11 3v3M11 12v2.5" />
    <rect x="3" y="4" width="4" height="7" rx="1" fill="currentColor" fillOpacity=".25" />
    <rect x="9" y="6" width="4" height="6" rx="1" />
  </svg>
);

const HOURS_NAME: Record<string, string> = { "1-10": "Азия", "8-20": "Европа", "14-22": "США", "22-6": "Ночь" };

/** Карточка шага: живая цифра с единицей, подпись с настройками, значок режима в углу. */
interface Metric {
  v: string;
  u?: string;
  sub: string;
  /** Часы торговли на карточке Сканера: подпись и торгует ли сейчас. */
  hrs?: { text: string; on: boolean };
  /** Частичная фиксация прибыли — значком на карточке «Выход». */
  ptp?: boolean;
  /** Запасной путь — значком на карточке «Стратегии» (в режиме «Стратегии» не нужен). */
  fb?: boolean;
  /** «монет» значком стопки монет — в паре со свечой на карточке Сканера. */
  coins?: string;
  /** Таймфрейм свечи на карточке Сканера: «15 мин», «1 ч». */
  tf?: string;
  /** Профиль монет в работе — значком на карточке Сканера; auto — бот меняет его сам по сезону. */
  prof?: string;
  auto?: boolean;
  tone?: "up" | "down";
  badge?: string;
}

interface Info {
  desc: string;
  fields: [string, string][];
}

/** Строка состояния под узлом и сводка полей для каждого узла — только из того, что бот уже отдаёт. */
function usePipeline(mode: string): {
  status: Record<BotNode, string>;
  info: Record<BotNode, Info>;
  need: Partial<Record<BotNode, string>>;
  metric: Record<BotNode, Metric>;
} {
  const c = useStore((s) => s.config);
  const presets = useStore((s) => s.riskPresets);
  const schema = useStore((s) => s.schema);
  const bot = useStore((s) => s.bot);
  const strategies = useStore((s) => s.strategies);
  const symbols = useStore((s) => s.symbols);
  const ai = useStore((s) => s.ai);
  const profiles = useStore((s) => s.profiles);
  const rollback = useStore((s) => s.rollbackAvailable);
  const open = useStore((s) => Object.keys(s.positions).length);
  const secrets = useStore((s) => s.secrets);
  const bybit = secrets?.find((s) => s.service === "bybit");
  const tg = secrets?.find((s) => s.service === "telegram");
  const feeds = (secrets ?? []).filter((s) => s.group === "Данные и соцсети");
  const show = (key: string) => {
    const d = schema.find((x) => x.key === key);
    return formatSetting(d, c[key]?.value ?? d?.default);
  };
  const paper = bot?.paper ?? true;
  const spot = cfg<string>(c, "MARKET_TYPE", "linear") === "spot";
  const hours = cfg<string>(c, "TRADING_HOURS", "");
  const fallback = cfg<boolean>(c, "LOCAL_FALLBACK", false);
  const auto = cfg(c, "AUTO_EXECUTE", false);
  const partial = cfg(c, "PARTIAL_TP_ENABLED", true);
  const core = strategies.filter((s) => !s.experimental);
  const exp = strategies.filter((s) => s.experimental);
  const tested = strategies.filter((s) => s.backtest && s.backtest.trades > 0);
  const edge = tested.filter((s) => s.backtest!.trades >= 100 && s.backtest!.sharpe > 0);
  const name = (id: string) => ai?.providers.find((p) => p.id === id)?.name ?? id;
  const active = profiles.find((p) => p.active);
  const guardsOff = paper && cfg<boolean>(c, "PAPER_GUARDS_OFF", false);
  // Что мешает блоку работать — плашка прямо на схеме, клик открывает его панель.
  const issue = useModeIssues();
  const need: Partial<Record<BotNode, string>> = {
    ai: takesPart(mode, "ai") ? issue.ai || undefined : undefined,
    sac: takesPart(mode, "sac") ? issue.dqn || undefined : undefined,
    // Ключ Bybit показывает блок «Биржа»; здесь — только то, что мешает самому ордеру в Live.
    order: !paper && (!bybit || bybit.status === "unset" || bybit.status === "bad") ? "Live: нужен рабочий ключ Bybit" : !auto && !paper && tg?.status !== "ok" ? "нужен Telegram для подтверждений" : undefined,
  };
  const status: Record<BotNode, string> = {
    scan: `${bot?.profile ? profileName(bot.profile) : "профиль —"} · ${spot ? "спот" : "фьючерсы"}`,
    // Роль стратегий зависит от режима: решают сами только в Local, иначе — подсказка для AI или запасной путь.
    strat:
      mode === "local"
        ? `решают · включено ${strategies.filter((s) => s.enabled).length}`
        : mode === "dqn"
          ? fallback
            ? `запасной путь · ${strategies.filter((s) => s.enabled).length}`
            : "не участвуют"
          : `подсказка для AI · ${strategies.filter((s) => s.enabled).length}`,
    ai: aiLine(ai),
    sac: mode === "dqn" ? "решает сам" : "вместе с AI",
    // Главные цифры риска прямо на холсте: пресет, риск на сделку, позиций сразу, плечо.
    risk: `${presetName(presetOf(c, presets))} · ${show("RISK_PER_TRADE")} · ${cfg(c, "MAX_POSITIONS", 3)} поз. · плечо ${levText(c, spot)}${guardsOff ? " · без защит" : ""}`,
    order: `Bybit ${bybit?.status === "ok" ? "▲" : bybit?.status === "bad" ? "▼" : "·"} ${paper ? "Paper" : "Live"} · ${auto ? "авто" : "с подтверждением"}`,
    track: partial ? `${pct(cfg(c, "PARTIAL_TP_FRACTION", 0.5))} на ${pct(cfg(c, "PARTIAL_TP_TRIGGER", 0.6))} · безубыток` : "без частичной фиксации",
  };
  const info: Record<BotNode, Info> = {
    scan: {
      desc: "Решает, где искать вход: рынок, профиль монет, сколько монет сканировать и в какие часы торговать.",
      fields: [
        ["Профиль рынка", bot?.profile ? profileName(bot.profile) : "—"],
        ["Рынок", spot ? "спот" : "фьючерсы"],
        ["Таймфрейм", show("TIMEFRAME")],
        ["Монет в скане", show("SCAN_TOP_N")],
        ["Часы торговли", hours ? `${hours} UTC` : "24/7"],
        ["Монеты вручную", `${symbols.forced.length} всегда · ${symbols.excluded.length} никогда`],
        ["Смена сезона", cfg<string>(c, "SEASON_SWITCH_MODE", "alert") === "auto" ? "сам меняет профиль" : "предлагает в Telegram"],
        ["Данные и соцсети", feeds.length ? `${feeds.filter((s) => s.status === "ok").length} из ${feeds.length} подключено` : "—"],
      ],
    },
    strat: {
      desc:
        mode === "local"
          ? "Решают сами: сделка открывается по сигналу включённых стратегий."
          : mode === "dqn"
            ? "Сделку решает SAC; стратегии работают, только пока модель не загружена и включён запасной путь."
            : "Сделку решает AI; включённые стратегии он видит как подсказку о типах сетапов.",
      fields: [
        ["Основные", `${core.filter((s) => s.enabled).length} из ${core.length}`],
        ["Экспериментальные", `${exp.filter((s) => s.enabled).length} из ${exp.length}`],
        ["Запасной путь", mode === "local" ? "не нужен в режиме «Стратегии»" : fallback ? "вкл" : "выкл"],
        ["С бэктестом", String(tested.length)],
        ["С плюсом в бэктесте", String(edge.length)],
      ],
    },
    ai: {
      desc: "Подтверждает или отклоняет сделку. Провайдеры спрашиваются по очереди — сначала бесплатные, потом платные: ошибка или лимит — сразу следующий.",
      fields: !ai
        ? [["Цепочка", "загружается"]]
        : [
            ["Основной", ai.chain[0] ? `${name(ai.chain[0].provider)} · ${ai.chain[0].model}` : "нет ключа"],
            ["В цепочке", String(ai.chain.length)],
            ["Последним ответил", ai.last ? name(ai.last) : "—"],
            ["Если все молчат", fallback ? "решают стратегии" : "сделки нет"],
          ],
    },
    sac: {
      desc:
        mode === "dqn"
          ? "Решает сам: обученная нейросеть открывает сделки."
          : mode === "hybrid"
            ? "Сделка открывается, только когда SAC и AI согласны."
            : "В этом режиме не участвует. Обучить можно заранее — модель пригодится в режимах «Нейросеть» и «AI + нейросеть».",
      fields: [
        ["Роль", mode === "dqn" ? "решает сам" : mode === "hybrid" ? "вместе с AI" : "не участвует"],
        ["Профиль", active?.label ?? "—"],
        ["Таймфрейм профиля", active?.timeframe ?? "—"],
        ["Модель", !active ? "—" : !active.model_exists ? "не обучена" : active.trained_at ? `обучена ${day(active.trained_at)}` : "есть"],
        ["Резервная копия", rollback ? "есть · откат одной кнопкой" : "нет"],
      ],
    },
    risk: {
      desc: "Последний фильтр перед ордером: размер позиции, плечо, лимиты и защиты портфеля.",
      // Без таблицы: всё это сразу ниже — в «Как рискуем» и «Правилах защиты», там же и меняется.
      fields: [],
    },
    order: {
      desc: "Отправляет заявку на биржу. В Paper сделки виртуальные, но по реальным ценам.",
      fields: [
        ["Биржа Bybit", keyState(bybit)],
        ["Рыночные данные", bybit?.status === "ok" ? "по вашему ключу" : paper ? "публичные — для Paper ключ не нужен" : "нужен рабочий ключ"],
        ["Telegram-бот", keyState(tg)],
        ["Как открывать", auto ? "автоматически" : paper ? "с подтверждением (в Paper не спрашивает)" : "с подтверждением"],
        ["Время на решение", `${cfg<number>(c, "TELEGRAM_CONFIRM_TIMEOUT", 60)} с`],
        ["Если не ответил", cfg<string>(c, "CONFIRM_ON_TIMEOUT", "execute") === "execute" ? "открыть сделку" : "пропустить"],
        ["Счёт", paper ? "Paper" : "Live"],
        ["Тестовая сеть Bybit", show("TESTNET")],
      ],
    },
    track: {
      desc: "Ведёт открытую позицию до выхода: частичная фиксация прибыли, стоп в безубыток, трейлинг-стоп.",
      fields: [
        ["Открыто позиций", String(open)],
        ["Частичная фиксация", show("PARTIAL_TP_ENABLED")],
        ["Когда", partial ? `${show("PARTIAL_TP_TRIGGER")} пути до TP` : "—"],
        ["Сколько закрыть", partial ? show("PARTIAL_TP_FRACTION") : "—"],
        ["После фиксации", partial ? "стоп в безубыток" : "—"],
        ["Отступ трейлинга", show("TRAILING_STOP_ATR_MULT")],
      ],
    },
  };
  // Главная цифра — что блок делает сейчас; настройки — в подписи, подробности — в панели шага.
  const fun = useStore((s) => s.lastCycle?.funnel);
  const slots = useStore((s) => s.slots);
  const pnl = useStore((s) => Object.values(s.positions).reduce((a, p) => a + positionPnl(p).pnl, 0));
  const on = strategies.filter((s) => s.enabled).length;
  const nScan = fun?.scanned ?? Number(cfg(c, "SCAN_TOP_N", 20));
  const metric: Record<BotNode, Metric> = {
    scan: {
      v: String(nScan),
      sub: "",
      coins: plural(nScan, ["монета", "монеты", "монет"]),
      tf: tfText(cfg<string>(c, "TIMEFRAME", "15m")),
      prof: bot?.profile ?? undefined,
      auto: cfg<string>(c, "SEASON_SWITCH_MODE", "alert") === "auto",
      hrs: { text: hours ? (HOURS_NAME[hours] ?? `${hours} UTC`) : "24/7", on: !hours || inHours(hours, new Date().getUTCHours()) },
    },
    strat: {
      v: String(on),
      u: "включено",
      fb: mode === "local" ? undefined : fallback,
      sub: mode === "local" ? "решают сделку" : mode === "dqn" ? (fallback ? "запасной путь" : "не участвуют") : fallback ? "подсказка · запасной путь" : "подсказка для AI",
    },
    ai: { v: "", sub: "" },
    sac: { v: "", sub: "" },
    risk: {
      v: `${slots.used}/${slots.max}`,
      u: "слотов",
      sub: `${show("RISK_PER_TRADE")} · ${levText(c, spot)}`,
    },
    order: {
      v: String(fun?.opened ?? 0),
      u: "за цикл",
      sub: auto ? "открывает сам" : "с подтверждением",
    },
    track: {
      v: open ? usd(pnl) : "—",
      sub: `${open} ${plural(open, ["позиция", "позиции", "позиций"])}`,
      tone: !open || Math.abs(pnl) < 0.005 ? undefined : pnl > 0 ? "up" : "down",
      ptp: partial,
    },
  };
  return { status, info, need, metric };
}

/** Кто решает и какая роль у стратегий — одной фразой под переключателем (как в backend/bitbot/workers/bot.py). */
const modeNote = (mode: string, fallback: boolean): string =>
  mode === "local"
    ? "Решают стратегии: входит самый уверенный сигнал. AI не участвует."
    : mode === "dqn"
      ? `Решает нейросеть SAC.${fallback ? " Без модели — стратегии." : ""}`
      : `Решает ${mode === "hybrid" ? "пара SAC и AI — нужно согласие обоих" : "AI"}, стратегии — подсказка.${fallback ? " Без AI — стратегии." : ""}`;

/** Часы цикла: когда был последний и сколько до следующего; полоска заполняется к следующему циклу. */
function CycleClock({ at }: { at?: number }) {
  const every = useStore((s) => cfg<number>(s.config, "TRADING_INTERVAL", 900));
  const kind = useStore(cycleKind);
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 5000);
    return () => clearInterval(t);
  }, []);
  if (!at) return <p className="cyc-clock">Жду первого цикла…</p>;
  const left = Math.max(0, at + every - now / 1000);
  const p = Math.min(100, ((every - left) / every) * 100);
  const live = kind === "scan" || kind === "monitor";
  return (
    <div className={`cyc-clock${live ? " live" : ""}`} aria-label="Цикл бота">
      <span className={`dot ${live ? "ok live" : "warn"}`} />
      <span>
        цикл <b>{new Date(at * 1000).toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" })}</b>
        {live ? (left > 30 ? ` · следующий через ${Math.ceil(left / 60)} мин` : " · следующий вот-вот") : " · бот не ищет сделки"}
      </span>
      <i style={{ width: `${live ? p : 0}%` }} />
    </div>
  );
}

/** Рынок — над схемой рядом с «кто решает»: от него зависят стратегии, плечо и ордера. */
function MarketSeg({ onSwitch }: { onSwitch: (spot: boolean) => void }) {
  const market = useStore((s) => cfg<string>(s.config, "MARKET_TYPE", "linear"));
  return (
    <div className="mode-tiles">
      <Tiles
        value={market}
        options={[
          ["linear", "Фьючерсы", <ArrowsDownUp />, "лонг и шорт"],
          ["spot", "Спот", <Coins />, "только покупка"],
        ]}
        label="Рынок"
        onChange={(v) => {
          void actions.patchConfig({ MARKET_TYPE: v });
          onSwitch(v === "spot");
        }}
      />
    </div>
  );
}

export function Pipeline() {
  const node = useStore((s) => s.node);
  const openNode = useStore((s) => s.openNode);
  const setTab = useStore((s) => s.setTab);
  const config = useStore((s) => s.config);
  const ai = useStore((s) => s.ai);
  const strategies = useStore((s) => s.strategies);
  const openN = useStore((s) => Object.keys(s.positions).length);
  const presets = useStore((s) => s.riskPresets);
  const schema = useStore((s) => s.schema);
  const riskNow = presetOf(config, presets);
  const autoNow = cfg<boolean>(config, "AUTO_EXECUTE", false);
  const paperNow = useStore((s) => s.bot?.paper ?? true);
  const guardsOff = cfg<boolean>(config, "PAPER_GUARDS_OFF", false);
  const mode = cfg<string>(config, "MODE", "ai");
  const fallback = cfg<boolean>(config, "LOCAL_FALLBACK", false);
  const { info, need, metric } = usePipeline(mode);
  const at = NODES.findIndex((n) => n.id === node);
  const cur = NODES[at] ?? NODES[0];
  const part = takesPart(mode, cur.id);
  // Блоки, которым чего-то не хватает в выбранном режиме: подсвечиваются при переключении и ведут в свою панель.
  const issues = NODES.filter((n) => takesPart(mode, n.id) && need[n.id]);
  // Что сломано — для проводов: к сломанному узлу ток не идёт. Биржа без ключа в Paper работает на публичных ценах — «с оговоркой».
  const bybitOk = useStore((s) => s.secrets?.find((x) => x.service === "bybit")?.status === "ok");
  const paperNow0 = useStore((s) => s.bot?.paper ?? true);
  const trouble: Partial<Record<string, "bad" | "warn">> = {
    exchange: bybitOk ? undefined : paperNow0 ? "warn" : "bad",
    decide: need.ai || need.sac ? "bad" : undefined,
  };
  for (const n of issues) if (n.id !== "ai" && n.id !== "sac") trouble[n.id] = "bad";
  // Нажали режим, который не включается: дорожка «Нейросеть» вспыхивает, даже если сейчас не участвует.
  const [flash, setFlash] = useState(0);
  // Режим сменился — подсказка про невключённый режим больше не нужна.
  useEffect(() => setFlash(0), [mode]);
  const sacNeed = useModeIssues().dqn;
  // Сменили рынок: блоки, которые от него зависят, коротко подсвечиваются и говорят, что поменялось.
  const [chg, setChg] = useState<{ k: number; chips: Partial<Record<BotNode, string>> } | null>(null);
  const switchMarket = (spot: boolean) => {
    const fut = strategies.filter((s) => s.enabled && s.markets === "futures").length;
    const k = Date.now();
    setChg({
      k,
      chips: {
        strat: fut ? `${spot ? "−" : "+"}${fut} ${plural(fut, ["стратегия", "стратегии", "стратегий"])}` : "те же стратегии",
        risk: spot ? "плечо 1×" : `плечо ${levText(config, false)}`,
        order: spot ? "без шортов" : "лонг и шорт",
        track: openN ? `${openN} ${plural(openN, ["позицию", "позиции", "позиций"])} доведёт` : undefined,
      },
    });
    setTimeout(() => setChg((x) => (x?.k === k ? null : x)), 5000);
  };
  // Итог последнего цикла (src/decision_trace.py); null — бот ещё не прошёл цикл после обновления.
  const cycle = useStore((s) => s.lastCycle);
  const when = cycle ? new Date(cycle.at * 1000).toLocaleTimeString("ru-RU") : "";
  const curRes = cycle?.nodes[cur.id];
  const [curGlyph, curCls] = mark(curRes?.ok);
  // Настройки узла выезжают панелью справа, полотно остаётся на виду. Переход к узлу из поиска (⌘K) тоже открывает.
  const open = useStore((s) => s.nodeOpen);
  const closeNode = useStore((s) => s.closeNode);
  const closeBtn = useRef<HTMLButtonElement>(null);
  const toLearn = () => setTab("learn");
  // Каналы Telegram — своя панель справа, как у шагов.
  const [chan, setChan] = useState(false);
  const [data, setData] = useState(false);
  useEffect(() => {
    if (!data) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setData(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [data]);
  useEffect(() => {
    if (!chan) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setChan(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [chan]);
  useEffect(() => {
    if (!open) return;
    closeBtn.current?.focus({ preventScroll: true });
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && closeNode();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  // Блок шага: иконка, название, точка статуса; главная цифра; итог цикла плашкой. Подробности — в панели.
  const box = (id: BotNode) => {
    const n = NODES.find((x) => x.id === id)!;
    const on = !(mode === "dqn" && id === "strat" && !fallback);
    const problem = on ? need[id] : undefined;
    const res = on ? cycle?.nodes[id] : undefined;
    const [glyph, cls] = mark(res?.ok);
    const mt = metric[id];
    const chip = chg?.chips[id];
    return (
      <div
        // Новый ключ при смене режима или рынка — анимация проигрывается заново.
        key={`${id}${problem ? mode : ""}${chip ? chg!.k : ""}`}
        data-node={id}
        className={`ch-node${node === id ? " sel" : ""}${on ? "" : " idle"}${problem ? " alarm" : ""}${chip ? " changed" : ""}`}
      >
      <button type="button" className="ch-node-open" aria-pressed={node === id} aria-haspopup="dialog" aria-label={`${n.title}: настройки`} onClick={() => openNode(id)}>
        <span className="ch-h">
          <span className="ch-ico">
            {/* У «Ордера» — логотип биржи: сразу видно, куда уходят сделки. */}
            {n.icon}
            <i className={`dot ${problem ? "bad" : on ? "ok" : ""}`} />
          </span>
          <b>{n.title}</b>
        </span>
        {problem ? (
          <span className="need">
            {problem.includes("ключ") && <Key weight="bold" />}
            {problem}
          </span>
        ) : (
          <>
            <span className={`ch-v${mt.tone ? ` ${mt.tone}` : ""}`}>
              <b>{mt.v}</b>
              {mt.u && <small>{mt.u}</small>}
              {mt.coins && (
                <i className="ch-coins" title={`${mt.v} ${mt.coins} — столько бот сканирует каждый цикл`}>
                  <Coins />
                </i>
              )}
              {/* Профиль и часы торговли — значками справа; что именно — в подсказке. */}
              {mt.ptp !== undefined && (
                <span className="ch-icons">
                  <i
                    className={`ch-mini ptp${mt.ptp ? " on" : ""}`}
                    title={mt.ptp ? "Частичная фиксация включена: часть позиции закрывается по пути к тейк-профиту, стоп — в безубыток" : "Частичная фиксация выключена"}
                  >
                    <Target />
                  </i>
                </span>
              )}
              {mt.fb !== undefined && (
                <span className="ch-icons">
                  <i
                    className={`ch-mini ptp${mt.fb ? " on" : ""}`}
                    title={mt.fb ? "Запасной путь включён: если AI или нейросеть молчат, бот торгует по стратегиям" : "Запасной путь выключен: если AI или нейросеть молчат, новых сделок нет"}
                  >
                    <Lifebuoy />
                  </i>
                </span>
              )}
              {(mt.prof || mt.hrs) && (
                <span className="ch-icons">
                  {mt.prof && (
                    <i
                      className={`ch-mini p-${mt.prof}`}
                      title={`Профиль: ${profileName(mt.prof)}${mt.auto ? " — бот сам меняет его по сезону" : ""}`}
                    >
                      {PROFILE_ICON[mt.prof] ?? <ChartBar />}
                      {mt.auto && <ArrowsClockwise className="ch-auto" />}
                    </i>
                  )}
                  {mt.hrs && (
                    <i className={`ch-mini hrs${mt.hrs.on ? "" : " off"}`} title={`Часы торговли: ${mt.hrs.text}${mt.hrs.on ? " — сейчас торгует" : " — сейчас пауза, новых сделок нет"}`}>
                      <Clock />
                    </i>
                  )}
                </span>
              )}
            </span>
            <span className="ch-u">
              {mt.badge && <em className={`ch-badge ${mt.badge}`}>{mt.badge}</em>}
              {mt.sub}
              {mt.tf && (
                <em className="ch-tf" title="Таймфрейм: длина одной свечи, по которой бот считает сигналы">
                  <Candle />
                  {mt.tf}
                </em>
              )}
            </span>
          </>
        )}
        {chip ? (
          <span className="ch-chg">{chip}</span>
        ) : (
          res && (
            // Коротко — первая часть итога; целиком — в подсказке и в панели шага.
            <span className={`ch-r ${cls}`} title={res.text}>
              {glyph} {res.text.split(" · ")[0]}
            </span>
          )
        )}
      </button>
      {/* Ордер: как открывать сделку — сам или с подтверждением; клик переключает сразу. */}
      {id === "order" && on && (
        <span className="ch-quick" role="group" aria-label="Как открывать сделки">
          <button type="button" className="ch-q q-auto" aria-pressed={autoNow} title="Автоматически: бот нашёл сигнал и сразу открыл сделку" onClick={() => !autoNow && void actions.patchConfig({ AUTO_EXECUTE: true })}>
            <Lightning />
          </button>
          <button type="button" className="ch-q q-confirm" aria-pressed={!autoNow} title="С подтверждением: каждая сделка приходит вам в Telegram и сюда" onClick={() => autoNow && void actions.patchConfig({ AUTO_EXECUTE: false })}>
            <HandTap />
          </button>
        </span>
      )}
      {/* Риск: три пресета значками — выбранный горит своим цветом, клик переключает сразу. */}
      {id === "risk" && on && (
        <span className="ch-quick" role="group" aria-label="Профиль риска">
          {Object.entries(PRESET_META).map(([k, m]) => (
            <button
              key={k}
              type="button"
              className={`ch-q q-${k}`}
              aria-pressed={k === riskNow}
              title={`${m.name}: ${m.d}`}
              onClick={() => k !== riskNow && applyPreset(k, presets, schema)}
            >
              {m.icon}
            </button>
          ))}
          {/* Paper без защит — для быстрого сбора опыта; в Live бот всегда с защитами (проверяет сервер). */}
          <button
            type="button"
            className="ch-q q-guards"
            aria-pressed={paperNow && guardsOff}
            disabled={!paperNow}
            title={
              !paperNow
                ? "В Live защиты всегда включены — этот режим только для Paper"
                : guardsOff
                  ? "Paper без защит: бот входит чаще, чтобы быстрее набрать опыт. При переходе на Live защиты включатся сами. Нажмите, чтобы вернуть защиты."
                  : "Снять защиты в Paper: бот будет входить чаще, чтобы быстрее набрать опыт для обучения. В Live не действует."
            }
            onClick={() => void actions.patchConfig({ PAPER_GUARDS_OFF: !guardsOff })}
          >
            {paperNow ? <ShieldSlash /> : <LockSimple />}
          </button>
        </span>
      )}
      </div>
    );
  };

  // Дорожка в рамке «Решение»: AI (с логотипом провайдера) или Нейросеть SAC.
  const lane = (id: "ai" | "sac") => {
    const n = NODES.find((x) => x.id === id)!;
    const on = takesPart(mode, id);
    const problem = on ? need[id] : id === "sac" && flash ? sacNeed : undefined;
    const res = on ? cycle?.nodes[id] : undefined;
    const [glyph, cls] = mark(res?.ok);
    // Логотип того, кто ответил последним; если его нет — основного в цепочке. Вся очередь — в подсказке.
    const last = id === "ai" && ai?.last && aiLogo(ai.last, 20) ? ai.last : "";
    const prov = id === "ai" ? last || ai?.chain[0]?.provider || "" : "";
    const logo = prov ? aiLogo(prov, 20) : null;
    const queue = id === "ai" ? (ai?.chain ?? []).map((c) => ai?.providers.find((p) => p.id === c.provider)?.name ?? c.provider).join(" → ") : "";
    const sub = problem ?? (!on ? "не участвует" : id === "ai" ? aiLine(ai) : mode === "dqn" ? "решает сама" : "вместе с AI");
    return (
      <button
        key={`${id}${problem ? `${mode}${id === "sac" ? flash : ""}` : ""}`}
        type="button"
        className={`ch-lane${on ? " on" : ""}${problem ? " alarm" : ""}${node === id ? " sel" : ""}`}
        title={queue ? `Очередь провайдеров: ${queue}` : undefined}
        aria-pressed={node === id}
        aria-haspopup="dialog"
        onClick={() => openNode(id)}
      >
        <span className="ch-ico">{logo ?? n.icon}</span>
        <span className="ch-lt">
          <b>
            {id === "ai" ? "AI" : "Нейросеть"}
            {/* «Думает»: три точки, пока дорожка участвует и бот работает; в момент решения — быстрее и ярче. */}
            {on && !problem && (
              <i className="ch-think" aria-hidden="true">
                <i />
                <i />
                <i />
              </i>
            )}
          </b>
          <small>{sub}</small>
        </span>
        {res && (
          <span className={`ch-mk ${cls}`} title={res.text}>
            {glyph}
          </span>
        )}
      </button>
    );
  };

  return (
    <section className="view enter" aria-label="Стратегия">
      <UiSwitchRow />
      <div className="pipe-canvas">
        <div className="pipe-top">
          <div className="pipe-mode">
            <span className="label">
              Кто принимает решение{" "}
              <Q tip="Стратегии — только индикаторы. AI — языковая модель. Нейросеть — обученная модель SAC. AI + нейросеть — оба должны согласиться. Клик по блоку — его настройки справа. Числа на проводах — сколько монет прошло в последнем цикле; ромб перед «Риском» — порог уверенности." />
            </span>
            <ModeSeg onBlocked={() => setFlash(Date.now())} />
          </div>
          <div className="pipe-mode">
            <span className="label">
              Рынок <Q tip="Спот — только покупки, без плеча. Фьючерсы — лонг и шорт с плечом. Меняется сразу; открытые позиции бот доводит до конца на их рынке." />
            </span>
            <MarketSeg onSwitch={switchMarket} />
          </div>
          <div className="pipe-sub">
            <p className="pipe-mode-note">{modeNote(mode, fallback)}</p>
            <CycleClock at={cycle?.at} />
          </div>
        </div>
        {issues.length > 0 && !flash && (
          <div className="pipe-fix" key={mode} role="status">
            <WarningCircle weight="fill" />
            <span>Чтобы режим заработал:</span>
            {issues.map((n) => (
              <button key={n.id} type="button" className="pipe-fix-go" onClick={() => openNode(n.id)}>
                {n.icon}
                {need[n.id]!.startsWith(n.title) ? need[n.id] : `${n.title}: ${need[n.id]}`} →
              </button>
            ))}
          </div>
        )}
        <PipeCanvas mode={mode} cycle={cycle} onChannels={() => setChan(true)} onData={() => setData(true)} trouble={trouble} node={box} lane={lane} />
      </div>
      <button type="button" className={`pipe-scrim${data ? " open" : ""}`} tabIndex={-1} aria-hidden="true" onClick={() => setData(false)} />
      <aside className={`pipe-panel${data ? " open" : ""}`} role="dialog" aria-label="Данные" aria-hidden={!data}>
        <div className="pipe-panel-h">
          <b>Данные: новости, соцсети, ончейн</b>
          <button type="button" className="btn pipe-close" aria-label="Закрыть данные" onClick={() => setData(false)}>
            ✕
          </button>
        </div>
        <div className="pipe-body">{data && <SignalsCard />}</div>
      </aside>
      <button type="button" className={`pipe-scrim${chan ? " open" : ""}`} tabIndex={-1} aria-hidden="true" onClick={() => setChan(false)} />
      <aside className={`pipe-panel${chan ? " open" : ""}`} role="dialog" aria-label="Telegram" aria-hidden={!chan}>
        <div className="pipe-panel-h">
          <b>Telegram</b>
          <button type="button" className="btn pipe-close" aria-label="Закрыть Telegram" onClick={() => setChan(false)}>
            ✕
          </button>
        </div>
        <div className="pipe-body">
          {chan && <TelegramPanel />}
        </div>
      </aside>
      <button type="button" className={`pipe-scrim${open ? " open" : ""}`} tabIndex={-1} aria-hidden="true" onClick={closeNode} />
      <aside className={`pipe-panel${open ? " open" : ""}`} role="dialog" aria-label={`Шаг «${cur.title}»`} aria-hidden={!open}>
        <div className="pipe-panel-h">
          <span className="pipe-sum-ico">{cur.icon}</span>
          <b>{cur.title}</b>
          <Q tip={info[cur.id].desc} />
          {!part && <span className="pill">не участвует</span>}
          <button ref={closeBtn} type="button" className="btn pipe-close" aria-label="Закрыть настройки шага" onClick={closeNode}>
            ✕
          </button>
        </div>
      <div className="pipe-body" key={cur.id}>
        {/* Что шаг сделал в последнем цикле — одной строкой; описание шага — в «?» у названия. */}
        {curRes && (
          <p className={`pipe-res-line ${curCls}`}>
            <i aria-hidden="true">{curGlyph}</i>
            <span>{curRes.text}</span>
            <time>{when}</time>
          </p>
        )}
        {/* У SAC настроек в панели нет — показываем его состояние. */}
        {cur.id === "sac" && (
          <article className="card">
            <dl className="pipe-fields">
              {info.sac.fields.map(([k, v]) => (
                <div key={k}>
                  <dt>{k}</dt>
                  <dd>{v}</dd>
                </div>
              ))}
            </dl>
          </article>
        )}
        {cur.id === "scan" && (
          <div className="cols">
            <div className="col">
              <WhatCard />
            </div>
            <div className="col">
              <HoursCard />
              <SymbolsCard />
            </div>
          </div>
        )}
        {cur.id === "strat" && (
          <div className="cols">
            <div className="col">
              <StrategiesCard />
            </div>
          </div>
        )}
        {cur.id === "ai" && (
          <>
            <AiPane />
            <ForecastCard />
          </>
        )}
        {cur.id === "sac" && (
          <div className="cols">
            <div className="col">
              <div className="btn-row">
                <button
                  className="btn"
                  type="button"
                  onClick={toLearn}
                >
                  Обучение и допуск модели
                </button>
              </div>
              <ForecastCard />
            </div>
          </div>
        )}
        {cur.id === "risk" && (
          <div className="cols">
            <div className="col">
              <RiskCard />
            </div>
            <div className="col">
              <RulesCard />
              <EmergencyCard />
            </div>
          </div>
        )}
        {cur.id === "order" && (
          <div className="cols">
            <div className="col">
              {need.order?.includes("Bybit") && (
                <article className="card">
                  <KeyLine service="bybit" />
                </article>
              )}
              <ConfirmCard />
            </div>
          </div>
        )}
        {cur.id === "track" && (
          <div className="cols">
            <div className="col">
              <TrackCard />
            </div>
          </div>
        )}
      </div>
      </aside>
    </section>
  );
}
