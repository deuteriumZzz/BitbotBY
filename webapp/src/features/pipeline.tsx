// Вкладка «Бот» как конвейер решений: полотно с узлами слева направо и мини-цепочкой AI,
// под ним — сводка выбранного узла и его карточки (те же, что были во вкладке раньше).
// Итогов последнего цикла на узлах пока нет: бот не сохраняет шаги решения (этап 2).

import { useEffect, useRef } from "react";

import type { BotNode, SecretItem, SettingDef } from "../lib/types";
import { actions, useStore } from "../store";
import { Q, Seg } from "../ui";
import { AiPane } from "./ai";
import { Lane, isStep, useLanes } from "./blocks";
import { ConfirmCard, RiskCard, StrategiesCard, TrackCard, WhatCard, cfg, presetName, presetOf, profileName } from "./bots";
import { chosenId } from "./chainops";
import { ModelCard } from "./jobs";
import { RulesCard } from "./rules";
import { UiSwitchRow } from "./uimode";
import { HoursCard, SymbolsCard } from "./settings";
import { NW, PipeCanvas } from "./pipecanvas";
import { SignalsCard } from "./signals";
import { mark } from "./trace";

const svg = (body: JSX.Element) => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {body}
  </svg>
);

export const NODES: readonly { id: BotNode; title: string; icon: JSX.Element }[] = [
  {
    id: "scan",
    title: "Сканер",
    icon: svg(
      <>
        <circle cx="11" cy="11" r="7" />
        <path d="m20 20-4-4" />
      </>,
    ),
  },
  { id: "strat", title: "Стратегии", icon: svg(<path d="M5 20V11M12 20V5M19 20v-6M3 20h18" />) },
  {
    id: "ai",
    title: "AI-цепочка",
    icon: svg(
      <>
        <rect x="6" y="6" width="12" height="12" rx="1" />
        <path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4" />
      </>,
    ),
  },
  {
    id: "sac",
    title: "SAC",
    icon: svg(
      <>
        <circle cx="12" cy="13" r="8" />
        <path d="M12 13l4-3M12 3v2" />
      </>,
    ),
  },
  { id: "risk", title: "Риск", icon: svg(<path d="M12 3l7 3v6c0 4-3 7-7 9-4-2-7-5-7-9V6z" />) },
  { id: "order", title: "Ордер", icon: svg(<path d="M4 8h14l-4-4M20 16H6l4 4" />) },
  { id: "track", title: "Сопровождение", icon: svg(<path d="M5 21V4h11l-2 4 2 4H5" />) },
];

const MODES = [
  ["local", "Local"],
  ["ai", "AI"],
  ["dqn", "SAC"],
  ["hybrid", "Hybrid"],
] as const;

const SIMPLE_MODES = [
  ["local", "Только стратегии"],
  ["ai", "Стратегии + AI"],
] as const;

/** «Кто принимает решение» — одно место для MODE; в «Простом» два понятных варианта. */
export function ModeSeg({ simple = false }: { simple?: boolean }) {
  const mode = useStore((s) => cfg<string>(s.config, "MODE", "ai"));
  const own = simple && mode !== "local" && mode !== "ai";
  return (
    <>
      <Seg
        className={simple ? "full" : undefined}
        value={own ? null : mode}
        options={simple ? SIMPLE_MODES : MODES}
        onChange={(v) => void actions.patchConfig({ MODE: v })}
        label="Кто принимает решение"
      />
      {own && <p className="note-soon">Сейчас своё, из «Профи»: {MODES.find(([m]) => m === mode)?.[1] ?? mode}. Выберите вариант выше, чтобы сменить.</p>}
    </>
  );
}

const AI_MODE: Record<string, string> = { free: "бесплатные", auto: "смешанный", paid: "платные" };

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

interface Info {
  desc: string;
  fields: [string, string][];
}

/** Строка состояния под узлом и сводка полей для каждого узла — только из того, что бот уже отдаёт. */
function usePipeline(mode: string): { status: Record<BotNode, string>; info: Record<BotNode, Info> } {
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
    ai: !ai ? "цепочка" : ai.single ? "один провайдер" : `${ai.chain.length} в цепочке`,
    sac: mode === "dqn" ? "решает сам" : "вместе с AI",
    risk: `${presetName(presetOf(c, presets))}${guardsOff ? " · без защит" : ""}`,
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
        ["Прогноз Chronos", show("CHRONOS_ENABLED")],
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
        ["Запасной путь", mode === "local" ? "не нужен в Local" : fallback ? "вкл" : "выкл"],
        ["С бэктестом", String(tested.length)],
        ["С плюсом в бэктесте", String(edge.length)],
      ],
    },
    ai: {
      desc: ai?.single
        ? "Решает один выбранный провайдер; остальные подключены, но не спрашиваются."
        : "Подтверждает или отклоняет сделку. Провайдеры спрашиваются по очереди: ошибка или лимит — сразу следующий.",
      fields: !ai
        ? [["Цепочка", "загружается"]]
        : [
            ["Как работать", ai.single ? "один провайдер" : "цепочка"],
            ["Какие модели", AI_MODE[ai.mode] ?? ai.mode],
            ai.single ? ["Решает", name(chosenId(ai)) || "—"] : ["В цепочке", String(ai.chain.length)],
            ["В резерве", String(ai.providers.filter((p) => p.disabled).length)],
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
            : "В этом режиме не участвует. Обучить можно заранее — модель пригодится в режимах SAC и Hybrid.",
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
      fields: [
        ["Как рискуем", presetName(presetOf(c, presets))],
        ["Риск на сделку", show("RISK_PER_TRADE")],
        ["Позиций сразу", show("MAX_POSITIONS")],
        ["Плечо", spot ? "1× на споте" : show("LEVERAGE_MODE")],
        ["Стоп-лосс", show("STOP_LOSS_PERCENT")],
        ["Дневной лимит убытка", show("DAILY_LOSS_LIMIT")],
        ["Убытков подряд до паузы", show("CIRCUIT_BREAKER_LOSSES")],
        ["Защиты портфеля", !paper ? "в Live всегда вкл" : guardsOff ? "выкл (Paper)" : "вкл"],
      ],
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
  return { status, info };
}

/** Кто решает и какая роль у стратегий — одной фразой под переключателем (как в src/core/trading_bot.py). */
const modeNote = (mode: string, fallback: boolean): string =>
  mode === "local"
    ? "Решают стратегии: на каждой монете запускаются все включённые, входит самый уверенный сигнал. AI не участвует."
    : mode === "dqn"
      ? `Решает нейросеть SAC. Стратегии ${fallback ? "включаются, только пока модель SAC не загружена (запасной путь)" : "не участвуют — запасной путь выключен"}.`
      : `Решает ${mode === "hybrid" ? "пара SAC и AI: сделка — только если оба согласны" : "AI"}. Включённые стратегии — только подсказка о типах сетапов${
          fallback ? "; если AI недоступен, решают они (запасной путь)" : ""
        }. Чтобы решали сами стратегии — выберите Local.`;

export function Pipeline() {
  const node = useStore((s) => s.node);
  const openNode = useStore((s) => s.openNode);
  const setTab = useStore((s) => s.setTab);
  const mode = useStore((s) => cfg<string>(s.config, "MODE", "ai"));
  const fallback = useStore((s) => cfg<boolean>(s.config, "LOCAL_FALLBACK", false));
  const { status, info } = usePipeline(mode);
  const at = NODES.findIndex((n) => n.id === node);
  const cur = NODES[at] ?? NODES[0];
  const part = takesPart(mode, cur.id);
  // Итог последнего цикла (src/decision_trace.py); null — бот ещё не прошёл цикл после обновления.
  const cycle = useStore((s) => s.lastCycle);
  const when = cycle ? new Date(cycle.at * 1000).toLocaleTimeString("ru-RU") : "";
  const curRes = cycle?.nodes[cur.id];
  const [curGlyph, curCls] = mark(curRes?.ok);
  // Настройки узла выезжают панелью справа, полотно остаётся на виду. Переход к узлу из поиска (⌘K) тоже открывает.
  const open = useStore((s) => s.nodeOpen);
  const closeNode = useStore((s) => s.closeNode);
  const closeBtn = useRef<HTMLButtonElement>(null);
  // Полосы под шагами — сводка; включается и выключается всё в панели шага.
  const lanes = useLanes();
  const toLearn = () => setTab("learn");
  useEffect(() => {
    if (!open) return;
    closeBtn.current?.focus({ preventScroll: true });
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && closeNode();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);
  return (
    <section className="view enter" aria-label="Цепочка">
      <UiSwitchRow />
      <div className="pipe-canvas">
        <div className="pipe-top">
          <div className="pipe-mode">
            <span className="label">
              Кто принимает решение{" "}
              <Q tip="Local — только индикаторы. AI — языковая модель. SAC — обученная нейросеть. Hybrid — SAC и AI должны согласиться. Узлы, которые в этом режиме не участвуют, погаснут." />
            </span>
            <ModeSeg />
            <p className="pipe-mode-note">{modeNote(mode, fallback)}</p>
          </div>
          <ul className="pipe-legend" aria-label="Обозначения">
            <li>
              <i className="on" />
              выбран
            </li>
            <li>
              <i />
              участвует
            </li>
            <li>
              <i className="idle" />
              не участвует в этом режиме
            </li>
            <li className="pipe-legend-res">
              итог цикла{when ? ` ${when}` : ""}: <b className="ok">▲</b>&nbsp;пройдено · <b>–</b>&nbsp;пропущено · <b className="bad">▼</b>&nbsp;остановил
            </li>
          </ul>
        </div>
        <PipeCanvas mode={mode} cycle={cycle} onMode={(m) => void actions.patchConfig({ MODE: m })}>
          {NODES.map(({ id, title, icon }, i) => {
            const on = takesPart(mode, id);
            const res = cycle?.nodes[id];
            const [glyph, cls] = mark(res?.ok);
            return (
                <div key={id} className="pipe-col" style={{ width: NW }}>
                <button
                  type="button"
                  className={`pipe-node${node === id ? " on" : ""}${on ? "" : " idle"}`}
                  aria-pressed={node === id}
                  aria-haspopup="dialog"
                  onClick={() => {
                    openNode(id);
                  }}
                >
                  <span className="hd">
                    {icon}
                    <i>{String(i + 1).padStart(2, "0")}</i>
                  </span>
                  <b>{title}</b>
                  <span className="st">{on ? status[id] : "не участвует"}</span>
                  {on && res && (
                    <span className={`res ${cls}`}>
                      <i aria-hidden="true">{glyph}</i> {res.text}
                    </span>
                  )}
                </button>
                  {isStep(id) ? (
                    <Lane step={id} items={lanes[id]} onOpen={() => openNode(id)} />
                  ) : (
                    <div className="lane2 none" aria-hidden="true" />
                  )}
                </div>
            );
          })}
        </PipeCanvas>
        <p className="pipe-hint">
          Нажмите на шаг — его настройки откроются справа. Штекер <i className="pw-plug-ico" aria-hidden="true" /> на проводе в «Риск» перетащите на Стратегии, AI
          или SAC — так меняется, кто решает. Числа на проводах — сколько монет прошло в последнем цикле. Ctrl + колесо — масштаб.
        </p>
      </div>
      <button type="button" className={`pipe-scrim${open ? " open" : ""}`} tabIndex={-1} aria-hidden="true" onClick={closeNode} />
      <aside className={`pipe-panel${open ? " open" : ""}`} role="dialog" aria-label={`Шаг «${cur.title}»`} aria-hidden={!open}>
        <div className="pipe-panel-h">
          <span className="pipe-sum-ico">{cur.icon}</span>
          <b>{cur.title}</b>
          <span className="label">
            шаг {String(at + 1).padStart(2, "0")} из {String(NODES.length).padStart(2, "0")}
          </span>
          <button ref={closeBtn} type="button" className="btn pipe-close" aria-label="Закрыть настройки шага" onClick={closeNode}>
            ✕
          </button>
        </div>
      <div className="pipe-body" key={cur.id}>
        <article className="card pipe-sum">
          <div className="card-h">
            <span className="pipe-sum-ico">{cur.icon}</span>
            <h2>{cur.title}</h2>
            <span className="pill sp">{part ? `шаг ${String(at + 1).padStart(2, "0")} из ${String(NODES.length).padStart(2, "0")}` : "не участвует в этом режиме"}</span>
          </div>
          <p className="pipe-desc">{info[cur.id].desc}</p>
          <p className={`pipe-res-line ${curCls}`}>
            <span className="label">Последний цикл{when ? ` · ${when}` : ""}</span>
            {curRes ? (
              <span>
                <i aria-hidden="true">{curGlyph}</i> {curRes.text}
              </span>
            ) : (
              <span>итогов пока нет — бот ещё не прошёл цикл после обновления</span>
            )}
          </p>
          <dl className="pipe-fields">
            {info[cur.id].fields.map(([k, v]) => (
              <div key={k}>
                <dt>{k}</dt>
                <dd>{v}</dd>
              </div>
            ))}
          </dl>
        </article>
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
              <SignalsCard />
              <StrategiesCard />
            </div>
          </div>
        )}
        {cur.id === "ai" && (
          <>
            <SignalsCard />
            <AiPane />
          </>
        )}
        {cur.id === "sac" && (
          <div className="cols">
            <div className="col">
              <ModelCard />
              <div className="btn-row">
                <button
                  className="btn"
                  type="button"
                  onClick={toLearn}
                >
                  Обучить модель и проверить →
                </button>
              </div>
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
            </div>
          </div>
        )}
        {cur.id === "order" && (
          <div className="cols">
            <div className="col">
              <ConfirmCard />
              <div className="btn-row">
                <button className="btn" type="button" onClick={() => setTab("settings")}>
                  Подключения: биржа, Telegram, ключи →
                </button>
              </div>
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
