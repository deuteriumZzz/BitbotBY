// Холст «Цепочки»: шаги одной сеткой слева направо, рамка «Решение» с дорожками AI и Нейросеть,
// тонкие провода с числом монет, шлюз фильтра рынка перед «Риском» и Telegram — источник над «Ордером».
// Сами блоки рисует pipeline.tsx (node / lane); здесь — раскладка, провода и прогон цикла.

import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";

import { CurrencyDollar, Database, Flask, Gavel, Key, Prohibit, SignIn, TelegramLogo } from "@phosphor-icons/react";

import { plural } from "../lib/format";
import { reducedMotion } from "../lib/platform";
import type { BotNode, CycleTrace } from "../lib/types";
import { actions } from "../store/actions";
import { positionPnl, useStore } from "../store";
import { Toggle } from "../components/controls";
import { cfg } from "./bots";
import { chanList } from "./settings/telegram";
import { Keys, keyIcon } from "./settings/keys";
import { brandLogo } from "../components/brand";
import { Icon } from "../components/icon";
import { Tiles } from "../components/irow";
import { Q } from "../components/tooltip";
import { cycleKind } from "./terminal/radar";
import { MARKET } from "./signals";
import { scrollWhenReady } from "./terminal";

/** Через какие шаги идёт сделка при этом MODE (как в SignalCombiner). */
export function route(mode: string): BotNode[] {
  const head: BotNode[] =
    mode === "local" ? ["scan", "strat"] : mode === "dqn" ? ["scan", "sac"] : mode === "hybrid" ? ["scan", "strat", "ai", "sac"] : ["scan", "strat", "ai"];
  return [...head, "risk", "order", "track"];
}

/** Сколько монет прошло по проводу a→b в последнем цикле. */
export function flow(f: NonNullable<CycleTrace["funnel"]>, a: BotNode, b: BotNode): number | null {
  if (a === "scan") return f.scanned;
  if (b === "ai") return f.candidates;
  if (b === "risk") return f.signals;
  if (a === "risk" || a === "order") return f.opened;
  return null;
}

/** До какого шага дошёл цикл: первый остановивший (▼), иначе последний пропустивший (▲). */
export function stopAt(r: BotNode[], nodes: CycleTrace["nodes"]): number {
  const bad = r.findIndex((n) => nodes[n]?.ok === false);
  if (bad >= 0) return bad;
  return r.reduce((last, n, i) => (nodes[n]?.ok === true ? i : last), 0);
}

/** Провода по порядку колонок: [откуда, куда] — для числа воронки. Решение считается одним шагом. */
const WIRES: [BotNode, BotNode][] = [
  ["scan", "strat"],
  ["strat", "ai"],
  ["ai", "risk"],
  ["risk", "order"],
  ["order", "track"],
];
/** Колонка шага: сколько проводов «пробежать» до него. */
const COL_OF: Record<BotNode, number> = { scan: 0, strat: 1, ai: 2, sac: 2, risk: 3, order: 4, track: 5 };

export function PipeCanvas(props: {
  mode: string;
  cycle: CycleTrace | null;
  onChannels: () => void;
  onData: () => void;
  /** Что сломано: узел → bad (не работает) или warn (работает с оговоркой). Провода к нему это показывают. */
  trouble: Partial<Record<string, "bad" | "warn">>;
  node: (id: BotNode) => ReactNode;
  lane: (id: "ai" | "sac") => ReactNode;
}) {
  const { mode, cycle, onChannels, onData, trouble, node, lane } = props;
  const paper = useStore((s) => s.bot?.paper ?? true);
  const openNode = useStore((s) => s.openNode);
  const chronos = useStore((s) => cfg<boolean>(s.config, "CHRONOS_ENABLED", false));
  const fcName = useStore((s) => FC_NAME[cfg<string>(s.config, "FORECASTER", "chronos")] ?? "Chronos");
  const minConf = useStore((s) => cfg<number>(s.config, paper ? "MIN_SIGNAL_CONFIDENCE_PAPER" : "MIN_SIGNAL_CONFIDENCE", paper ? 0.6 : 0.65));
  const f = cycle?.state === "scan" ? cycle.funnel : undefined;
  const mo = useMotion(mode, cycle);
  const tg = useChannels();
  // Посты каналов — оценка настроения для стратегий и AI; нужен вход в Telegram и хотя бы один канал.
  const sigOn = tg.signed && tg.all.length > 0 && mode !== "dqn";

  // ── Провода Telegram рисуются поверх сетки по реальным местам блоков ──
  const chRef = useRef<HTMLDivElement>(null);
  const [geo, setGeo] = useState<{ sig: string; byp: string; dat: string; sl: [number, number]; bl: [number, number]; w: number; h: number } | null>(null);
  // Схема «живёт», пока бот ищет сделки или ведёт позиции: по проводам идёт ток, точки дышат.
  const live = useStore((s) => {
    const k = cycleKind(s);
    return k === "scan" || k === "monitor";
  });
  const data = useData();
  useLayoutEffect(() => {
    const ch = chRef.current;
    if (!ch) return;
    // offset* — координаты внутри сетки: не зависят от анимации появления вкладки (transform у предков).
    const els = () => [".ch-tg", ".ch-dec", '[data-node="order"]', ".ch-data"].map((q) => ch.querySelector<HTMLElement>(q));
    const calc = () => {
      const [t, d, o, dt] = els();
      if (!t || !d || !o || !dt) return;
      // «Данные» → «Решение» слева: вправо до щели перед рамкой, вниз и в рамку сбоку.
      const dx = dt.offsetLeft + dt.offsetWidth;
      const dy = dt.offsetTop + dt.offsetHeight / 2;
      const gx = d.offsetLeft - 14;
      const my = d.offsetTop + 34;
      const sx = t.offsetLeft + t.offsetWidth / 2;
      const sy = t.offsetTop + t.offsetHeight;
      const ey = d.offsetTop;
      const bx = t.offsetLeft + t.offsetWidth;
      const by = t.offsetTop + t.offsetHeight / 2;
      const ox = o.offsetLeft + o.offsetWidth / 2;
      const oy = o.offsetTop;
      setGeo({
        sig: `M ${sx} ${sy} L ${sx} ${ey}`,
        byp: `M ${bx} ${by} L ${ox - 16} ${by} Q ${ox} ${by} ${ox} ${by + 16} L ${ox} ${oy}`,
        dat: `M ${dx} ${dy} L ${gx - 10} ${dy} Q ${gx} ${dy} ${gx} ${dy + 10} L ${gx} ${my - 10} Q ${gx} ${my} ${gx + 10} ${my} L ${d.offsetLeft} ${my}`,
        sl: [sx, (sy + ey) / 2],
        bl: [(bx + ox) / 2, by],
        w: ch.offsetWidth,
        h: ch.offsetHeight,
      });
    };
    calc();
    const ro = new ResizeObserver(calc);
    ro.observe(ch);
    els().forEach((el) => el && ro.observe(el));
    return () => ro.disconnect();
  }, [mode]);

  // ── Прогон цикла: провода по очереди вспыхивают до шага, где монеты остановились ──
  const [run, setRun] = useState<{ k: number; to: number; ok: boolean } | null>(null);
  const seen = useRef(cycle?.at);
  useEffect(() => {
    if (!cycle || cycle.at === seen.current) return;
    seen.current = cycle.at;
    if (cycle.state !== "scan" || reducedMotion()) return;
    const r = route(mode);
    const at = r[stopAt(r, cycle.nodes)];
    const k = Date.now();
    setRun({ k, to: COL_OF[at], ok: cycle.nodes[at]?.ok !== false });
    setTimeout(() => setRun((x) => (x?.k === k ? null : x)), 4000);
  }, [cycle?.at]); // eslint-disable-line react-hooks/exhaustive-deps

  const wire = (i: number, extra?: ReactNode) => {
    const [a, b] = WIRES[i];
    const n = f ? flow(f, a, b) : null;
    // Стратегии → Решение в режимах с AI — подсказка, а не сделка: пунктир.
    const hint = i === 1 && (mode === "ai" || mode === "hybrid");
    // В «Нейросети» стратегии не участвуют: провод через них приглушён.
    const dim = mode === "dqn" && (i === 0 || i === 1);
    const lit = run && i < run.to;
    // Провод сообщает о концах: сломан узел — красный пунктир без тока; с оговоркой — жёлтый, ток медленный.
    const ends = [a, b].map((x) => trouble[x === "ai" ? "decide" : x]);
    const st = ends.includes("bad") ? "bad" : ends.includes("warn") ? "warn" : "";
    // Поток монет: точек столько, сколько монет прошло (до 10) — воронка видна без цифр.
    // Без итога скана (бот ведёт позиции) — тихий фоновый поток из 2 точек, чтобы было видно, что всё живо.
    const dots = !live || dim || st === "bad" ? 0 : n ? Math.min(n, 10) : 2;
    return (
      <div className={`ch-w${hint ? " hint" : ""}${dim ? " dim" : ""}${live && !n ? " calm" : ""}${st ? ` ${st}` : ""}`}>
        <i className="ch-line" aria-hidden="true" />
        {Array.from({ length: dots }, (_, k) => (
          <i key={k} className="ch-p" style={{ animationDelay: `${(-k * 1.8) / dots}s` }} aria-hidden="true" />
        ))}
        {lit && <i className="ch-run" key={`${run.k}${i}`} style={{ animationDelay: `${i * 0.35}s` }} aria-hidden="true" />}
        {n !== null && (
          <b className={`ch-n${n ? "" : " zero"}`} title={`Прошло монет: ${n}`}>
            {n}
          </b>
        )}
        {extra}
      </div>
    );
  };

  return (
    <div className={`ch-wrap${live ? " live" : ""} k-${mo.kind}`} {...mo.attrs} style={{ ["--cyc" as string]: mo.cyc }}>
      <div className="ch" ref={chRef}>
        <DataSource onOpen={onData} d={data} />
        <TelegramSource onOpen={onChannels} ch={tg} />
        {geo && (
          <svg className="ch-ov" width={geo.w} height={geo.h} aria-hidden="true">
            <path className={`ch-ow data${data.on && mode !== "dqn" ? " on" : ""}`} d={geo.dat} />
            <path className={`ch-ow${sigOn ? " on" : ""}`} d={geo.sig} />
            <path className={`ch-ow byp${tg.trade ? " on" : ""}`} d={geo.byp} />
          </svg>
        )}
        {geo &&
          (sigOn ? (
            <span className="ch-ol on" style={{ left: geo.sl[0], top: geo.sl[1] }} title="Посты каналов дают оценку настроения рынка: её видят стратегии и AI. Решение остаётся за ними.">
              подсказывает
            </span>
          ) : mode === "dqn" ? (
            <span className="ch-ol" style={{ left: geo.sl[0], top: geo.sl[1] }}>
              нейросеть каналы не читает
            </span>
          ) : (
            null
          ))}
        {geo && (
          <span className={`ch-ol sw${tg.trade ? " on" : ""}`} style={{ left: geo.bl[0], top: geo.bl[1] }}>
            <Toggle on={tg.on} label="Сделки по постам доверенных каналов в обход решения" onChange={(v) => void actions.patchConfig({ TELEGRAM_TRUSTED_TRADING: v })} />
            <span title="Пост доверенного канала с монетой и направлением сразу открывает сделку — без стратегий и AI. Риск и защиты при этом работают.">
              в обход · {tg.trusted.length ? `доверенных ${tg.trusted.length}` : "нет доверенных"}
            </span>
          </span>
        )}
        <ExchangeSource />
        <div className={`ch-w ex${live ? " calm" : ""}${trouble.exchange ? ` ${trouble.exchange}` : ""}`} aria-hidden="true">
          <i className="ch-line" />
          {live && trouble.exchange !== "bad" && [0, 1].map((k) => <i key={k} className="ch-p" style={{ animationDelay: `${-k * 1.6}s` }} />)}
        </div>
        {node("scan")}
        {wire(0)}
        {node("strat")}
        {wire(1)}
        <div className={`ch-dec m-${mode}`} data-node="decide" role="group" aria-label="Решение">
          {/* Ярлык на кромке, как «15 мин» у Сканера: здесь принимается решение о сделке. */}
          <span className="ch-dec-h">
            <Gavel weight="bold" />
            Решение
          </span>
          {lane("ai")}
          {lane("sac")}
          <ForecastChip mode={mode} />
        </div>
        {wire(
          2,
          <button
            type="button"
            className="ch-gate"
            onClick={() => (useStore.setState({ riskManual: true }), openNode("risk"), scrollWhenReady("risk-conf"))}
            title={`Фильтр рынка${chronos && mode === "hybrid" ? ` и прогноз ${fcName}` : ""}: дальше в «Риск», только если уверенность сделки ≥ ${minConf}. Нажмите — откроется порог.`}
            aria-label={`Порог уверенности ${minConf}`}
          >
            <i aria-hidden="true" />
            <span>≥ {minConf}</span>
          </button>,
        )}
        {node("risk")}
        {wire(3)}
        {node("order")}
        {wire(4)}
        {node("track")}
      </div>
    </div>
  );
}

/** Каналы Telegram в двух режимах: все — сигнал для решения, доверенные — ещё и сделка по посту. */
function useChannels() {
  const config = useStore((s) => s.config);
  // Бот есть, если ключ сохранён и не отклонён; «не проверен» — тоже есть, просто без галочки.
  const tgStatus = useStore((s) => s.secrets?.find((x) => x.service === "telegram")?.status);
  const bot = tgStatus !== undefined && tgStatus !== "unset" && tgStatus !== "bad";
  const alerts = useStore((s) => s.alerts);
  const all = chanList(`${cfg(config, "TELEGRAM_SENTIMENT_CHANNELS", "")},${cfg(config, "TELEGRAM_TRUSTED_CHANNELS", "")}`);
  const signed = useStore((s) => Boolean(s.secrets?.find((x) => x.service === "telethon")?.signed_in));
  const trusted = chanList(cfg(config, "TELEGRAM_TRUSTED_CHANNELS", ""));
  const on = cfg<boolean>(config, "TELEGRAM_TRUSTED_TRADING", false);
  return { signed, trusted, on, bot, tgStatus, alerts, all, trade: on && signed && trusted.length > 0 };
}

/** Telegram — источник над «Решением»: бот и уведомления, каналы; провода — сигнал в решение и обход в ордер. */
function TelegramSource({ onOpen, ch }: { onOpen: () => void; ch: ReturnType<typeof useChannels> }) {
  const { signed, trusted, bot, tgStatus, alerts, all } = ch;
  const post = useStore((s) => s.trusted);
  // Ник аккаунта из сообщения о входе («Вход выполнен как @name»).
  const account = useStore((s) => /@[\w\d_]+/.exec(s.secrets?.find((x) => x.service === "telethon")?.message ?? "")?.[0] ?? "");
  const fresh = post && Date.now() / 1000 - post.at < 86_400 ? post : null;
  const dot = !bot ? "bad" : !signed ? "warn" : "ok";
  return (
    <div className="ch-tg">
      <button type="button" className="ch-tg-open" onClick={onOpen}>
        <span className="ch-h">
          <span className="ch-ico">
            <TelegramLogo weight="fill" color="#2AABEE" />
            <i className={`dot ${dot}`} />
          </span>
          <b>Telegram</b>
          {/* После входа — кружок аккаунта (первая буква ника); ник — по наведению. */}
          {signed && (
            <span className="ch-avatar" title={account ? `Каналы читаются как ${account}` : "Вход в Telegram выполнен"}>
              {(account.replace("@", "")[0] ?? "✓").toUpperCase()}
            </span>
          )}
        </span>
        {/* До входа вместо пустых счётчиков — сам призыв войти; после — сколько каналов бот слушает. */}
        {!signed && (
          <span className="ch-tg-login">
            <SignIn weight="bold" />
            Войти
          </span>
        )}
        <span className="ch-tg-kv">
          {signed && (
            <span title="Каналы, которые бот читает">
              <b>{all.length}</b> {plural(all.length, ["канал слушает", "канала слушает", "каналов слушает"])}
            </span>
          )}
          {signed && (
            <span title="Доверенные — их посты могут открывать сделки в обход решения">
              <b>{trusted.length}</b> доверенных
            </span>
          )}
          <span title="Уведомления бота в Telegram">
            <b>
              {alerts.filter((a) => a.enabled).length}/{alerts.length}
            </b>{" "}
            уведомл.
          </span>
        </span>
        {!bot ? (
          <span className="need">{tgStatus === "bad" ? "бот не работает" : "нужен бот"}</span>
        ) : (
          fresh && (
            <span className={`ch-r ${fresh.ok ? "ok" : "bad"}`} title={fresh.ok ? undefined : fresh.reason}>
              {fresh.ok ? "✓" : "✕"} @{fresh.channel}
              {fresh.symbol ? ` · ${fresh.symbol.split("/")[0]}` : ""}
            </span>
          )
        )}
      </button>
    </div>
  );
}

const FC_NAME: Record<string, string> = { chronos: "Chronos", timesfm: "TimesFM" };

type Fc = "off" | "chronos" | "timesfm";
const FC_OPTS: readonly (readonly [Fc, string, ReactNode, string])[] = [
  ["off", "Без прогноза", <Prohibit key="o" weight="bold" />, "сделку не сверяем"],
  ["chronos", "Chronos", Icon.chronos, "Amazon"],
  ["timesfm", "TimesFM", Icon.timesfm, "Google"],
];

/** Прогноз цены: выключен, Chronos или TimesFM. Меняется только здесь — значок на «Решении» и карточка в панелях AI и SAC. */
function useForecast() {
  const on = useStore((s) => cfg<boolean>(s.config, "CHRONOS_ENABLED", false));
  const who = useStore((s) => cfg<string>(s.config, "FORECASTER", "chronos"));
  const cur: Fc = on ? (who === "timesfm" ? "timesfm" : "chronos") : "off";
  const pick = (v: Fc) => void actions.patchConfig(v === "off" ? { CHRONOS_ENABLED: false } : { CHRONOS_ENABLED: true, FORECASTER: v });
  return { cur, pick };
}

const FC_TIP = "Прогноз цены сверяет сделку: совпал — уверенность +0.05, против — −0.15.";

function ForecastChip({ mode }: { mode: string }) {
  const { cur, pick } = useForecast();
  const idle = mode !== "hybrid";
  return (
    <span
      className={`ch-fc${cur !== "off" ? " on" : ""}${idle ? " idle" : ""}`}
      role="radiogroup"
      aria-label="Прогноз цены"
      title={`${FC_TIP}${idle ? " Работает только в режиме «AI + нейросеть»." : ""}`}
    >
      {FC_OPTS.map(([v, name, icon, by]) => (
        <button key={v} type="button" role="radio" aria-checked={cur === v} className={cur === v ? "cur" : ""} title={v === "off" ? name : `${name} (${by}) — прогноз цены по свечам`} onClick={() => pick(v)}>
          {icon}
        </button>
      ))}
    </span>
  );
}

/** Та же настройка в выезжающих панелях AI и SAC. */
export function ForecastCard() {
  const { cur, pick } = useForecast();
  const mode = useStore((s) => s.bot?.mode ?? "ai");
  return (
    <article className="card">
      <div className="card-h">
        <h2>Прогноз цены</h2>
        <Q tip={`${FC_TIP} Обе модели прогнозируют по свечам; выбор меняется сразу.`} />
      </div>
      <Tiles value={cur} options={FC_OPTS} label="Прогноз цены" onChange={pick} />
      {mode !== "hybrid" && cur !== "off" && <p className="note-soon">Работает только в режиме «AI + нейросеть» — в этом режиме не используется.</p>}
    </article>
  );
}

/** Источники данных с ключом (без Telegram — у него свой блок) и сколько из них работает. */
function useData() {
  const secrets = useStore((s) => s.secrets);
  const items = (secrets ?? []).filter((s) => s.group === "Данные и соцсети" && s.service !== "telethon");
  const ok = items.filter((s) => s.status === "ok").length;
  return { items, ok, on: ok > 0, free: MARKET.filter((m) => !m.key).length };
}

/** «Данные» — новости, соцсети, ончейн, макро-календарь. Каждый логотип — кнопка: сразу открывает ключ
 *  этого источника; заголовок открывает панель со всеми источниками. */
function DataSource({ onOpen, d }: { onOpen: () => void; d: ReturnType<typeof useData> }) {
  const [key, setKey] = useState<string | null>(null);
  return (
    <div className="ch-data">
      <button type="button" className="ch-data-open" onClick={onOpen}>
        <span className="ch-h">
          <span className="ch-ico">
            <Database />
            <i className={`dot ${d.on ? "ok" : "warn"}`} />
          </span>
          <b>Данные</b>
        </span>
      </button>
      <span className="ch-logos">
        {d.items.map((s) => (
          <button
            key={s.service}
            type="button"
            className={`ch-logo ${s.status}`}
            title={`${s.name}: ${s.status === "ok" ? "работает" : s.status === "unset" ? "не подключён" : "нужно проверить"} — открыть ключ`}
            aria-label={`${s.name}: открыть ключ`}
            onClick={() => setKey(s.service)}
          >
            {keyIcon(s.service)}
          </button>
        ))}
      </span>
      <span className="ch-u">
        {d.on ? "подсказывают решению" : "ни один не подключён"} · с ключом {d.ok}/{d.items.length}, без ключей ещё {d.free}
      </span>
      <Keys open={key} setOpen={setKey} only={[]} />
    </div>
  );
}

/** Биржа — первая в цепочке: откуда цены и куда уходят ордера. Клик — ключ и секрет Bybit. */
function ExchangeSource() {
  const item = useStore((s) => s.secrets?.find((x) => x.service === "bybit"));
  const paper = useStore((s) => s.bot?.paper ?? true);
  const [open, setOpen] = useState<string | null>(null);
  const st = item?.status ?? "unset";
  const last4 = item?.fields.find((f) => f.secret && f.last4)?.last4;
  const ok = st === "ok";
  const dot = ok ? "ok" : st === "bad" || st === "unset" ? "bad" : "warn";
  const text = ok ? "подключена" : st === "unset" ? "нет ключа" : st === "bad" ? "ключ не работает" : "проверить ключ";
  return (
    <div className={`ch-ex${ok ? "" : ` ${dot}`}`} data-node="exchange">
      <button type="button" className="ch-ex-open" onClick={() => setOpen("bybit")} aria-label={`Биржа Bybit: ${text}. Открыть ключ и секрет`}>
        <span className="ch-ex-t">Биржа</span>
        <span className="ch-ex-logo">
          {brandLogo("bybit")}
          <i className={`dot ${dot}`} />
        </span>
        {/* Режим — только значком в углу (подпись есть в шапке): колба — Paper, доллар — Live. */}
        <em className={`ch-badge ch-mode ${paper ? "paper" : "live"}`} title={paper ? "Paper: сделки виртуальные" : "Live: реальные деньги"} aria-label={paper ? "Paper" : "Live"}>
          {paper ? <Flask weight="bold" /> : <CurrencyDollar weight="bold" />}
        </em>
        <span className={`ch-ex-key ${dot}`} title={ok && last4 ? `Ключ ••••${last4}` : undefined}>
          {!ok && <Key weight="bold" />}
          {text}
        </span>
      </button>
      <Keys open={open} setOpen={setOpen} only={[]} />
    </div>
  );
}

/** Где бот сейчас: этап цикла → блок схемы (rejected — блок, где монету отсеяли). */
const STAGE_NODE: Record<string, string> = { scan: "scan", prescore: "strat", signal: "decide", guards: "risk", open: "order" };
const LANE_NODE = ["scan", "strat", "decide", "risk"];

/** Живая схема: подсветка текущего этапа, «AI думает», вспышки сделок, импульсы источников, темп по состоянию бота.
 *  Всё — атрибуты data-* на обёртке; сами эффекты в CSS (и выключаются при «уменьшить движение»). */
function useMotion(mode: string, cycle: CycleTrace | null) {
  const radar = useStore((s) => s.radar);
  const kind = useStore(cycleKind);
  const guard = useStore((s) => Object.values(s.guards).some((g) => g.active));
  const feedTop = useStore((s) => s.feed[0]);
  const post = useStore((s) => s.trusted);
  const every = useStore((s) => cfg<number>(s.config, "TRADING_INTERVAL", 900));
  const pnl = useStore((s) => Math.round(Object.values(s.positions).reduce((a, p) => a + positionPnl(p).pnl, 0)));
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  // Короткие события: значение держится ~1,6 с, потом сбрасывается — CSS-анимация проигрывается заново на каждое.
  const [flash, setFlash] = useState<Record<string, string>>({});
  const pulse = (k: string, v: string, ms = 1600) => {
    setFlash((f) => ({ ...f, [k]: v }));
    setTimeout(() => setFlash((f) => (f[k] === v ? { ...f, [k]: "" } : f)), ms);
  };
  const seenFeed = useRef(feedTop?.id);
  useEffect(() => {
    if (!feedTop || feedTop.id === seenFeed.current) return;
    seenFeed.current = feedTop.id;
    if (feedTop.kind === "open") pulse("burst", "order");
    if (feedTop.kind === "win" || feedTop.kind === "loss") pulse("close", feedTop.kind);
  }, [feedTop?.id]); // eslint-disable-line react-hooks/exhaustive-deps
  const seenPost = useRef(post?.at);
  useEffect(() => {
    if (!post || post.at === seenPost.current) return;
    seenPost.current = post.at;
    pulse("tg", post.ok ? "trade" : "signal");
  }, [post?.at]); // eslint-disable-line react-hooks/exhaustive-deps
  const seenStage = useRef(radar.stageAt);
  useEffect(() => {
    if (!radar.stageAt || radar.stageAt === seenStage.current) return;
    seenStage.current = radar.stageAt;
    if (radar.stage === "scan") pulse("dat", "1");
    if (radar.stage === "rejected") {
      const last = Object.values(radar.rejected).sort((a, b) => b.at - a.at)[0];
      if (last) pulse("rej", LANE_NODE[last.lane] ?? "", 900);
    }
  }, [radar.stageAt]); // eslint-disable-line react-hooks/exhaustive-deps
  const prevPnl = useRef(pnl);
  useEffect(() => {
    if (pnl !== prevPnl.current) pulse("pnl", pnl > prevPnl.current ? "up" : "down", 800);
    prevPnl.current = pnl;
  }, [pnl]); // eslint-disable-line react-hooks/exhaustive-deps
  // Этап «свежий», если пришёл недавно и цикл после него ещё не закончился.
  const fresh = radar.stageAt && now - radar.stageAt < 12_000 && (!cycle || cycle.at * 1000 < radar.stageAt);
  const busy = fresh && radar.stage ? (STAGE_NODE[radar.stage] ?? "") : "";
  // Модель думает, пока стратегии отбирают монеты и пока идёт решение (AI или нейросеть участвуют).
  const thinking = (busy === "strat" || busy === "decide") && mode !== "local";
  const live = kind === "scan" || kind === "monitor";
  const cyc = cycle ? Math.min(100, Math.max(0, ((now / 1000 - cycle.at) / every) * 100)) : 0;
  return {
    kind,
    cyc,
    // Пустое значение — без атрибута: иначе CSS [data-…] сработал бы всегда.
    attrs: Object.fromEntries(
      Object.entries({
        "data-busy": busy,
        "data-think": thinking ? "1" : "",
        "data-guard": guard ? "1" : "",
        "data-burst": flash.burst,
        "data-close": flash.close,
        "data-tg": flash.tg,
        "data-dat": flash.dat,
        "data-rej": flash.rej,
        "data-pnl": flash.pnl,
        "data-soon": live && cyc > 90 ? "1" : "",
      }).filter(([, v]) => v),
    ),
  };
}
