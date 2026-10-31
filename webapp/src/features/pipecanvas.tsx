// Холст «Цепочки»: провода между шагами, развилка «кто решает», дорожка доверенных каналов
// Telegram и прогон последнего цикла. Шаги с их полосами блоков рисует pipeline.tsx — здесь
// они лежат в мире, который масштабируется (как схема AI) и прокручивается вбок.
// Порядок шагов зашит в боте, поэтому сами шаги не двигаются: перетаскивается только штекер
// провода, который ведёт в «Риск», — это и есть «кто решает» (MODE, с «Отменить»).

import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";

import { plural } from "../lib/format";
import { reducedMotion } from "../lib/platform";
import type { BotNode, CycleTrace, SecretItem } from "../lib/types";
import { actions, useStore } from "../store";
import { Toggle } from "../ui";
import { cfg } from "./bots";
import { chanList } from "./settings";
import { MARKET, setChronos } from "./signals";
import { scrollWhenReady } from "./terminal";

export const NW = 150; // ширина шага
const GX = 184; // шаг колонок
export const GAP = GX - NW;
export const X0 = 16;
const TOP = 14; // верхний ряд: сезон рынка, каналы Telegram, лимиты каналов
const TH = 72;
const MID = TOP + TH + 40; // средний ряд: профиль рынка над «Сканером»
export const ROW = MID + TH + 86; // верх ряда шагов: над ним дуги развилки
const PY = ROW + 32; // высота портов шагов
const TY = TOP + TH / 2;
const W = X0 * 2 + GX * 6 + NW;

const COL: Record<BotNode, number> = { scan: 0, strat: 1, ai: 2, sac: 3, risk: 4, order: 5, track: 6 };
const xOf = (n: BotNode) => X0 + GX * COL[n];
const MODE_OF: Partial<Record<BotNode, [string, string]>> = { strat: ["local", "Local"], ai: ["ai", "AI"], sac: ["dqn", "SAC"] };

/** Через какие шаги идёт сделка при этом MODE (как в SignalCombiner). */
export function route(mode: string): BotNode[] {
  const head: BotNode[] =
    mode === "local" ? ["scan", "strat"] : mode === "dqn" ? ["scan", "sac"] : mode === "hybrid" ? ["scan", "strat", "ai", "sac"] : ["scan", "strat", "ai"];
  return [...head, "risk", "order", "track"];
}

/** Отрезок провода от правого порта a до левого порта b (без «M»); через шаги — дугой над ними. */
function seg(a: BotNode, b: BotNode, lift = 64): string {
  const x1 = xOf(a) + NW;
  const x2 = xOf(b);
  if (COL[b] - COL[a] === 1) return `L ${x2} ${PY}`;
  const top = ROW - lift;
  return `C ${x1 + 24} ${top}, ${x2 - 24} ${top}, ${x2} ${PY}`;
}
const wire = (a: BotNode, b: BotNode, lift?: number) => `M ${xOf(a) + NW} ${PY} ${seg(a, b, lift)}`;
/** Середина провода — место для числа воронки. */
const mid = (a: BotNode, b: BotNode, lift = 64): [number, number] => [
  (xOf(a) + NW + xOf(b)) / 2,
  COL[b] - COL[a] === 1 ? PY : 0.25 * PY + 0.75 * (ROW - lift),
];

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

const pct = (v: unknown, d: number) => `${+((Number(v ?? d) || d) * 100).toFixed(2)}%`;

/** onMode — выбор штекером; записывает его дом MODE (pipeline.tsx, «Кто принимает решение»). */
export function PipeCanvas(props: { mode: string; cycle: CycleTrace | null; onMode: (mode: string) => void; children: ReactNode }) {
  const { mode, cycle, onMode, children } = props;
  const view = useRef<HTMLDivElement>(null);
  const world = useRef<HTMLDivElement>(null);
  const [z, setZ] = useState(1);
  const [h, setH] = useState(ROW + 200);
  const [plug, setPlug] = useState<{ x: number; y: number; to: BotNode | null } | null>(null);
  const [run, setRun] = useState<{ k: number; d: string; s: number; at: BotNode; ok: boolean } | null>(null);
  const fallback = useStore((s) => cfg<boolean>(s.config, "LOCAL_FALLBACK", false));
  const r = route(mode);
  const dec = r[r.indexOf("risk") - 1];

  // ── Масштаб: по ширине при открытии и смене ширины, но не мельче 85% — на телефоне текст
  // остаётся читаемым, лента прокручивается пальцем. Ctrl + колесо и кнопки — вручную ──
  const fit = () => {
    const w = view.current?.clientWidth ?? W;
    setZ(Math.max(0.85, Math.min(1, w / W)));
    view.current?.scrollTo({ left: 0 });
  };
  useLayoutEffect(() => {
    const v = view.current;
    const wd = world.current;
    if (!v || !wd) return;
    let lastW = 0;
    const ro = new ResizeObserver(() => {
      setH(wd.offsetHeight);
      if (v.clientWidth !== lastW) (lastW = v.clientWidth), fit();
    });
    ro.observe(v);
    ro.observe(wd);
    // React вешает колесо пассивно — preventDefault для Ctrl + колесо нужен свой слушатель.
    const wheel = (e: WheelEvent) => {
      if (!e.ctrlKey) return;
      e.preventDefault();
      setZ((x) => Math.max(0.5, Math.min(1.4, x * (e.deltaY < 0 ? 1.08 : 1 / 1.08))));
    };
    v.addEventListener("wheel", wheel, { passive: false });
    return () => (ro.disconnect(), v.removeEventListener("wheel", wheel));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const zoom = (k: number) => setZ((x) => Math.max(0.5, Math.min(1.4, x * k)));

  // Мышью фон тянется вбок; пальцем — обычная прокрутка.
  const pan = (e: React.PointerEvent) => {
    if (e.pointerType !== "mouse" || e.button !== 0 || (e.target as HTMLElement).closest("button,[draggable=true],.lane2,.pw-plug")) return;
    const v = view.current!;
    const sx = e.clientX;
    const s0 = v.scrollLeft;
    const mv = (ev: PointerEvent) => (v.scrollLeft = s0 - (ev.clientX - sx));
    const up = () => (window.removeEventListener("pointermove", mv), window.removeEventListener("pointerup", up));
    window.addEventListener("pointermove", mv);
    window.addEventListener("pointerup", up);
  };

  // ── Штекер: тянут на Стратегии, AI или SAC — так меняется, кто решает ──
  const toWorld = (cx: number, cy: number) => {
    const b = world.current!.getBoundingClientRect();
    return { x: (cx - b.left) / z, y: (cy - b.top) / z };
  };
  const target = (x: number, y: number): BotNode | null =>
    y < ROW - 40 || y > ROW + 150 ? null : ((["strat", "ai", "sac"] as BotNode[]).find((n) => x >= xOf(n) - GAP / 2 && x <= xOf(n) + NW + GAP / 2) ?? null);
  const grab = (e: React.PointerEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const mv = (ev: PointerEvent) => {
      const p = toWorld(ev.clientX, ev.clientY);
      setPlug({ ...p, to: target(p.x, p.y) });
    };
    const up = (ev: PointerEvent) => {
      window.removeEventListener("pointermove", mv);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
      const p = toWorld(ev.clientX, ev.clientY);
      const to = ev.type === "pointerup" ? target(p.x, p.y) : null;
      setPlug(null);
      const next = to && MODE_OF[to]?.[0];
      if (next && (next !== mode || mode === "hybrid")) onMode(next);
    };
    window.addEventListener("pointermove", mv);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
    mv(e.nativeEvent);
  };

  // ── Прогон цикла: точка бежит по проводу до шага, где монеты остановились ──
  const seen = useRef(cycle?.at);
  useEffect(() => {
    if (!cycle || cycle.at === seen.current) return;
    seen.current = cycle.at;
    if (cycle.state !== "scan" || reducedMotion()) return;
    const stop = stopAt(r, cycle.nodes);
    if (stop < 1) return;
    let d = `M ${xOf(r[0]) + NW} ${PY}`;
    for (let i = 1; i <= stop; i++) d += ` ${seg(r[i - 1], r[i])}${i < stop ? ` L ${xOf(r[i]) + NW} ${PY}` : ""}`;
    const k = Date.now();
    const s = Math.min(2.4, 0.45 * stop + 0.4);
    setRun({ k, d, s, at: r[stop], ok: cycle.nodes[r[stop]]?.ok !== false });
    // Без отмены при новом цикле: иначе точка зависает, если следующий цикл пришёл раньше.
    setTimeout(() => setRun((x) => (x?.k === k ? null : x)), (s + 1.2) * 1000);
  }, [cycle?.at]); // eslint-disable-line react-hooks/exhaustive-deps

  const f = cycle?.state === "scan" ? cycle.funnel : undefined;
  const edges = r.slice(1).map((b, i) => [r[i], b] as const);
  const plugTo = plug?.to;
  return (
    <div className="pw-wrap">
      <div className="cg-tools" role="toolbar" aria-label="Масштаб цепочки">
        <button className="btn" type="button" aria-label="Приблизить" onClick={() => zoom(1.15)}>
          +
        </button>
        <button className="btn" type="button" aria-label="Отдалить" onClick={() => zoom(1 / 1.15)}>
          −
        </button>
        <button className="btn" type="button" onClick={fit}>
          По размеру
        </button>
        <span className="cg-z">{Math.round(z * 100)}%</span>
      </div>
      <div className="pw-view" ref={view} onPointerDown={pan}>
        <div style={{ width: W * z, height: h * z }}>
          <div className="pw-world" ref={world} style={{ width: W, transform: `scale(${z})` }}>
            <svg className="pw-wires" width={W} height={h} aria-hidden="true">
              {(Object.keys(COL) as BotNode[]).slice(1).map((n) => (
                <path key={n} className="pw-rail" d={`M ${xOf(n) - GAP} ${PY} L ${xOf(n)} ${PY}`} />
              ))}
              {fallback && mode !== "local" && <path className="pw-wire fall" d={wire("strat", "risk", 104)} />}
              {/* Живой путь: приглушённая основа прорисовывается при смене режима, по ней бегут пакеты. */}
              {edges.map(([a, b]) => (
                <g key={`${mode}${a}${b}`} className={`pw-edge${b === "risk" && plug ? " lifted" : ""}${a === "strat" && b === "ai" ? " hint" : ""}`}>
                  <path className="pw-wire" d={wire(a, b)} pathLength={1} />
                  <path className="pw-flow" d={wire(a, b)} />
                </g>
              ))}
              {(Object.keys(COL) as BotNode[]).map((n) => (
                <g key={n} className="pw-port">
                  {n !== "scan" && <circle className={edges.some(([, b]) => b === n) ? "on" : ""} cx={xOf(n)} cy={PY} r={3.5} />}
                  {n !== "track" && <circle className={edges.some(([a]) => a === n) ? "on" : ""} cx={xOf(n) + NW} cy={PY} r={3.5} />}
                </g>
              ))}
              {plug && <path className="pw-wire ghost" d={`M ${plug.x} ${plug.y} C ${plug.x + 60} ${plug.y}, ${xOf("risk") - 60} ${PY}, ${xOf("risk")} ${PY}`} />}
              <MarketWires />
              <SignalWires mode={mode} />
            </svg>

            {fallback && mode !== "local" && (
              <span className="pw-tag fall" style={{ left: mid("strat", "risk", 104)[0], top: mid("strat", "risk", 104)[1] }}>
                запасной путь
              </span>
            )}
            {(mode === "ai" || mode === "hybrid") && (
              <span className="pw-tag" style={{ left: mid("strat", "ai")[0], top: ROW - 12 }}>
                подсказка
              </span>
            )}
            {f &&
              edges.map(([a, b]) => {
                const n = flow(f, a, b);
                if (n === null) return null;
                const [x, y] = mid(a, b);
                return (
                  <span key={`${a}${b}`} className={`pw-n${n ? "" : " zero"}`} style={{ left: x, top: y }} title={`Прошло монет: ${n}`}>
                    {n}
                  </span>
                );
              })}

            <MarketLane />
            <SignalLane mode={mode} />

            {plugTo && (
              <span className="pw-drop" style={{ left: xOf(plugTo) - 6, top: ROW - 6, width: NW + 12 }}>
                <b>Решает: {MODE_OF[plugTo]?.[1]}</b>
              </span>
            )}
            <span
              className={`pw-plug${plug ? " held" : ""}`}
              style={{ left: plug ? plug.x : xOf(dec) + NW, top: plug ? plug.y : PY }}
              onPointerDown={grab}
              title="Кто решает: перетащите штекер на Стратегии, AI или SAC"
              aria-hidden="true"
            />

            {run && <i className="cg-dot pw-dot" key={run.k} style={{ offsetPath: `path("${run.d}")`, animationDuration: `${run.s}s` }} />}
            {run && <i className={`pw-hit${run.ok ? "" : " bad"}`} key={`h${run.k}`} style={{ left: xOf(run.at), top: PY, animationDelay: `${run.s}s` }} />}

            <div className="pw-row" style={{ paddingTop: ROW, paddingLeft: X0, gap: GAP }}>
              {children}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/** Каналы Telegram в двух режимах: все — сигнал для решения, доверенные — ещё и сделка по посту. */
function useChannels() {
  const config = useStore((s) => s.config);
  const signed = useStore((s) => Boolean(s.secrets?.find((x) => x.service === "telethon")?.signed_in));
  const trusted = chanList(cfg(config, "TELEGRAM_TRUSTED_CHANNELS", ""));
  const on = cfg<boolean>(config, "TELEGRAM_TRUSTED_TRADING", false);
  return { config, signed, trusted, on, trade: on && signed && trusted.length > 0 };
}

const MX = xOf("scan") + NW / 2; // цепочка рынка: «Сезон» → «Профиль» → «Сканер», сверху вниз
// «Сигналы для решения» — над «Стратегиями» и AI, на два верхних ряда; «Доверенные каналы» — рядом, над SAC.
const SX = xOf("strat");
const SW = NW + GX;
const SH = MID + TH - TOP;
const GATE_X = xOf("risk") + 60; // куда провод фильтра рынка входит в порог над «Риском»
/** Кому идут сигналы: AI (в промпт) — в режимах с AI, иначе стратегиям. */
const sigTo = (mode: string): BotNode => (mode === "ai" || mode === "hybrid" ? "ai" : "strat");

/** Провод сбоку от ряда шагов; key по состоянию — ожил: заново прорисовывается и по нему пускается поток. */
const side = (live: boolean, paths: string[], ports: number[][], k: string) => (
  <g key={`${k}${live}`} className={`pw-edge side${live ? "" : " off"}`}>
    {paths.map((d) => (
      <g key={d}>
        <path className="pw-wire" d={d} pathLength={live ? 1 : undefined} />
        {live && <path className="pw-flow" d={d} />}
      </g>
    ))}
    <g className="pw-port">
      {ports.map(([x, y]) => (
        <circle key={`${x}:${y}`} className={live ? "on" : ""} cx={x} cy={y} r={3.5} />
      ))}
    </g>
  </g>
);

/** Сезон меняет профиль сам (авто) или только предлагает; профиль задаёт, какие монеты смотрит сканер. */
function MarketWires() {
  const auto = useStore((s) => cfg<string>(s.config, "SEASON_SWITCH_MODE", "alert") === "auto");
  return (
    <>
      {side(auto, [`M ${MX} ${TOP + TH} L ${MX} ${MID}`], [[MX, TOP + TH], [MX, MID]], "season")}
      {side(true, [`M ${MX} ${MID + TH} L ${MX} ${ROW}`], [[MX, MID + TH], [MX, ROW]], "profile")}
    </>
  );
}

function MarketLane() {
  const season = useStore((s) => s.market?.season ?? null);
  const profiles = useStore((s) => s.profiles);
  const config = useStore((s) => s.config);
  const openNode = useStore((s) => s.openNode);
  const auto = cfg<string>(config, "SEASON_SWITCH_MODE", "alert") === "auto";
  const active = profiles.find((p) => p.active);
  const name = !season ? "нет данных" : season.signal === "altcoin" ? "альтсезон" : season.signal === "bluechip" ? "сезон BTC" : "нейтрально";
  const facts = season
    ? [season.altcoin_index !== null && `индекс ${Math.round(season.altcoin_index)}`, season.btc_dominance !== null && `BTC ${Math.round(season.btc_dominance)}%`]
        .filter(Boolean)
        .join(" · ")
    : "";
  const scanN = cfg<number>(config, "SCAN_TOP_N", 20);
  return (
    <>
      <button type="button" className="pipe-node pw-tg" style={{ left: xOf("scan"), top: TOP, width: NW, height: TH }} onClick={() => openNode("scan")}>
        <b>Сезон рынка</b>
        <span className="st">{name}</span>
        {facts && <span className="st">{facts}</span>}
      </button>
      <span className="pw-tag side" style={{ left: MX + 10, top: (TOP + TH + MID) / 2 }}>
        {auto ? "меняет сам" : "предлагает"}
      </span>
      <button type="button" className="pipe-node pw-tg" style={{ left: xOf("scan"), top: MID, width: NW, height: TH }} onClick={() => openNode("scan")}>
        <b>Профиль рынка</b>
        <span className="st">{active ? `${active.label} · ${active.timeframe}` : "—"}</span>
        <span className="st">
          {scanN} {plural(scanN, ["монета", "монеты", "монет"])} в скане
        </span>
      </button>
    </>
  );
}

/** Сигналы — тому, кто решает; Telegram ещё и вторым режимом: доверенные → «Лимиты каналов» → «Ордер», мимо всех проверок. */
function SignalWires({ mode }: { mode: string }) {
  const { trade } = useChannels();
  const tx = xOf(sigTo(mode)) + NW / 2;
  const x2 = xOf("order") + NW / 2;
  return (
    <>
      {side(mode !== "dqn", [`M ${tx} ${TOP + SH} L ${tx} ${ROW}`], [[tx, TOP + SH], [tx, ROW]], `sig${tx}`)}
      {/* Фильтр рынка и Chronos работают после решения — провод к порогу перед «Риском» (в Local их нет). */}
      {side(
        mode !== "local",
        [`M ${SX + SW - 24} ${TOP + SH} C ${SX + SW + 120} ${TOP + SH}, ${GATE_X} ${ROW - 80}, ${GATE_X} ${ROW - 37}`],
        [[SX + SW - 24, TOP + SH], [GATE_X, ROW - 37]],
        "filter",
      )}
      {side(
        trade,
        [
          `M ${SX + SW} ${TY} L ${xOf("sac")} ${TY}`,
          `M ${xOf("sac") + NW} ${TY} L ${xOf("risk")} ${TY}`,
          `M ${xOf("risk") + NW} ${TY} C ${x2} ${TY}, ${x2} ${TY}, ${x2} ${ROW}`,
        ],
        [[SX + SW, TY], [xOf("sac"), TY], [xOf("sac") + NW, TY], [xOf("risk"), TY], [xOf("risk") + NW, TY], [x2, ROW]],
        "trade",
      )}
    </>
  );
}

const srcDot = (s: SecretItem): string =>
  s.service === "telethon" ? (s.signed_in ? "ok" : "") : s.status === "ok" ? "ok" : s.status === "bad" ? "bad" : s.status === "warn" ? "warn" : "";

function SignalLane({ mode }: { mode: string }) {
  const { config, signed, trusted, on, trade } = useChannels();
  const secrets = useStore((s) => s.secrets);
  const post = useStore((s) => s.trusted);
  const paper = useStore((s) => s.bot?.paper ?? true);
  const setTab = useStore((s) => s.setTab);
  const openNode = useStore((s) => s.openNode);
  const to = sigTo(mode);
  // Три группы, как в SignalCombiner: промпт AI, фильтр рынка (13), Chronos (только Hybrid).
  const pick = (ids: string[]) => ids.map((id) => (secrets ?? []).find((s) => s.service === id)).filter((s): s is SecretItem => Boolean(s));
  const prompt = pick(["news", "twitter", "telethon"]);
  const filterKeys = pick(MARKET.flatMap((m) => (m.key ? [m.key] : [])));
  const fg = useStore((s) => s.market?.fear_greed ?? null);
  const chronos = cfg<boolean>(config, "CHRONOS_ENABLED", false);
  const chans = chanList(`${cfg(config, "TELEGRAM_SENTIMENT_CHANNELS", "")},${cfg(config, "TELEGRAM_TRUSTED_CHANNELS", "")}`).length;
  const minConf = cfg<number>(config, paper ? "MIN_SIGNAL_CONFIDENCE_PAPER" : "MIN_SIGNAL_CONFIDENCE", paper ? 0.6 : 0.65);
  const toChannels = () => (setTab("settings"), scrollWhenReady("tg-channels"));
  const fresh = post && Date.now() / 1000 - post.at < 86_400 ? post : null;
  const slots = cfg<number>(config, "TELEGRAM_TRUSTED_SLOTS", 2);
  return (
    <>
      <div className="pipe-node pw-tg pw-sig" style={{ left: SX, top: TOP, width: SW, height: SH }}>
        <button type="button" className="pw-open" onClick={() => openNode(to)}>
          <b>Сигналы для решения</b>
          <dl className="pw-sigs">
            <div className={mode === "ai" || mode === "hybrid" ? "" : "off"}>
              <dt>в промпт AI</dt>
              <dd>
                {prompt.map((s) => (
                  <span key={s.service}>
                    <i className={`dot ${srcDot(s)}`} aria-hidden="true" />
                    {s.service === "telethon" ? `Telegram · ${chans}` : s.service === "twitter" ? "X" : s.name}
                  </span>
                ))}
              </dd>
            </div>
            <div className={mode === "local" ? "off" : ""}>
              <dt>фильтр рынка · 13</dt>
              <dd>
                {filterKeys.map((s) => (
                  <span key={s.service}>
                    <i className={`dot ${srcDot(s)}`} aria-hidden="true" />
                    {s.name}
                  </span>
                ))}
                <span className="pw-free">+ {MARKET.filter((m) => !m.key).length} сами{fg ? ` · страх ${fg.value}` : ""}</span>
              </dd>
            </div>
          </dl>
        </button>
        <div className={`pw-chr${chronos && mode === "hybrid" ? "" : " off"}`}>
          <span>
            <i>Chronos</i>
            {!chronos ? "выключен" : mode === "hybrid" ? "прогноз цены: +0.05 / −0.15" : "только в Hybrid"}
          </span>
          <Toggle on={chronos} label="Прогноз Chronos" onChange={setChronos} />
        </div>
      </div>
      <span className="pw-tag" style={{ left: xOf(to) + NW / 2, top: (TOP + SH + ROW) / 2 }}>
        {mode === "dqn" ? "не для SAC" : to === "ai" ? "в AI" : "стратегия «Настроение»"}
      </span>
      <button type="button" className="pw-gate" style={{ left: xOf("risk") + 4, top: ROW - 20 }} onClick={() => openNode(to)} title="Порог — в карточке «Сигналы для решения»">
        {mode !== "local" && <span>фильтр рынка{chronos && mode === "hybrid" ? " · Chronos" : ""}</span>}
        <span>дальше, если уверенность ≥ {minConf}</span>
      </button>
      <div className={`pipe-node pw-tg pw-trust${trade ? "" : " idle"}`} style={{ left: xOf("sac"), top: TOP, width: NW, height: SH }}>
        <button type="button" className="pw-open" onClick={toChannels}>
          <b>Доверенные каналы</b>
          <span className="st">{!signed ? "нужен вход в Telegram" : trusted.length ? `${trusted.length} ${plural(trusted.length, ["канал", "канала", "каналов"])}` : "не выбраны"}</span>
          <span className="st">сделка по посту — мимо сигналов, стратегий и AI</span>
          {fresh && (
            <span className={`res ${fresh.ok ? "ok" : "bad"}`} title={fresh.ok ? undefined : fresh.reason}>
              <i aria-hidden="true">{fresh.ok ? "▲" : "▼"}</i> @{fresh.channel}
              {fresh.symbol ? ` · ${fresh.symbol.split("/")[0]}` : ""}
              {fresh.ok ? ` · ${fresh.latency_s} с` : " · без сделки"}
            </span>
          )}
        </button>
        <span className="pw-trust-sw">
          <Toggle on={on} label="Сделки по доверенным каналам" onChange={(v) => void actions.patchConfig({ TELEGRAM_TRUSTED_TRADING: v })} />
          {on ? "сделки вкл" : "сделки выкл"}
        </span>
      </div>
      <button type="button" className={`pipe-node pw-tg${trade ? "" : " idle"}`} style={{ left: xOf("risk"), top: TOP, width: NW, height: TH }} onClick={toChannels}>
        <b>Лимиты каналов</b>
        <span className="st">
          риск {pct(config.TELEGRAM_TRUSTED_RISK?.value, 0.005)} · стоп {pct(config.TELEGRAM_TRUSTED_SL?.value, 0.03)} · {slots}{" "}
          {plural(slots, ["слот", "слота", "слотов"])}
        </span>
      </button>
    </>
  );
}
