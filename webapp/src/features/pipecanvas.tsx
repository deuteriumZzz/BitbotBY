// Холст «Цепочки»: провода между шагами, развилка «кто решает», дорожка доверенных каналов
// Telegram и прогон последнего цикла. Шаги с их полосами блоков рисует pipeline.tsx — здесь
// они лежат в мире, который масштабируется (как схема AI) и прокручивается вбок.
// Порядок шагов зашит в боте, поэтому сами шаги не двигаются: перетаскивается только штекер
// провода, который ведёт в «Риск», — это и есть «кто решает» (MODE, с «Отменить»).

import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";

import { plural } from "../lib/format";
import { reducedMotion } from "../lib/platform";
import type { BotNode, CycleTrace } from "../lib/types";
import { actions, useStore } from "../store";
import { Toggle } from "../ui";
import { cfg } from "./bots";
import { chanList } from "./settings";
import { scrollWhenReady } from "./terminal";

export const NW = 150; // ширина шага
const GX = 184; // шаг колонок
export const GAP = GX - NW;
export const X0 = 16;
const TOP = 14; // ряд доверенных каналов
const TH = 76;
export const ROW = TOP + TH + 74; // верх ряда шагов: над ним дуги развилки
const PY = ROW + 32; // высота портов шагов
const TY = TOP + TH / 2;
const TW = NW + 50; // узел «Доверенные каналы» шире — название в одну строку
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
              {edges.map(([a, b]) => (
                <path
                  key={`${mode}${a}${b}`}
                  className={`pw-wire${b === "risk" && plug ? " lifted" : ""}${a === "strat" && b === "ai" ? " hint" : ""}`}
                  d={wire(a, b)}
                  pathLength={1}
                />
              ))}
              {plug && <path className="pw-wire ghost" d={`M ${plug.x} ${plug.y} C ${plug.x + 60} ${plug.y}, ${xOf("risk") - 60} ${PY}, ${xOf("risk")} ${PY}`} />}
              <TrustedWires />
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

            <TrustedLane />

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

/** Провода доверенных каналов: мимо сканера, стратегий и AI — в «Лимиты каналов» и сразу в «Ордер». */
function TrustedWires() {
  const on = useStore((s) => cfg<boolean>(s.config, "TELEGRAM_TRUSTED_TRADING", false));
  const x2 = xOf("order") + NW / 2;
  const cls = `pw-wire tg${on ? "" : " off"}`;
  return (
    <>
      <path className={cls} d={`M ${xOf("scan") + TW} ${TY} L ${xOf("risk")} ${TY}`} />
      <path className={cls} d={`M ${xOf("risk") + NW} ${TY} C ${x2} ${TY}, ${x2} ${TY}, ${x2} ${ROW}`} />
    </>
  );
}

function TrustedLane() {
  const config = useStore((s) => s.config);
  const post = useStore((s) => s.trusted);
  const tg = useStore((s) => s.secrets?.find((x) => x.service === "telethon"));
  const setTab = useStore((s) => s.setTab);
  const on = cfg<boolean>(config, "TELEGRAM_TRUSTED_TRADING", false);
  const chans = chanList(cfg(config, "TELEGRAM_TRUSTED_CHANNELS", ""));
  const toChannels = () => (setTab("settings"), scrollWhenReady("tg-channels"));
  const fresh = post && Date.now() / 1000 - post.at < 86_400 ? post : null;
  const state = !tg?.signed_in
    ? "нужен вход в Telegram"
    : chans.length
      ? `${chans.length} ${plural(chans.length, ["канал", "канала", "каналов"])}: ${chans.slice(0, 2).map((c) => `@${c}`).join(", ")}${chans.length > 2 ? "…" : ""}`
      : "не выбраны";
  const slots = cfg<number>(config, "TELEGRAM_TRUSTED_SLOTS", 2);
  return (
    <>
      <button type="button" className={`pipe-node pw-tg${on ? "" : " idle"}`} style={{ left: xOf("scan"), top: TOP, width: TW, height: TH }} onClick={toChannels}>
        <b>Доверенные каналы</b>
        <span className="st">{state}</span>
        {fresh && (
          <span className={`res ${fresh.ok ? "ok" : "bad"}`} title={fresh.ok ? undefined : fresh.reason}>
            <i aria-hidden="true">{fresh.ok ? "▲" : "▼"}</i> @{fresh.channel}
            {fresh.symbol ? ` · ${fresh.symbol.split("/")[0]}` : ""}
            {fresh.ok ? ` · ${fresh.latency_s} с` : " · без сделки"}
          </span>
        )}
      </button>
      <div className="pw-tg-switch" style={{ left: (xOf("scan") + TW + xOf("risk")) / 2, top: TY }}>
        <span>{on ? "сделки по постам — мимо сканера, стратегий и AI" : "сделки по каналам выключены"}</span>
        <Toggle on={on} label="Сделки по доверенным каналам" onChange={(v) => void actions.patchConfig({ TELEGRAM_TRUSTED_TRADING: v })} />
      </div>
      <button type="button" className={`pipe-node pw-tg${on ? "" : " idle"}`} style={{ left: xOf("risk"), top: TOP, width: NW, height: TH }} onClick={toChannels}>
        <b>Лимиты каналов</b>
        <span className="st">
          риск {pct(config.TELEGRAM_TRUSTED_RISK?.value, 0.005)} · стоп {pct(config.TELEGRAM_TRUSTED_SL?.value, 0.03)} · {slots}{" "}
          {plural(slots, ["слот", "слота", "слотов"])}
        </span>
      </button>
    </>
  );
}
