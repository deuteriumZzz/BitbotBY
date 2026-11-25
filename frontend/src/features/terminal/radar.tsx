// Терминал: радар монет — где монета в цикле (скан → сигнал → отказ/вход).

import { useEffect, useLayoutEffect, useRef, useState } from "react";

import { arrow, base, plural } from "../../lib/format";

import { reducedMotion } from "../../lib/platform";
import type { Position } from "../../lib/types";
import { positionPnl, useStore, type Store } from "../../store";
import { profileName } from "../bots";

import { Q } from "../../ui";

const EASE = "cubic-bezier(.2,.8,.2,1)";

// ── Радар: монеты едут по конвейеру скан → отбор → сигнал → проверки → позиция ──

const LANE_CAP = 5;

const REJECT_SHOW_MS = 1300;

/** FLIP: чип с тем же символом плавно переезжает из старого места в новое, даже между колонками. */
function useFlip(ref: React.RefObject<HTMLElement>): void {
  const last = useRef(new Map<string, { x: number; y: number }>());
  useLayoutEffect(() => {
    const host = ref.current;
    if (!host) return;
    const rm = reducedMotion();
    const next = new Map<string, { x: number; y: number }>();
    host.querySelectorAll<HTMLElement>(".chip[data-sym]").forEach((c) => {
      // offsetLeft/Top не учитывают transform — замер не сбивается идущей анимацией.
      let x = 0;
      let y = 0;
      for (
        let el: HTMLElement | null = c;
        el && el !== host;
        el = el.offsetParent as HTMLElement | null
      ) {
        x += el.offsetLeft;
        y += el.offsetTop;
      }
      const k = c.dataset.sym!;
      next.set(k, { x, y });
      if (rm) return;
      const a = last.current.get(k);
      if (!a)
        c.animate(
          [
            { opacity: 0, transform: "scale(.6)" },
            { opacity: 1, transform: "none" },
          ],
          { duration: 420, easing: EASE },
        );
      else if (Math.abs(a.x - x) + Math.abs(a.y - y) > 0.5)
        c.animate(
          [
            { transform: `translate(${a.x - x}px,${a.y - y}px)` },
            { transform: "none" },
          ],
          { duration: 640, easing: EASE },
        );
    });
    last.current = next;
  });
}

const CYCLE_PILL: Record<string, [string, string]> = {
  scan: ["Полный скан", "brand live"],
  monitor: ["Мониторинг", "ok live"],
  paused: ["На паузе", "warn"],
  guard: ["Защита", "warn"],
  off: ["Нет связи", "bad"],
};

/** Что сейчас делает бот — одно слово для радара и верхней панели. */
export function cycleKind(
  s: Store,
): "off" | "paused" | "guard" | "monitor" | "scan" {
  if (!s.alive || s.cycle.state === "offline") return "off";
  if (s.bot?.paused || s.cycle.state === "paused") return "paused";
  if (Object.values(s.guards).some((g) => g.active)) return "guard";
  return s.cycle.state === "monitor" ? "monitor" : "scan";
}

// Угол луча монеты на экране радара — постоянный, чтобы засветка ехала к центру по прямой.
const bearing = (s: string): number =>
  [...s].reduce((h, c) => (h * 31 + c.charCodeAt(0)) % 360, 7);

const RING = [44, 35, 26, 17, 7]; // скан → отбор → сигнал → проверки → позиции

/** Экран радара: кольца — этапы конвейера, монеты — засветки, луч крутится, пока бот сканирует. */
function Scope(props: {
  lanes: string[][];
  side: (s: string) => string;
  rejected: Set<string>;
  open: Position[];
  on: boolean;
}) {
  const blips = [
    ...props.lanes.flatMap((list, lane) =>
      list.slice(0, 10).map((s) => ({
        s,
        lane,
        cls: props.rejected.has(s) ? "rej" : props.side(s),
      })),
    ),
    ...props.open.map((p) => ({
      s: p.symbol,
      lane: 4,
      cls: positionPnl(p).pnl >= 0 ? "win" : "loss",
    })),
  ];
  return (
    <svg
      className={`scope${props.on ? " on" : ""}`}
      viewBox="-50 -50 100 100"
      aria-hidden="true"
    >
      {RING.slice(0, 4).map((r) => (
        <circle key={r} className="sc-ring" r={r} />
      ))}
      <path className="sc-cross" d="M-47 0H47M0-47V47" />
      {Array.from({ length: 36 }, (_, i) => (
        <line
          key={i}
          className="sc-deg"
          x1="0"
          y1={i % 3 ? -46 : -43}
          x2="0"
          y2="-48"
          style={{ transform: `rotate(${i * 10}deg)` }}
        />
      ))}
      <g className="sc-beam">
        <path className="sc-wedge" d="M0 0L0-46A46 46 0 0 0-26.4-37.7Z" />
        <line className="sc-ray" x1="0" y1="0" x2="0" y2="-46" />
      </g>
      {blips.map((b) => (
        <g
          key={b.s}
          className={`sc-blip ${b.cls} l${b.lane}`}
          style={{
            transform: `rotate(${bearing(b.s)}deg) translateY(${-RING[b.lane]}px)`,
          }}
        >
          <circle r={b.lane >= 2 ? 2.4 : 1.7} />
        </g>
      ))}
    </svg>
  );
}

export function Radar() {
  const radar = useStore((s) => s.radar);
  const positions = useStore((s) => s.positions);
  const slots = useStore((s) => s.slots);
  const kind = useStore(cycleKind);
  const profile = useStore((s) => s.bot?.profile);
  const mode = useStore((s) => s.bot?.mode);
  // Кто из AI-провайдеров ответил последним — подпись колонки «Сигнал».
  const aiName = useStore((s) => {
    const id = s.ai?.last || s.bot?.ai;
    return id ? (s.ai?.providers.find((p) => p.id === id)?.name ?? id) : "";
  });
  const host = useRef<HTMLDivElement>(null);
  const [, setNow] = useState(0);

  // Пока есть «свежие» отказы — перерисовываем, чтобы они исчезли вовремя.
  const fresh = Object.values(radar.rejected).some(
    (r) => Date.now() - r.at < REJECT_SHOW_MS + 600,
  );
  useEffect(() => {
    if (!fresh) return;
    const t = setInterval(() => setNow(Date.now()), 400);
    return () => clearInterval(t);
  }, [fresh, radar]);

  const open = Object.values(positions);
  const openSyms = new Set(open.map((p) => p.symbol));
  // "" — обычный чип, "rej" — только что отсеян, "gone" — уже не показываем.
  const rej = (s: string, lane: number) => {
    const r = radar.rejected[s];
    return r
      ? r.lane === lane && Date.now() - r.at < REJECT_SHOW_MS
        ? "rej"
        : "gone"
      : "";
  };
  // Монета уже ушла дальше по конвейеру — в этой колонке её не рисуем.
  const later = (s: string, from: number) =>
    openSyms.has(s) ||
    (from < 3 && radar.guards.includes(s)) ||
    (from < 2 && Boolean(radar.signal[s])) ||
    (from < 1 && radar.prescore.includes(s)) ||
    Boolean(radar.rejected[s] && radar.rejected[s].lane !== from);

  const laneSyms = [
    radar.scan.filter((s) => !later(s, 0)),
    radar.prescore.filter((s) => !later(s, 1)),
    Object.keys(radar.signal).filter((s) => !later(s, 2)),
    [
      ...radar.guards,
      ...Object.keys(radar.rejected).filter(
        (s) => radar.rejected[s].lane === 3 && !radar.guards.includes(s),
      ),
    ].filter((s) => !openSyms.has(s)),
  ].map((list, lane) => list.filter((s) => rej(s, lane) !== "gone"));

  useFlip(host);
  const nSlots = Math.max(Math.min(slots.max, 6), open.length);
  const [pillText, pillDot] = CYCLE_PILL[kind];
  const heads: [string, string][] = [
    [
      "Скан",
      `${radar.scan.length} ${plural(radar.scan.length, ["монета", "монеты", "монет"])}`,
    ],
    ["Отбор", `${radar.prescore.length} в топе`],
    [
      "Сигнал",
      mode === "local"
        ? "локальные стратегии"
        : aiName && aiName !== "local"
          ? `AI: ${aiName}`
          : "стратегии + AI",
    ],
    ["Проверки", "риски"],
  ];
  const note =
    kind === "off"
      ? "Нет связи с ботом — радар заморожен на последних данных."
      : kind === "paused"
        ? "Бот на паузе: новые сделки не открываются. Стоп-лоссы и тейк-профиты открытых позиций работают на бирже."
        : kind === "monitor"
          ? `Все ${slots.max} ${plural(slots.max, ["слот", "слота", "слотов"])} заняты. Бот не ищет новые сделки и следит за открытыми.`
          : radar.note ||
            `Бот ищет сделки${profile ? ` среди монет профиля ${profileName(profile)}` : ""}.`;

  return (
    <article className={`card radar ${kind === "scan" ? "scan" : "halt"}`}>
      <div className="card-h">
        <h2>Радар монет</h2>
        <Q tip="Так бот выбирает сделку: сканирует рынок, оставляет лучшие монеты, ищет сигнал стратегий и AI, проверяет риски — и только потом открывает позицию." />
        <span className="pill sp">
          <span className={`dot ${pillDot}`} />
          {pillText}
        </span>
      </div>
      <div className="radar-top">
        <Scope
          lanes={laneSyms}
          side={(s) => radar.signal[s]?.side ?? ""}
          rejected={
            new Set(
              Object.keys(radar.rejected).filter(
                (s) => rej(s, radar.rejected[s].lane) === "rej",
              ),
            )
          }
          open={open}
          on={kind === "scan"}
        />
        <p className="radar-note" aria-live="polite">
          {note}
        </p>
      </div>
      <div className="lanes" ref={host}>
        {laneSyms.map((list, lane) => (
          <div className="lane" key={lane}>
            <div className="lane-h">
              <b>{heads[lane][0]}</b>
              <span>{heads[lane][1]}</span>
            </div>
            <div className="lane-body">
              {list.slice(0, LANE_CAP).map((s) => {
                const sig = lane >= 2 ? radar.signal[s] : undefined;
                return (
                  <div
                    key={s}
                    data-sym={s}
                    className={`chip ${sig ? sig.side : ""} ${rej(s, lane)}`}
                    title={radar.rejected[s]?.reason ?? sig?.reasoning}
                  >
                    <span>{base(s)}</span>
                    <i>
                      {sig
                        ? `${sig.side === "long" ? "L" : "S"} ${sig.confidence?.toFixed(2).slice(1) ?? ""}`
                        : ""}
                    </i>
                  </div>
                );
              })}
              {list.length > LANE_CAP && (
                <span className="more">+{list.length - LANE_CAP}</span>
              )}
            </div>
          </div>
        ))}
        <div className="lane lane-pos">
          <div className="lane-h">
            <b>Позиции</b>
            <span>
              {open.length}/{slots.max}
            </span>
          </div>
          <div className="lane-body">
            {Array.from({ length: nSlots }, (_, i) => {
              const p = open[i];
              if (!p)
                return (
                  <div className="slot" key={`free${i}`}>
                    свободно
                  </div>
                );
              const { pnl } = positionPnl(p);
              return (
                <div className="slot full" key={p.symbol}>
                  <div
                    data-sym={p.symbol}
                    className={`chip ${pnl >= 0 ? "win" : "loss"}`}
                  >
                    <span>{base(p.symbol)}</span>
                    <i>{arrow(pnl)}</i>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </article>
  );
}
