// Терминал: кривая капитала (equity) за выбранный период.

import { useLayoutEffect, useMemo, useRef, useState } from "react";

import { arrow, fmtN, tone, usd } from "../../lib/format";

import { useStore } from "../../store";

import { hideTip, showTip } from "../../components/tooltip";

const DAY = 86_400_000;

export const isoDay = (d: Date): string =>
  `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;

/** Баланс на конец каждого из 30 дней: идём назад от текущего, вычитая P&L дня. */
function useEquity(): { d: Date; b: number }[] {
  const balance = useStore((s) => s.balance);
  const daily = useStore((s) => s.daily);
  return useMemo(() => {
    const byDate = new Map(daily.map((r) => [r.date, r.net]));
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const out: { d: Date; b: number }[] = [];
    let b = balance;
    for (let i = 0; i < 30; i++) {
      const d = new Date(today.getTime() - i * DAY);
      out.push({ d, b });
      b -= byDate.get(isoDay(d)) ?? 0;
    }
    return out.reverse();
  }, [balance, daily]);
}

export function Equity() {
  const pts = useEquity();
  const ref = useRef<SVGSVGElement>(null);
  const [W, setW] = useState(320);
  const [hover, setHover] = useState<number | null>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(Math.max(200, el.clientWidth)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const H = 88;
  const n = pts.length;
  const bs = pts.map((p) => p.b);
  // Шкала не уже 5% от баланса: иначе изменение на 2% рисуется обрывом во всю высоту.
  const lo = Math.min(...bs);
  const hi = Math.max(...bs);
  const span = Math.max(hi - lo, Math.abs(hi) * 0.05 || 1);
  const mn = (hi + lo) / 2 - span / 2;
  const mx = mn + span;
  const x = (i: number) => (i * (W - 8)) / (n - 1) + 4;
  const y = (v: number) => H - 8 - ((v - mn) / (mx - mn || 1)) * (H - 20);
  const line = pts
    .map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.b).toFixed(1)}`)
    .join("");
  const area = `${line}L${x(n - 1).toFixed(1)},${H}L${x(0).toFixed(1)},${H}Z`;
  const onMove = (ev: React.PointerEvent) => {
    const r = ref.current!.getBoundingClientRect();
    const i = Math.max(
      0,
      Math.min(n - 1, Math.round(((ev.clientX - r.left) / r.width) * (n - 1))),
    );
    const p = pts[i];
    const chg = i ? p.b - pts[i - 1].b : 0;
    setHover(i);
    showTip(
      <>
        {p.d.toLocaleDateString("ru-RU", {
          weekday: "short",
          day: "numeric",
          month: "long",
        })}
        <br />
        <b>${fmtN(p.b)}</b>{" "}
        <b className={tone(chg)}>{`${arrow(chg)} ${usd(chg)}`}</b>
      </>,
      {
        left: r.left + (x(i) / W) * r.width - 1,
        width: 2,
        top: r.top,
        bottom: r.top + (y(p.b) / H) * r.height,
      },
      ref.current,
    );
  };
  return (
    <div className="eq">
      <svg
        ref={ref}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Баланс за 30 дней"
      >
        <defs>
          <linearGradient id="eq-fade" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" className="eq-fade-a" />
            <stop offset="1" className="eq-fade-b" />
          </linearGradient>
        </defs>
        {/* Шкала с делениями по дням — как у самописца. */}
        {pts.map((_, i) => (
          <line
            key={i}
            className="eq-tick"
            x1={x(i)}
            x2={x(i)}
            y1={H - (i % 7 ? 3 : 6)}
            y2={H}
          />
        ))}
        <line
          className="eq-base"
          x1="4"
          x2={W - 4}
          y1={y(bs[0])}
          y2={y(bs[0])}
        />
        {/* style.d — чтобы кривая перетекала при обновлении (CSS-переход d; где не поддержан — просто сменится). */}
        <path
          className="eq-area"
          d={area}
          style={{ d: `path("${area}")` } as React.CSSProperties}
        />
        <path
          className="eq-line"
          d={line}
          pathLength={1}
          style={{ d: `path("${line}")` } as React.CSSProperties}
        />
        <circle className="eq-now" cx={x(n - 1)} cy={y(bs[n - 1])} r="3.5" />
        {hover !== null && (
          <g>
            <line
              className="eq-cross"
              x1={x(hover)}
              x2={x(hover)}
              y1="0"
              y2={H}
            />
            <circle className="eq-hov" cx={x(hover)} cy={y(bs[hover])} r="4" />
          </g>
        )}
        <rect
          width={W}
          height={H}
          fill="transparent"
          onPointerMove={onMove}
          onPointerLeave={() => {
            setHover(null);
            hideTip();
          }}
        />
      </svg>
      <div className="eq-cap">
        <span>
          {pts[0].d.toLocaleDateString("ru-RU", {
            day: "numeric",
            month: "short",
          })}
        </span>
        <span>баланс за 30 дней</span>
        <span>сегодня</span>
      </div>
    </div>
  );
}
