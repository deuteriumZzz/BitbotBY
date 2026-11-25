// Календарь результатов по дням (терминал и журнал).

import { useMemo } from "react";

import { arrow, plural, tone, usd } from "../../lib/format";

import { useStore } from "../../store";

import { Q, hideTip, showTip } from "../../components/tooltip";
import { isoDay } from "./equity";

const DOW = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"];

/** Календарь P&L — живёт в «Журнале». */
export function Calendar() {
  const daily = useStore((s) => s.daily);
  const days = useMemo(() => {
    const byDate = new Map(daily.map((r) => [r.date, r]));
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const start = new Date(today);
    start.setDate(today.getDate() - 28 - ((today.getDay() + 6) % 7));
    return Array.from({ length: 35 }, (_, i) => {
      const d = new Date(start);
      d.setDate(start.getDate() + i);
      return {
        d,
        row: byDate.get(isoDay(d)),
        future: d > today,
        today: +d === +today,
      };
    });
  }, [daily]);
  const peak = Math.max(1, ...days.map((x) => Math.abs(x.row?.net ?? 0)));
  const show = (el: HTMLElement, i: number) => {
    const { d, row, future } = days[i];
    if (future) return;
    showTip(
      <>
        {d.toLocaleDateString("ru-RU", {
          weekday: "long",
          day: "numeric",
          month: "long",
        })}
        <br />
        {row?.trades ? (
          <>
            <b
              className={tone(row.net)}
            >{`${arrow(row.net)} ${usd(row.net)}`}</b>{" "}
            чистыми · {row.trades}{" "}
            {plural(row.trades, ["сделка", "сделки", "сделок"])}
          </>
        ) : (
          "сделок не было"
        )}
      </>,
      el.getBoundingClientRect(),
      el,
    );
  };
  const fmtD = (d: Date) =>
    d.toLocaleDateString("ru-RU", { day: "numeric", month: "short" });
  return (
    <article className="card cal">
      <div className="card-h">
        <h2>Календарь P&amp;L</h2>
        <Q tip="Клетка — день. Зелёная — день в плюсе, красная со штриховкой — в минусе. Чем ярче, тем больше сумма." />
        <span className="label sp">
          {fmtD(days[0].d)} — {fmtD(days[34].d)}
        </span>
      </div>
      <div className="cal-grid" onPointerLeave={hideTip} onBlur={hideTip}>
        {DOW.map((d) => (
          <span className="dow" key={d}>
            {d}
          </span>
        ))}
        {days.map(({ d, row, future, today }, i) => {
          const net = row?.net ?? 0;
          const has = !future && Boolean(row?.trades);
          // Яркость — от самого крупного дня в окне; потолок 60%, чтобы число в клетке читалось.
          const mix = Math.round(16 + (Math.abs(net) / peak) * 44);
          return (
            <span
              key={i}
              role="img"
              tabIndex={future ? -1 : 0}
              className={`cell${future ? " future" : ""}${today ? " today" : ""}${has && net < 0 ? " loss" : ""}`}
              style={
                has
                  ? {
                      backgroundColor: `color-mix(in oklab, var(${net >= 0 ? "--up" : "--down"}) ${mix}%, var(--cell0))`,
                    }
                  : undefined
              }
              aria-label={
                future
                  ? `${d.getDate()}: впереди`
                  : has
                    ? `${d.getDate()}: ${usd(net)}`
                    : `${d.getDate()}: сделок не было`
              }
              onPointerOver={(e) => show(e.currentTarget, i)}
              onFocus={(e) => show(e.currentTarget, i)}
            >
              {d.getDate()}
            </span>
          );
        })}
      </div>
      <div className="cal-legend">
        <span>убыток</span>
        <span className="ramp" />
        <span>прибыль</span>
      </div>
    </article>
  );
}
