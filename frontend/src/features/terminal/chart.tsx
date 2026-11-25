// Терминал: свечной график с уровнями позиции (lightweight-charts).

import {
  createChart,
  type IChartApi,
  type IPriceLine,
  type ISeriesApi,
  type SeriesMarker,
  type UTCTimestamp,
} from "lightweight-charts";
import { useEffect, useRef, useState } from "react";

import { api } from "../../lib/api";
import { base, price } from "../../lib/format";

import { reducedMotion } from "../../lib/platform";
import type { Candle, Journal, JournalTrade } from "../../lib/types";
import { useStore } from "../../store";

import { Seg } from "../../components/controls";

// ── График ────────────────────────────────────────────────────────────────────

const TZ = -new Date().getTimezoneOffset() * 60; // lightweight-charts рисует время в UTC

const BAR = 900;

/** Цвет токена темы: библиотеке графиков нужен готовый hex, а не var(). */
const tok = (name: string): string =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** Уровень плавно доезжает до новой цены (трейлинг подтянул стоп) — без рывка. */
function glide(
  line: IPriceLine,
  from: number,
  to: number,
  onStep: (v: number) => void,
): () => void {
  if (from === to || reducedMotion()) {
    line.applyOptions({ price: to });
    onStep(to);
    return () => undefined;
  }
  const t0 = performance.now();
  let raf = 0;
  const step = (t: number) => {
    const k = Math.min(1, (t - t0) / 600);
    const v = from + (to - from) * (1 - (1 - k) ** 3);
    line.applyOptions({ price: v });
    onStep(v);
    if (k < 1) raf = requestAnimationFrame(step);
  };
  raf = requestAnimationFrame(step);
  return () => cancelAnimationFrame(raf);
}

export function Chart() {
  const positions = useStore((s) => s.positions);
  const syms = Object.keys(positions);
  const [sel, setSel] = useState<string>(syms[0] ?? "BTC/USDT");
  const sym = positions[sel] || !syms.length ? sel : syms[0];
  const box = useRef<HTMLDivElement>(null);
  const chart = useRef<{
    c: IChartApi;
    s: ISeriesApi<"Candlestick">;
    fill: ISeriesApi<"Baseline">;
  } | null>(null);
  const bars = useRef<Candle[]>([]);
  const lines = useRef<{
    sl: IPriceLine;
    tp: IPriceLine;
    entry: IPriceLine;
  } | null>(null);
  const shown = useRef({ sl: 0, tp: 0 });
  const [empty, setEmpty] = useState(false);
  const pos = positions[sym];
  const posRef = useRef(pos);
  posRef.current = pos;

  // Заливка между входом и ценой: своя сторона — бирюзой, против позиции — красным.
  const paintFill = () => {
    const ch = chart.current;
    const p = posRef.current;
    if (!ch) return;
    if (!p) return ch.fill.setData([]);
    const [good, bad] = [`${tok("--up")}30`, `${tok("--down")}30`];
    const [top, bottom] = p.side === "long" ? [good, bad] : [bad, good];
    ch.fill.applyOptions({
      baseValue: { type: "price", price: p.entry },
      topFillColor1: top,
      topFillColor2: top,
      bottomFillColor1: bottom,
      bottomFillColor2: bottom,
    });
    // Только время жизни позиции: до открытия заливать нечего. Нет даты — нет заливки.
    const from = p.opened_at
      ? Math.floor(p.opened_at / BAR) * BAR + TZ
      : Infinity;
    ch.fill.setData(
      bars.current
        .filter((b) => b.time >= from)
        .map((b) => ({ time: b.time as UTCTimestamp, value: b.close })),
    );
  };

  // Прошлые сделки по монете, как во FreqUI: вход — стрелка, выход — кружок с итогом.
  const markTrades = (list: JournalTrade[]) => {
    const ch = chart.current;
    const all = bars.current;
    if (!ch || !all.length) return;
    const [first, last] = [all[0].time, all[all.length - 1].time];
    const out: SeriesMarker<UTCTimestamp>[] = [];
    const add = (sec: number, m: Omit<SeriesMarker<UTCTimestamp>, "time">) => {
      const t = Math.floor(sec / BAR) * BAR + TZ;
      if (Number.isFinite(t) && t >= first && t <= last) out.push({ ...m, time: t as UTCTimestamp });
    };
    for (const t of list) {
      const long = t.side === "long";
      add(Date.parse(t.opened) / 1000, {
        position: long ? "belowBar" : "aboveBar",
        shape: long ? "arrowUp" : "arrowDown",
        color: tok("--cyan"),
        text: long ? "LONG" : "SHORT",
      });
      const win = t.pnl >= 0;
      add(Date.parse(t.closed) / 1000, {
        position: long ? "aboveBar" : "belowBar",
        shape: "circle",
        color: tok(win ? "--up" : "--down"),
        text: `${win ? "▲" : "▼"} ${(t.pnl_pct * 100).toFixed(1)}%`,
      });
    }
    const p = posRef.current;
    if (p?.opened_at)
      add(p.opened_at, {
        position: p.side === "long" ? "belowBar" : "aboveBar",
        shape: p.side === "long" ? "arrowUp" : "arrowDown",
        color: tok("--brand"),
        text: "открыта",
      });
    ch.s.setMarkers(out.sort((a, b) => a.time - b.time));
  };

  useEffect(() => {
    if (!box.current) return;
    const c = createChart(box.current, {
      autoSize: true,
      layout: {
        background: { color: tok("--panel") },
        textColor: tok("--ink-2"),
        fontFamily: "'JetBrains Mono', ui-monospace, monospace",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: tok("--line-soft") },
        horzLines: { color: tok("--line-soft") },
      },
      rightPriceScale: { borderColor: tok("--line") },
      timeScale: {
        borderColor: tok("--line"),
        timeVisible: true,
        secondsVisible: false,
      },
      crosshair: { mode: 0 },
      localization: { priceFormatter: price, locale: "ru-RU" },
    });
    // Заливка — первой, чтобы свечи рисовались поверх.
    const fill = c.addBaselineSeries({
      lineWidth: 1,
      topLineColor: "transparent",
      bottomLineColor: "transparent",
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    const [up, down] = [tok("--up"), tok("--down")];
    const s = c.addCandlestickSeries({
      upColor: up,
      downColor: down,
      borderVisible: false,
      wickUpColor: up,
      wickDownColor: down,
    });
    chart.current = { c, s, fill };
    return () => {
      c.remove();
      chart.current = null;
    };
  }, []);

  // История свечей из кэша бота при смене монеты.
  useEffect(() => {
    let stale = false;
    bars.current = [];
    void api<Candle[]>(
      "GET",
      `/api/candles?symbol=${encodeURIComponent(sym)}&tf=15m&limit=200`,
    )
      .then((data) => {
        if (stale || !chart.current) return;
        bars.current = data.map((b) => ({ ...b, time: b.time + TZ }));
        chart.current.s.setData(
          bars.current.map((b) => ({ ...b, time: b.time as UTCTimestamp })),
        );
        paintFill();
        chart.current.c.timeScale().fitContent();
        setEmpty(!data.length);
        void api<Journal>("GET", "/api/journal?period=7d")
          .then((j) => !stale && markTrades(j.trades.filter((t) => t.symbol === sym)))
          .catch(() => undefined);
      })
      .catch(() => !stale && setEmpty(true));
    return () => {
      stale = true;
    };
  }, [sym]);

  // Живая цена дорисовывает последнюю свечу.
  const live = pos?.price;
  useEffect(() => {
    const ch = chart.current;
    const all = bars.current;
    if (!ch || live === undefined || !all.length) return;
    const t = Math.floor(Date.now() / 1000 / BAR) * BAR + TZ;
    let last = all[all.length - 1];
    if (t > last.time) {
      last = { time: t, open: last.close, high: live, low: live, close: live };
      all.push(last);
    } else
      Object.assign(last, {
        close: live,
        high: Math.max(last.high, live),
        low: Math.min(last.low, live),
      });
    ch.s.update({ ...last, time: last.time as UTCTimestamp });
    if (posRef.current?.opened_at)
      ch.fill.update({ time: last.time as UTCTimestamp, value: live });
  }, [live]);

  // Уровни позиции: SL, TP и вход — заново при смене монеты или позиции.
  const has = Boolean(pos);
  useEffect(() => {
    const ch = chart.current;
    if (!ch) return;
    const old = lines.current;
    if (old)
      [old.sl, old.tp, old.entry].forEach((l) => ch.s.removePriceLine(l));
    lines.current = null;
    paintFill();
    const p = posRef.current;
    if (!p) return;
    const line = (v: number, color: string, title: string, dashed: boolean) =>
      ch.s.createPriceLine({
        price: v,
        color,
        lineWidth: 1,
        lineStyle: dashed ? 2 : 0,
        axisLabelVisible: title !== "вход",
        title,
      });
    lines.current = {
      sl: line(p.sl, tok("--down"), "SL", true),
      tp: line(p.tp, tok("--up"), "TP", true),
      entry: line(p.entry, tok("--brand"), "вход", false),
    };
    shown.current = { sl: p.sl, tp: p.tp };
  }, [sym, has, pos?.entry, pos?.side]); // eslint-disable-line react-hooks/exhaustive-deps

  // Трейлинг или правка SL/TP: линии доезжают до нового уровня.
  useEffect(() => {
    const l = lines.current;
    if (!l || !pos) return;
    const stopSl = glide(
      l.sl,
      shown.current.sl,
      pos.sl,
      (v) => (shown.current.sl = v),
    );
    const stopTp = glide(
      l.tp,
      shown.current.tp,
      pos.tp,
      (v) => (shown.current.tp = v),
    );
    return () => {
      stopSl();
      stopTp();
    };
  }, [pos?.sl, pos?.tp]); // eslint-disable-line react-hooks/exhaustive-deps

  const tabs = syms.length ? syms : ["BTC/USDT"];
  return (
    <article className="card chart">
      <div className="card-h">
        <h2>График</h2>
        <Seg
          className="sm"
          value={sym}
          options={tabs.map((s) => [s, base(s)] as const)}
          onChange={setSel}
          label="Монета"
        />
        <div className="legend sp">
          <span>
            <i style={{ borderColor: "var(--down)" }} />
            SL
          </span>
          <span>
            <i style={{ borderColor: "var(--up)" }} />
            TP
          </span>
          <span>
            <i
              style={{ borderColor: "var(--brand)", borderTopStyle: "solid" }}
            />
            вход
          </span>
          <span title="Прошлые сделки: стрелка — вход, кружок — выход с итогом">
            <b style={{ color: "var(--cyan)" }}>▲</b>
            сделки
          </span>
        </div>
      </div>
      <div className="chart-wrap">
        <div ref={box} style={{ position: "absolute", inset: 0 }} />
        {empty && (
          <div className="fallback">
            Нет сохранённых свечей {base(sym)} — бот ещё не загружал историю по
            этой монете.
          </div>
        )}
      </div>
    </article>
  );
}
