// Вкладка «Терминал»: баланс, метрики, радар монет, живые позиции, график, лента, календарь.

import { createChart, type IChartApi, type IPriceLine, type ISeriesApi, type UTCTimestamp } from "lightweight-charts";
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type FormEvent } from "react";

import { api } from "../lib/api";
import { arrow, base, fmtN, hhmm, pct, plural, price, priceDecimals, tone, usd } from "../lib/format";
import { reducedMotion } from "../lib/platform";
import type { Candle, Position } from "../lib/types";
import { actions, positionPnl, useStore, type Store } from "../store";
import { profileName, stratName } from "./bots";
import { Flash, Q, Seg, Tween, hideTip, showTip, useArm } from "../ui";

const EASE = "cubic-bezier(.2,.8,.2,1)";
const PERIODS = [
  ["24h", "24ч"],
  ["7d", "7д"],
  ["30d", "30д"],
  ["all", "Всё"],
] as const;

export function Terminal() {
  return (
    <section className="view term enter" aria-label="Терминал">
      <Hero />
      <MetricsCard />
      <Radar />
      <Positions />
      <Chart />
      <Feed />
      <Calendar />
    </section>
  );
}

// ── Баланс и кривая equity ────────────────────────────────────────────────────

function Hero() {
  const balance = useStore((s) => s.balance);
  const paper = useStore((s) => s.bot?.paper ?? true);
  const period = useStore((s) => s.period);
  const setPeriod = useStore((s) => s.setPeriod);
  const m = useStore((s) => s.metrics[s.period]);
  const positions = useStore((s) => s.positions);
  const list = Object.values(positions);
  const unreal = list.reduce((sum, p) => sum + positionPnl(p).pnl, 0);
  return (
    <article className="card hero">
      <div className="card-h">
        <span className="label">Баланс · {paper ? "Paper" : "Live"}</span>
        <Seg className="sm sp" value={period} options={PERIODS} onChange={setPeriod} label="Период" />
      </div>
      <div className="bal">
        <Tween value={balance} fmt={(v) => `$${fmtN(v)}`} />
      </div>
      <div className="hero-stats">
        <div>
          <span className="label">
            Закрытые <Q tip="Прибыль и убыток по уже закрытым сделкам за выбранный период, до вычета комиссий." />
          </span>
          <b className={m ? tone(m.realized) : ""}>{m ? <Tween value={m.realized} fmt={(v) => `${arrow(v)} ${usd(v)}`} /> : "—"}</b>
          <small>{m ? `${m.n} ${plural(m.n, ["сделка", "сделки", "сделок"])}` : "—"}</small>
        </div>
        <div>
          <span className="label">
            Открытые <Q tip="Сколько вы получили бы, если закрыть открытые позиции прямо сейчас. Меняется вместе с ценой." />
          </span>
          <b className={tone(unreal)}>{list.length ? `${arrow(unreal)} ${usd(unreal)}` : "$0.00"}</b>
          <small>
            {list.length} {plural(list.length, ["позиция", "позиции", "позиций"])}
          </small>
        </div>
        <div>
          <span className="label">
            Чистыми <Q tip="Закрытые сделки минус комиссии биржи. Это то, что реально осталось на счёте." />
          </span>
          <b className={m ? tone(m.net) : ""}>{m ? <Tween value={m.net} fmt={(v) => `${arrow(v)} ${usd(v)}`} /> : "—"}</b>
          <small>{m ? `комиссии −$${fmtN(m.fees)}` : "—"}</small>
        </div>
      </div>
      <Equity />
    </article>
  );
}

const DAY = 86_400_000;
const isoDay = (d: Date): string =>
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

function Equity() {
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
  const mn = Math.min(...bs);
  const mx = Math.max(...bs);
  const x = (i: number) => (i * (W - 8)) / (n - 1) + 4;
  const y = (v: number) => H - 8 - ((v - mn) / (mx - mn || 1)) * (H - 20);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.b).toFixed(1)}`).join("");
  const onMove = (ev: React.PointerEvent) => {
    const r = ref.current!.getBoundingClientRect();
    const i = Math.max(0, Math.min(n - 1, Math.round(((ev.clientX - r.left) / r.width) * (n - 1))));
    const p = pts[i];
    const chg = i ? p.b - pts[i - 1].b : 0;
    setHover(i);
    showTip(
      <>
        {p.d.toLocaleDateString("ru-RU", { weekday: "short", day: "numeric", month: "long" })}
        <br />
        <b>${fmtN(p.b)}</b> <b className={tone(chg)}>{`${arrow(chg)} ${usd(chg)}`}</b>
      </>,
      { left: r.left + (x(i) / W) * r.width - 1, width: 2, top: r.top, bottom: r.top + (y(p.b) / H) * r.height },
      ref.current,
    );
  };
  return (
    <div className="eq">
      <svg ref={ref} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Баланс за 30 дней">
        <defs>
          <linearGradient id="eqg" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#7000FF" stopOpacity=".38" />
            <stop offset="1" stopColor="#7000FF" stopOpacity="0" />
          </linearGradient>
        </defs>
        <line x1="4" x2={W - 4} y1={y(bs[0])} y2={y(bs[0])} stroke="#2B3139" strokeDasharray="3 4" />
        <path d={`${line}L${x(n - 1)},${H}L${x(0)},${H}Z`} fill="url(#eqg)" />
        <path d={line} fill="none" stroke="#A47CFF" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
        <circle cx={x(n - 1)} cy={y(bs[n - 1])} r="4" fill="#A47CFF" stroke="#181A20" strokeWidth="2" />
        {hover !== null && (
          <g>
            <line x1={x(hover)} x2={x(hover)} y1="0" y2={H} stroke="#6C7480" />
            <circle cx={x(hover)} cy={y(bs[hover])} r="4" fill="#EAECEF" stroke="#181A20" strokeWidth="2" />
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
        <span>{pts[0].d.toLocaleDateString("ru-RU", { day: "numeric", month: "short" })}</span>
        <span>баланс за 30 дней</span>
        <span>сегодня</span>
      </div>
    </div>
  );
}

// ── Метрики ───────────────────────────────────────────────────────────────────

function MetricsCard() {
  const m = useStore((s) => s.metrics[s.period]);
  if (!m) return <article className="card metrics" aria-busy="true" />;
  const wr = m.win_rate ?? 0;
  const aw = m.avg_win ?? 0;
  const al = Math.abs(m.avg_loss ?? 0);
  const tot = aw + al || 1;
  return (
    <article className="card metrics">
      <div className="tile">
        <div className="t-h">
          <span>Доля прибыльных</span>
          <Q tip="Win rate — сколько сделок закрылись в плюс. 40% — нормально, если прибыльные сделки крупнее убыточных." />
        </div>
        <div className="v">{m.win_rate === null ? "—" : <Tween value={wr * 100} fmt={(v) => `${fmtN(v, 1)}%`} />}</div>
        <div className="bar">
          <i style={{ background: "var(--up)", width: `${wr * 100}%` }} />
        </div>
        <small>
          {Math.round(wr * m.n)} из {m.n} {plural(m.n, ["сделки", "сделок", "сделок"])}
        </small>
      </div>
      <div className="tile">
        <div className="t-h">
          <span>Profit factor</span>
          <Q tip="Сумма всех прибылей, делённая на сумму всех убытков. Больше 1 — в плюсе, меньше 1 — в минусе." />
        </div>
        <div className={`v ${m.profit_factor === null ? "" : m.profit_factor >= 1 ? "up" : "down"}`}>
          {m.profit_factor === null ? "—" : <Tween value={m.profit_factor} fmt={(v) => fmtN(v, 2)} />}
        </div>
        <small>
          {m.profit_factor === null ? "нет убыточных сделок" : m.profit_factor >= 1 ? "прибыли больше убытков" : "убытки больше прибылей"}
        </small>
      </div>
      <div className="tile">
        <div className="t-h">
          <span>Средняя сделка</span>
          <Q tip="Средний размер прибыльной и убыточной сделки. Полоски показывают их соотношение." />
        </div>
        <div className="v small">
          <span className="up">{m.avg_win === null ? "—" : `+$${fmtN(aw, 1)}`}</span> /{" "}
          <span className="down">{m.avg_loss === null ? "—" : `−$${fmtN(al, 1)}`}</span>
        </div>
        <div className="bar">
          <i style={{ background: "var(--up)", width: `${(aw / tot) * 100}%` }} />
          <i style={{ background: "var(--down)", width: `${(al / tot) * 100}%` }} />
        </div>
      </div>
      <div className="tile">
        <div className="t-h">
          <span>Риск : прибыль</span>
          <Q tip="R:R — во сколько раз средняя прибыльная сделка больше средней убыточной." />
        </div>
        <div className="v">{m.rr === null ? "—" : <Tween value={m.rr} fmt={(v) => `1 : ${fmtN(v, 2)}`} />}</div>
        <small>прибыль на 1 $ риска</small>
      </div>
      <div className="tile">
        <div className="t-h">
          <span>Макс. просадка</span>
          <Q tip="Самое глубокое падение баланса от пика до дна за период. Столько бот уже терял в худшую полосу." />
        </div>
        <div className="v down">
          <Tween value={m.max_drawdown_pct} fmt={(v) => `−${fmtN(v, 1)}%`} />
        </div>
        <small>от пика баланса</small>
      </div>
      <div className="tile">
        <div className="t-h">
          <span>Коэф. Шарпа</span>
          <Q tip="Доходность с поправкой на её разброс. Считаем от 50 закрытых сделок: на меньшем числе цифра случайна." />
        </div>
        {m.sharpe === null ? (
          <>
            <div className="v small muted">мало данных</div>
            <small>{m.n} из 50 сделок, нужных для расчёта</small>
          </>
        ) : (
          <>
            <div className={`v ${tone(m.sharpe)}`}>
              <Tween value={m.sharpe} fmt={(v) => fmtN(v, 2)} />
            </div>
            <small>по {m.n} сделкам</small>
          </>
        )}
      </div>
    </article>
  );
}

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
      for (let el: HTMLElement | null = c; el && el !== host; el = el.offsetParent as HTMLElement | null) {
        x += el.offsetLeft;
        y += el.offsetTop;
      }
      const k = c.dataset.sym!;
      next.set(k, { x, y });
      if (rm) return;
      const a = last.current.get(k);
      if (!a) c.animate([{ opacity: 0, transform: "scale(.6)" }, { opacity: 1, transform: "none" }], { duration: 420, easing: EASE });
      else if (Math.abs(a.x - x) + Math.abs(a.y - y) > 0.5)
        c.animate([{ transform: `translate(${a.x - x}px,${a.y - y}px)` }, { transform: "none" }], { duration: 640, easing: EASE });
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
export function cycleKind(s: Store): "off" | "paused" | "guard" | "monitor" | "scan" {
  if (!s.alive || s.cycle.state === "offline") return "off";
  if (s.bot?.paused || s.cycle.state === "paused") return "paused";
  if (Object.values(s.guards).some((g) => g.active)) return "guard";
  return s.cycle.state === "monitor" ? "monitor" : "scan";
}

function Radar() {
  const radar = useStore((s) => s.radar);
  const positions = useStore((s) => s.positions);
  const slots = useStore((s) => s.slots);
  const kind = useStore(cycleKind);
  const profile = useStore((s) => s.bot?.profile);
  // Кто из AI-провайдеров ответил последним — подпись колонки «Сигнал».
  const aiName = useStore((s) => {
    const id = s.ai?.last || s.bot?.ai;
    return id ? (s.ai?.providers.find((p) => p.id === id)?.name ?? id) : "";
  });
  const host = useRef<HTMLDivElement>(null);
  const [, setNow] = useState(0);

  // Пока есть «свежие» отказы — перерисовываем, чтобы они исчезли вовремя.
  const fresh = Object.values(radar.rejected).some((r) => Date.now() - r.at < REJECT_SHOW_MS + 600);
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
    return r ? (r.lane === lane && Date.now() - r.at < REJECT_SHOW_MS ? "rej" : "gone") : "";
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
    [...radar.guards, ...Object.keys(radar.rejected).filter((s) => radar.rejected[s].lane === 3 && !radar.guards.includes(s))].filter(
      (s) => !openSyms.has(s),
    ),
  ].map((list, lane) => list.filter((s) => rej(s, lane) !== "gone"));

  useFlip(host);
  const nSlots = Math.max(Math.min(slots.max, 6), open.length);
  const [pillText, pillDot] = CYCLE_PILL[kind];
  const heads: [string, string][] = [
    ["Скан", `${radar.scan.length} ${plural(radar.scan.length, ["монета", "монеты", "монет"])}`],
    ["Отбор", `${radar.prescore.length} в топе`],
    ["Сигнал", aiName && aiName !== "local" ? `AI: ${aiName}` : "стратегии + AI"],
    ["Проверки", "риски"],
  ];
  const note =
    kind === "off"
      ? "Нет связи с ботом — радар заморожен на последних данных."
      : kind === "paused"
        ? "Бот на паузе: новые сделки не открываются. Стоп-лоссы и тейк-профиты открытых позиций работают на бирже."
        : kind === "monitor"
          ? `Все ${slots.max} ${plural(slots.max, ["слот", "слота", "слотов"])} заняты. Бот не ищет новые сделки и следит за открытыми.`
          : radar.note || `Бот ищет сделки${profile ? ` среди монет профиля ${profileName(profile)}` : ""}.`;

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
                  <div key={s} data-sym={s} className={`chip ${sig ? sig.side : ""} ${rej(s, lane)}`} title={radar.rejected[s]?.reason ?? sig?.reasoning}>
                    <span>{base(s)}</span>
                    <i>{sig ? `${sig.side === "long" ? "L" : "S"} ${sig.confidence?.toFixed(2).slice(1) ?? ""}` : ""}</i>
                  </div>
                );
              })}
              {list.length > LANE_CAP && <span className="more">+{list.length - LANE_CAP}</span>}
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
                  <div data-sym={p.symbol} className={`chip ${pnl >= 0 ? "win" : "loss"}`}>
                    <span>{base(p.symbol)}</span>
                    <i>{arrow(pnl)}</i>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
      <p className="radar-note" aria-live="polite">
        {note}
      </p>
    </article>
  );
}

// ── Живые позиции ─────────────────────────────────────────────────────────────

interface Leaving {
  p: Position;
  pnl: number;
}

function Positions() {
  const positions = useStore((s) => s.positions);
  const list = Object.values(positions);
  const prev = useRef(positions);
  const seen = useRef<Set<string> | null>(null);
  const [leaving, setLeaving] = useState<Leaving[]>([]);

  // Закрытая позиция ещё 1.15 с «уходит» с подсветкой результата.
  useEffect(() => {
    const gone = Object.values(prev.current).filter((p) => !positions[p.symbol]);
    prev.current = positions;
    if (!gone.length) return;
    const items = gone.map((p) => ({ p, pnl: positionPnl(p).pnl }));
    setLeaving((l) => [...l, ...items]);
    setTimeout(() => setLeaving((l) => l.filter((x) => !items.includes(x))), reducedMotion() ? 0 : 1150);
  }, [positions]);

  // Карточки при первом показе появляются без анимации, новые — со вспышкой.
  if (seen.current === null) seen.current = new Set(list.map((p) => p.symbol));
  const fresh = new Set(list.map((p) => p.symbol).filter((s) => !seen.current!.has(s)));
  useEffect(() => {
    seen.current = new Set(Object.keys(positions));
  }, [positions]);

  return (
    <article className="card positions">
      <div className="card-h">
        <h2>Открытые позиции</h2>
        <span className="pill sp">{list.length}</span>
      </div>
      <div className="pos-list">
        {list.map((p) => (
          <PositionCard key={p.symbol} p={p} enter={fresh.has(p.symbol)} />
        ))}
        {leaving.map(({ p, pnl }) => (
          <article key={`out-${p.symbol}-${p.trade_id}`} className={`pos leaving ${pnl >= 0 ? "win" : "loss"}`} aria-hidden="true">
            <div className="pos-h">
              <span className="coin">{base(p.symbol)}</span>
              <div className="pos-t">
                <b>{p.symbol}</b>
                <small>закрыта</small>
              </div>
              <div className="pos-pnl">
                <b className={tone(pnl)}>{`${arrow(pnl)} ${usd(pnl)}`}</b>
                <span className={`burst ${tone(pnl)}`}>{`${arrow(pnl)} ${usd(pnl)}`}</span>
              </div>
            </div>
          </article>
        ))}
        {!list.length && !leaving.length && <p className="empty">Открытых позиций нет. Бот ищет сделки — следите за радаром.</p>}
      </div>
    </article>
  );
}

const progOf = (p: Position, v: number) =>
  Math.max(0, Math.min(1, p.side === "long" ? (v - p.sl) / (p.tp - p.sl) : (p.sl - v) / (p.sl - p.tp)));

function PositionCard({ p, enter }: { p: Position; enter: boolean }) {
  const online = useStore((s) => s.alive);
  const { pnl, move } = positionPnl(p);
  const cur = p.price ?? p.entry;
  const e = progOf(p, p.entry) * 100;
  const mk = progOf(p, cur) * 100;
  const [closeArmed, armClose] = useArm();
  const [editing, setEditing] = useState(false);
  const [whyOpen, setWhyOpen] = useState(false);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [burst, setBurst] = useState<{ k: number; pnl: number } | null>(null);
  const prevQty = useRef(p.qty);

  // Частичное закрытие: всплывает зафиксированная часть P&L.
  useEffect(() => {
    if (p.qty < prevQty.current) {
      const part = (cur - p.entry) * (prevQty.current - p.qty) * (p.side === "long" ? 1 : -1);
      setBurst({ k: Date.now(), pnl: part });
    }
    prevQty.current = p.qty;
  }, [p.qty]); // eslint-disable-line react-hooks/exhaustive-deps

  const can = online && p.trade_id !== null && !busy;
  const run = async (fn: () => Promise<boolean>) => {
    setBusy(true);
    await fn();
    setBusy(false);
  };
  const onSave = (ev: FormEvent<HTMLFormElement>) => {
    ev.preventDefault();
    const f = new FormData(ev.currentTarget);
    const sl = parseFloat(String(f.get("sl")).replace(",", "."));
    const tp = parseFloat(String(f.get("tp")).replace(",", "."));
    const ok = p.side === "long" ? sl < cur && tp > cur : sl > cur && tp < cur;
    if (!Number.isFinite(sl) || !Number.isFinite(tp) || !ok) {
      setErr(
        p.side === "long"
          ? `Для LONG стоп должен быть ниже цены ${price(cur)}, а тейк — выше.`
          : `Для SHORT стоп должен быть выше цены ${price(cur)}, а тейк — ниже.`,
      );
      return;
    }
    setErr("");
    void run(async () => {
      const done = await actions.updatePosition(p, { sl, tp });
      if (done) setEditing(false);
      return done;
    });
  };

  return (
    <article className={`pos ${pnl >= 0 ? "win" : "loss"} ${enter ? "enter" : ""}`}>
      <div className="pos-h">
        <span className="coin">{base(p.symbol)}</span>
        <div className="pos-t">
          <b>
            {p.symbol}
            <span className={`side ${p.side}`}>{p.side === "long" ? "LONG" : "SHORT"}</span>
          </b>
          <small>
            {p.strategy ? `${stratName(p.strategy)}${p.confidence ? ` · уверенность ${p.confidence.toFixed(2)}` : ""}` : `сделка #${p.trade_id ?? "—"}`}
          </small>
        </div>
        <div className="pos-pnl">
          <b className={`num ${tone(pnl)}`}>{`${arrow(pnl)} ${usd(pnl)}`}</b>
          <small>{pct(move)}</small>
          {burst && (
            <span key={burst.k} className={`burst ${tone(burst.pnl)}`} onAnimationEnd={() => setBurst(null)}>
              {`${arrow(burst.pnl)} ${usd(burst.pnl)}`}
            </span>
          )}
        </div>
      </div>
      {p.reasoning && (
        <p className={`pos-why ${whyOpen ? "open" : ""}`} title={p.reasoning} onClick={() => setWhyOpen(!whyOpen)}>
          <em>Почему открыта:</em> {p.reasoning}
        </p>
      )}
      <div className="range">
        <span>
          SL <Flash value={p.sl} text={price(p.sl)} />
        </span>
        <div className="track">
          <i
            className="fill"
            style={{ left: `${Math.min(e, mk)}%`, width: `${Math.abs(mk - e)}%`, background: pnl >= 0 ? "var(--up)" : "var(--down)" }}
          />
          <i className="entry" style={{ left: `${e}%` }} />
          <i className="mark" style={{ left: `${mk}%` }} />
        </div>
        <span>
          TP <Flash value={p.tp} text={price(p.tp)} />
        </span>
      </div>
      <div className="pos-meta">
        <span>
          Вход <b className="num">{price(p.entry)}</b>
        </span>
        <span>
          Сейчас <Flash className="num" value={cur} text={price(cur)} />
        </span>
        <span>
          Объём <b className="num">${fmtN(p.entry * p.qty, 0)}</b>
        </span>
      </div>
      <div className="acts">
        <button
          className={`btn danger ${closeArmed ? "armed" : ""}`}
          type="button"
          disabled={!can}
          onClick={() => armClose() && void run(() => actions.closePosition(p))}
        >
          {closeArmed ? "Точно?" : "Закрыть"}
        </button>
        <button className="btn" type="button" aria-pressed={editing} disabled={!can} onClick={() => setEditing(!editing)}>
          SL / TP
        </button>
        <button
          className="btn"
          type="button"
          aria-pressed={p.trailing !== false}
          disabled={!can}
          title="Стоп-лосс подтягивается за ценой. Выключите, чтобы стоп стоял на месте."
          onClick={() => void run(() => actions.updatePosition(p, { trailing: p.trailing === false }))}
        >
          Трейлинг
        </button>
        <button className="btn" type="button" disabled={!can} onClick={() => void run(() => actions.closePosition(p, 0.5))}>
          Закрыть 50%
        </button>
      </div>
      {editing && (
        <form className="edit" noValidate onSubmit={onSave}>
          <label>
            Стоп-лосс
            <input className="inp" name="sl" inputMode="decimal" defaultValue={p.sl.toFixed(priceDecimals(p.sl))} autoFocus />
          </label>
          <label>
            Тейк-профит
            <input className="inp" name="tp" inputMode="decimal" defaultValue={p.tp.toFixed(priceDecimals(p.tp))} />
          </label>
          <button className="btn primary" type="submit" disabled={busy}>
            Сохранить
          </button>
          {err && <span className="err">{err}</span>}
        </form>
      )}
    </article>
  );
}

// ── График ────────────────────────────────────────────────────────────────────

const TZ = -new Date().getTimezoneOffset() * 60; // lightweight-charts рисует время в UTC
const BAR = 900;

function Chart() {
  const positions = useStore((s) => s.positions);
  const syms = Object.keys(positions);
  const [sel, setSel] = useState<string>(syms[0] ?? "BTC/USDT");
  const sym = positions[sel] || !syms.length ? sel : syms[0];
  const box = useRef<HTMLDivElement>(null);
  const chart = useRef<{ c: IChartApi; s: ISeriesApi<"Candlestick"> } | null>(null);
  const bars = useRef<Candle[]>([]);
  const lines = useRef<IPriceLine[]>([]);
  const [empty, setEmpty] = useState(false);
  const pos = positions[sym];

  useEffect(() => {
    if (!box.current) return;
    const c = createChart(box.current, {
      autoSize: true,
      layout: { background: { color: "#181A20" }, textColor: "#A1A9B5", fontFamily: "'JetBrains Mono', ui-monospace, monospace", fontSize: 11 },
      grid: { vertLines: { color: "rgba(43,49,57,.45)" }, horzLines: { color: "rgba(43,49,57,.45)" } },
      rightPriceScale: { borderColor: "#2B3139" },
      timeScale: { borderColor: "#2B3139", timeVisible: true, secondsVisible: false },
      crosshair: { mode: 0 },
      localization: { priceFormatter: price, locale: "ru-RU" },
    });
    const s = c.addCandlestickSeries({ upColor: "#0ECB81", downColor: "#F6465D", borderVisible: false, wickUpColor: "#0ECB81", wickDownColor: "#F6465D" });
    chart.current = { c, s };
    return () => {
      c.remove();
      chart.current = null;
    };
  }, []);

  // История свечей из кэша бота при смене монеты.
  useEffect(() => {
    let stale = false;
    bars.current = [];
    void api<Candle[]>("GET", `/api/candles?symbol=${encodeURIComponent(sym)}&tf=15m&limit=200`)
      .then((data) => {
        if (stale || !chart.current) return;
        bars.current = data.map((b) => ({ ...b, time: b.time + TZ }));
        chart.current.s.setData(bars.current.map((b) => ({ ...b, time: b.time as UTCTimestamp })));
        chart.current.c.timeScale().fitContent();
        setEmpty(!data.length);
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
    } else Object.assign(last, { close: live, high: Math.max(last.high, live), low: Math.min(last.low, live) });
    ch.s.update({ ...last, time: last.time as UTCTimestamp });
  }, [live]);

  // Уровни позиции: SL, TP и вход.
  useEffect(() => {
    const ch = chart.current;
    if (!ch) return;
    lines.current.forEach((l) => ch.s.removePriceLine(l));
    lines.current = pos
      ? [
          ch.s.createPriceLine({ price: pos.sl, color: "#F6465D", lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: "SL" }),
          ch.s.createPriceLine({ price: pos.tp, color: "#0ECB81", lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: "TP" }),
          ch.s.createPriceLine({ price: pos.entry, color: "#6C7480", lineWidth: 1, lineStyle: 0, axisLabelVisible: false, title: "вход" }),
        ]
      : [];
  }, [pos?.sl, pos?.tp, pos?.entry, sym]); // eslint-disable-line react-hooks/exhaustive-deps

  const tabs = syms.length ? syms : ["BTC/USDT"];
  return (
    <article className="card chart">
      <div className="card-h">
        <h2>График</h2>
        <Seg className="sm" value={sym} options={tabs.map((s) => [s, base(s)] as const)} onChange={setSel} label="Монета" />
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
            <i style={{ borderColor: "var(--ink-3)", borderTopStyle: "solid" }} />
            вход
          </span>
        </div>
      </div>
      <div className="chart-wrap">
        <div ref={box} style={{ position: "absolute", inset: 0 }} />
        {empty && <div className="fallback">Нет сохранённых свечей {base(sym)} — бот ещё не загружал историю по этой монете.</div>}
      </div>
    </article>
  );
}

// ── Лента и календарь ─────────────────────────────────────────────────────────

function Feed() {
  const feed = useStore((s) => s.feed);
  const wsUp = useStore((s) => s.wsUp);
  return (
    <article className="card feed">
      <div className="card-h">
        <h2>Что делает бот</h2>
        <span className="pill sp">
          <span className={`dot ${wsUp ? "ok live" : "bad"}`} />
          {wsUp ? "в реальном времени" : "переподключаюсь…"}
        </span>
      </div>
      <ul aria-live="polite">
        {feed.map((f) => (
          <li key={f.id} className={`ev ev-${f.kind}`}>
            <time>{hhmm(new Date(f.at))}</time>
            <span className="dot" />
            <p>{f.text}</p>
          </li>
        ))}
        {!feed.length && (
          <li className="ev ev-skip">
            <time>{hhmm()}</time>
            <span className="dot" />
            <p>Жду событий от бота…</p>
          </li>
        )}
      </ul>
    </article>
  );
}

const DOW = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"];

function Calendar() {
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
      return { d, row: byDate.get(isoDay(d)), future: d > today, today: +d === +today };
    });
  }, [daily]);
  const show = (el: HTMLElement, i: number) => {
    const { d, row, future } = days[i];
    if (future) return;
    showTip(
      <>
        {d.toLocaleDateString("ru-RU", { weekday: "long", day: "numeric", month: "long" })}
        <br />
        {row?.trades ? (
          <>
            <b className={tone(row.net)}>{`${arrow(row.net)} ${usd(row.net)}`}</b> чистыми · {row.trades}{" "}
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
  const fmtD = (d: Date) => d.toLocaleDateString("ru-RU", { day: "numeric", month: "short" });
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
          const mix = Math.round(22 + Math.min(1, Math.abs(net) / 55) * 78);
          return (
            <span
              key={i}
              role="img"
              tabIndex={future ? -1 : 0}
              className={`cell${future ? " future" : ""}${today ? " today" : ""}${has && net < 0 ? " loss" : ""}`}
              style={has ? { backgroundColor: `color-mix(in oklab, var(${net >= 0 ? "--up" : "--down"}) ${mix}%, var(--cell0))` } : undefined}
              aria-label={future ? `${d.getDate()}: впереди` : has ? `${d.getDate()}: ${usd(net)}` : `${d.getDate()}: сделок не было`}
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
