// Вкладка «Терминал»: баланс, метрики, радар монет, живые позиции, график, лента, календарь.

import {
  createChart,
  type IChartApi,
  type IPriceLine,
  type ISeriesApi,
  type SeriesMarker,
  type UTCTimestamp,
} from "lightweight-charts";
import {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
} from "react";

import { api } from "../lib/api";
import {
  arrow,
  base,
  fmtN,
  hhmm,
  pct,
  plural,
  price,
  priceDecimals,
  tone,
  usd,
} from "../lib/format";
import { reducedMotion } from "../lib/platform";
import type { Candle, Journal, JournalTrade, Position } from "../lib/types";
import { actions, positionPnl, useStore, type Store } from "../store";
import { cfg, profileName, stratName } from "./bots";
import { MarketPulse } from "./market";
import { ReadinessCard } from "./setup";
import { TraceSteps } from "./trace";
import { Flash, Gauge, Q, Seg, Tween, hideTip, showTip, useArm } from "../ui";

const EASE = "cubic-bezier(.2,.8,.2,1)";
const PERIODS = [
  ["24h", "24ч"],
  ["7d", "7д"],
  ["30d", "30д"],
  ["all", "Всё"],
] as const;

/** Строка здоровья: бот, связь, кто решает и когда был последний цикл. */
function HealthStrip() {
  const alive = useStore((s) => s.alive);
  const wsUp = useStore((s) => s.wsUp);
  const mode = useStore((s) => s.bot?.mode);
  const paused = useStore((s) => s.bot?.paused ?? false);
  const cycleAt = useStore((s) => s.cycleAt);
  const guardsOff = useStore(
    (s) =>
      (s.bot?.paper ?? true) &&
      cfg<boolean>(s.config, "PAPER_GUARDS_OFF", false),
  );
  const ai = useStore((s) => {
    const id = s.ai?.last || s.bot?.ai;
    return id ? (s.ai?.providers.find((p) => p.id === id)?.name ?? id) : "";
  });
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 5000);
    return () => clearInterval(t);
  }, []);
  const ago = cycleAt ? Math.max(0, Math.round((now - cycleAt) / 1000)) : null;
  const agoText =
    ago === null
      ? "ещё не было"
      : ago < 60
        ? `${ago} с назад`
        : `${Math.round(ago / 60)} мин назад`;
  // Цикл идёт каждые несколько минут; тишина дольше 15 минут при работающем боте — повод насторожиться.
  const stale = !paused && alive && ago !== null && ago > 900;
  // «Бот работает» уже видно в шапке (переключатель и «онлайн»): здесь — только когда не так.
  const down = !wsUp || !alive;
  const items: [string, string, string][] = [
    ...(down
      ? [
          ["Бот", !wsUp ? "нет связи" : "молчит", "bad"] as [
            string,
            string,
            string,
          ],
        ]
      : []),
    [
      "Решает",
      mode === "local"
        ? "локальные стратегии"
        : ai && ai !== "local"
          ? `AI · ${ai}`
          : "AI, ждёт ответа",
      mode === "local" || ai ? "ok" : "warn",
    ],
    ["Цикл", agoText, stale ? "bad" : ago === null ? "warn" : "ok"],
    ...(guardsOff
      ? [
          ["Защиты портфеля", "выключены (Paper)", "warn"] as [
            string,
            string,
            string,
          ],
        ]
      : []),
  ];
  return (
    <ul className="health" aria-label="Состояние системы">
      {items.map(([k, v, c]) => (
        <li key={k}>
          <span className={`dot ${c}`} />
          <small>{k}</small>
          <b>{v}</b>
        </li>
      ))}
    </ul>
  );
}

export function Terminal() {
  return (
    <section className="view term enter" aria-label="Терминал">
      <HealthStrip />
      <ReadinessCard />
      <Hero />
      <MetricsCard />
      <MarketPulse />
      <Radar />
      <Positions />
      <Chart />
      <Feed />
    </section>
  );
}

// ── Баланс и кривая equity ────────────────────────────────────────────────────

function Hero() {
  const balance = useStore((s) => s.balance);
  // Баланс коротко вспыхивает цветом сделки, когда закрытая позиция «приземляется» в него.
  const prevBal = useRef(balance);
  const [balFx, setBalFx] = useState("");
  useEffect(() => {
    if (balance === prevBal.current) return;
    setBalFx(balance > prevBal.current ? "fl-up" : "fl-down");
    prevBal.current = balance;
    const t = setTimeout(() => setBalFx(""), 900);
    return () => clearTimeout(t);
  }, [balance]);
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
        <Seg
          className="sm sp"
          value={period}
          options={PERIODS}
          onChange={setPeriod}
          label="Период"
        />
      </div>
      <div className={`bal ${balFx}`}>
        <Tween value={balance} fmt={(v) => `$${fmtN(v)}`} />
      </div>
      <div className="hero-stats">
        <div>
          <span className="label">
            Закрытые{" "}
            <Q tip="Прибыль и убыток по уже закрытым сделкам за выбранный период, до вычета комиссий." />
          </span>
          <b className={m ? tone(m.realized) : ""}>
            {m ? (
              <Tween value={m.realized} fmt={(v) => `${arrow(v)} ${usd(v)}`} />
            ) : (
              "—"
            )}
          </b>
          <small>
            {m ? `${m.n} ${plural(m.n, ["сделка", "сделки", "сделок"])}` : "—"}
          </small>
        </div>
        <div>
          <span className="label">
            Открытые{" "}
            <Q tip="Сколько вы получили бы, если закрыть открытые позиции прямо сейчас. Меняется вместе с ценой." />
          </span>
          <b className={tone(unreal)}>
            {list.length ? `${arrow(unreal)} ${usd(unreal)}` : "$0.00"}
          </b>
          <small>
            {list.length}{" "}
            {plural(list.length, ["позиция", "позиции", "позиций"])}
          </small>
        </div>
        <div>
          <span className="label">
            Чистыми{" "}
            <Q tip="Закрытые сделки минус комиссии биржи. Это то, что реально осталось на счёте." />
          </span>
          <b className={m ? tone(m.net) : ""}>
            {m ? (
              <Tween value={m.net} fmt={(v) => `${arrow(v)} ${usd(v)}`} />
            ) : (
              "—"
            )}
          </b>
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

// ── Метрики ───────────────────────────────────────────────────────────────────

function MetricsCard() {
  const m = useStore((s) => s.metrics[s.period]);
  // На телефоне видны две главные метрики, остальные раскрываются кнопкой.
  const [open, setOpen] = useState(false);
  if (!m) return <article className="card metrics" aria-busy="true" />;
  const wr = m.win_rate ?? 0;
  const aw = m.avg_win ?? 0;
  const al = Math.abs(m.avg_loss ?? 0);
  const tot = aw + al || 1;
  return (
    <article className={`card metrics${open ? " open" : ""}`}>
      <div className="tile">
        <div className="t-h">
          <span>Доля прибыльных</span>
          <Q tip="Win rate — сколько сделок закрылись в плюс. 40% — нормально, если прибыльные сделки крупнее убыточных." />
        </div>
        <div className="v">
          {m.win_rate === null ? (
            "—"
          ) : (
            <Tween value={wr * 100} fmt={(v) => `${fmtN(v, 1)}%`} />
          )}
        </div>
        <div className="bar">
          <i style={{ background: "var(--up)", width: `${wr * 100}%` }} />
        </div>
        <small>
          {Math.round(wr * m.n)} из {m.n}{" "}
          {plural(m.n, ["сделки", "сделок", "сделок"])}
        </small>
      </div>
      <div className="tile">
        <div className="t-h">
          <span>Профит-фактор</span>
          <Q tip="Сумма всех прибылей, делённая на сумму всех убытков. Больше 1 — в плюсе, меньше 1 — в минусе." />
        </div>
        <div
          className={`v ${m.profit_factor === null ? "" : m.profit_factor >= 1 ? "up" : "down"}`}
        >
          {m.profit_factor === null ? (
            "—"
          ) : (
            <Tween value={m.profit_factor} fmt={(v) => fmtN(v, 2)} />
          )}
        </div>
        <small>
          {m.profit_factor === null
            ? "нет убыточных сделок"
            : m.profit_factor >= 1
              ? "прибыли больше убытков"
              : "убытки больше прибылей"}
        </small>
      </div>
      <div className="tile">
        <div className="t-h">
          <span>Средняя сделка</span>
          <Q tip="Средний размер прибыльной и убыточной сделки. Полоски показывают их соотношение." />
        </div>
        <div className="v small">
          <span className="up">
            {m.avg_win === null ? "—" : `+$${fmtN(aw, 1)}`}
          </span>{" "}
          /{" "}
          <span className="down">
            {m.avg_loss === null ? "—" : `−$${fmtN(al, 1)}`}
          </span>
        </div>
        <div className="bar">
          <i
            style={{ background: "var(--up)", width: `${(aw / tot) * 100}%` }}
          />
          <i
            style={{ background: "var(--down)", width: `${(al / tot) * 100}%` }}
          />
        </div>
      </div>
      <div className="tile">
        <div className="t-h">
          <span>Риск : прибыль</span>
          <Q tip="R:R — во сколько раз средняя прибыльная сделка больше средней убыточной." />
        </div>
        <div className="v">
          {m.rr === null ? (
            "—"
          ) : (
            <Tween value={m.rr} fmt={(v) => `1 : ${fmtN(v, 2)}`} />
          )}
        </div>
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
      <button
        className="btn metrics-more"
        type="button"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
      >
        {open ? "Скрыть метрики" : "Ещё 4 метрики"}
      </button>
    </article>
  );
}

// ── Качество исполнения: задержка ордера и проскальзывание ────────────────────

/** Качество исполнения — живёт в «Журнале». */
export function ExecutionCard() {
  const e = useStore((s) => s.execution[s.period]);
  if (!e) return <article className="card exec" aria-busy="true" />;
  const few = e.latency_ms_avg === null || e.slippage_pct_avg === null;
  return (
    <article className="card exec">
      <div className="card-h">
        <h2>Качество исполнения</h2>
        <Q tip="Насколько ордер отстал от решения бота: сколько миллисекунд прошло от отправки до исполнения и на сколько процентов цена исполнения хуже цены сигнала. Считаем от 20 сделок: на меньшем числе цифра случайна." />
      </div>
      {few ? (
        <p className="muted">
          Мало данных: {e.n} из {e.min_n}{" "}
          {plural(e.min_n, ["сделки", "сделок", "сделок"])}, нужных для расчёта.
        </p>
      ) : (
        <div className="exec-rows">
          <div className="exec-row">
            <span className="label">Задержка ордера</span>
            <b>{fmtN(e.latency_ms_avg ?? 0, 0)} мс</b>
            <small>в среднем, от отправки до исполнения</small>
          </div>
          <div className="exec-row">
            <span className="label">Проскальзывание</span>
            <b className={tone(-(e.slippage_pct_avg ?? 0))}>
              {fmtN(e.slippage_pct_avg ?? 0, 3)}%
            </b>
            <small>
              в среднем, плюс — хуже цены сигнала; худшая сделка{" "}
              {fmtN(e.slippage_pct_worst ?? 0, 3)}%
            </small>
          </div>
          <small className="muted">
            {e.simulated
              ? `Paper: исполнение симулировано, по ${e.n} сделкам.`
              : `По ${e.n} сделкам.`}
          </small>
        </div>
      )}
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

function Radar() {
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
    const gone = Object.values(prev.current).filter(
      (p) => !positions[p.symbol],
    );
    prev.current = positions;
    if (!gone.length) return;
    const items = gone.map((p) => ({ p, pnl: positionPnl(p).pnl }));
    setLeaving((l) => [...l, ...items]);
    setTimeout(
      () => setLeaving((l) => l.filter((x) => !items.includes(x))),
      reducedMotion() ? 0 : 1150,
    );
  }, [positions]);

  // Карточки при первом показе появляются без анимации, новые — со вспышкой.
  if (seen.current === null) seen.current = new Set(list.map((p) => p.symbol));
  const fresh = new Set(
    list.map((p) => p.symbol).filter((s) => !seen.current!.has(s)),
  );
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
          <article
            key={`out-${p.symbol}-${p.trade_id}`}
            className={`pos leaving ${pnl >= 0 ? "win" : "loss"}`}
            aria-hidden="true"
          >
            <div className="pos-h">
              <span className="coin">{base(p.symbol)}</span>
              <div className="pos-t">
                <b>{p.symbol}</b>
                <small>закрыта</small>
              </div>
              <div className="pos-pnl">
                <b className={tone(pnl)}>{`${arrow(pnl)} ${usd(pnl)}`}</b>
                <span
                  className={`burst ${tone(pnl)}`}
                >{`${arrow(pnl)} ${usd(pnl)}`}</span>
              </div>
            </div>
          </article>
        ))}
        {!list.length && !leaving.length && (
          <div className="empty">
            <p>Открытых позиций нет. Бот ищет сделки — следите за радаром.</p>
            <button className="btn" type="button" onClick={() => useStore.getState().openHelp(true)}>
              Как бот выбирает сделки
            </button>
          </div>
        )}
      </div>
    </article>
  );
}

const progOf = (p: Position, v: number) =>
  Math.max(
    0,
    Math.min(
      1,
      p.side === "long"
        ? (v - p.sl) / (p.tp - p.sl)
        : (p.sl - v) / (p.sl - p.tp),
    ),
  );

const inView = (r: DOMRect): boolean =>
  r.bottom > 0 && r.top < innerHeight && r.width > 0;

/** Путь сделки: засветка вылетает из слота «Позиции» на радаре и садится в значок новой карточки. */
function flyFromRadar(symbol: string, card: HTMLElement | null): void {
  const from = document.querySelector(
    `.lane-pos [data-sym="${CSS.escape(symbol)}"]`,
  );
  const to = card?.querySelector<HTMLElement>(".coin");
  if (!from || !to || reducedMotion()) return;
  const a = from.getBoundingClientRect();
  const b = to.getBoundingClientRect();
  if (!inView(a) || !inView(b)) return;
  const [x0, y0] = [a.left + a.width / 2, a.top + a.height / 2];
  const [dx, dy] = [b.left + b.width / 2 - x0, b.top + b.height / 2 - y0];
  const dot = document.createElement("i");
  dot.className = "fly";
  Object.assign(dot.style, { left: `${x0}px`, top: `${y0}px` });
  document.body.appendChild(dot);
  // Дуга: середина пути приподнята — так видно, откуда и куда.
  const lift = -Math.min(80, Math.abs(dx) * 0.25 + 30);
  dot
    .animate(
      [
        { transform: "translate(0,0)" },
        { transform: `translate(${dx / 2}px,${dy / 2 + lift}px)` },
        { transform: `translate(${dx}px,${dy}px)` },
      ],
      { duration: 700, easing: "cubic-bezier(.3,.7,.1,1)" },
    )
    .finished.then(() => {
      dot.remove();
      to.classList.add("arrive");
      setTimeout(() => to.classList.remove("arrive"), 700);
    })
    .catch(() => dot.remove());
}

function PositionCard({ p, enter }: { p: Position; enter: boolean }) {
  const online = useStore((s) => s.alive);
  const self = useRef<HTMLElement>(null);
  // Новая позиция: ждём, пока чип доедет до слота на радаре, потом засветка летит в карточку.
  useEffect(() => {
    if (!enter) return;
    const t = setTimeout(() => flyFromRadar(p.symbol, self.current), 450);
    return () => clearTimeout(t);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
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
      const part =
        (cur - p.entry) *
        (prevQty.current - p.qty) *
        (p.side === "long" ? 1 : -1);
      setBurst({ k: Date.now(), pnl: part });
    }
    prevQty.current = p.qty;
  }, [p.qty]); // eslint-disable-line react-hooks/exhaustive-deps

  const can = online && p.trade_id !== null && !busy;
  // Свайп влево (только пальцем) открывает полосу «Закрыть». Она не закрывает сделку сама:
  // нажатие по ней запускает то же двустороннее подтверждение, что и кнопка в карточке.
  const SWIPE_W = 96;
  const [dx, setDx] = useState(0);
  const [dragging, setDragging] = useState(false);
  const drag = useRef<{ x: number; y: number; base: number; on: boolean } | null>(null);
  const revealed = dx <= -SWIPE_W / 2;
  const onSwipeDown = (e: React.PointerEvent) => {
    if (e.pointerType !== "touch" || !can || editing) return;
    drag.current = { x: e.clientX, y: e.clientY, base: dx, on: false };
  };
  const onSwipeMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const mx = e.clientX - d.x;
    if (!d.on) {
      if (Math.abs(e.clientY - d.y) > 10) return void (drag.current = null); // это скролл
      if (Math.abs(mx) < 10) return;
      d.on = true;
      setDragging(true);
    }
    setDx(Math.max(-SWIPE_W, Math.min(0, d.base + mx)));
  };
  const onSwipeEnd = () => {
    const d = drag.current;
    drag.current = null;
    if (!d?.on) return;
    setDragging(false);
    setDx((v) => (v < -SWIPE_W / 2 ? -SWIPE_W : 0));
  };
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
    <div className="pos-swipe">
      <button
        className={`pos-swipe-act ${closeArmed ? "armed" : ""}`}
        type="button"
        tabIndex={revealed ? 0 : -1}
        aria-hidden={!revealed}
        disabled={!can}
        onClick={() => armClose() && void run(() => actions.closePosition(p))}
      >
        {closeArmed ? "Точно?" : "Закрыть"}
      </button>
    <article
      ref={self}
      className={`pos ${pnl >= 0 ? "win" : "loss"} ${enter ? "enter" : ""}`}
      style={dx || dragging ? { transform: `translateX(${dx}px)`, transition: dragging ? "none" : undefined, touchAction: "pan-y" } : { touchAction: "pan-y" }}
      onPointerDown={onSwipeDown}
      onPointerMove={onSwipeMove}
      onPointerUp={onSwipeEnd}
      onPointerCancel={onSwipeEnd}
      onClick={() => revealed && setDx(0)}
    >
      <div className="pos-h">
        <span className="coin">{base(p.symbol)}</span>
        <div className="pos-t">
          <b>
            {p.symbol}
            <span className={`side ${p.side}`}>
              {p.side === "long" ? "LONG" : "SHORT"}
            </span>
          </b>
          <small>
            {p.strategy
              ? `${stratName(p.strategy)}${p.confidence ? ` · уверенность ${p.confidence.toFixed(2)}` : ""}`
              : `сделка #${p.trade_id ?? "—"}`}
          </small>
        </div>
        {p.confidence ? (
          <Gauge value={p.confidence} label="Уверенность сигнала" />
        ) : null}
        <div className="pos-pnl">
          <b className={`num ${tone(pnl)}`}>{`${arrow(pnl)} ${usd(pnl)}`}</b>
          <small>{pct(move)}</small>
          {burst && (
            <span
              key={burst.k}
              className={`burst ${tone(burst.pnl)}`}
              onAnimationEnd={() => setBurst(null)}
            >
              {`${arrow(burst.pnl)} ${usd(burst.pnl)}`}
            </span>
          )}
        </div>
      </div>
      {(p.reasoning || p.trace?.length) && (
        <p
          className={`pos-why ${whyOpen ? "open" : ""}`}
          title={p.reasoning}
          onClick={() => setWhyOpen(!whyOpen)}
        >
          <em>Почему открыта:</em> {p.reasoning || "по шагам решения"}
          {p.trace?.length && !whyOpen ? <span className="pos-why-more"> · шаги ▸</span> : null}
        </p>
      )}
      {whyOpen && p.trace?.length ? <TraceSteps steps={p.trace} /> : null}
      <div className="range">
        <span>
          SL <Flash value={p.sl} text={price(p.sl)} />
        </span>
        <div className="track">
          <i
            className="fill"
            style={{
              left: `${Math.min(e, mk)}%`,
              width: `${Math.abs(mk - e)}%`,
              background: pnl >= 0 ? "var(--up)" : "var(--down)",
            }}
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
        <button
          className="btn"
          type="button"
          aria-pressed={editing}
          disabled={!can}
          onClick={() => setEditing(!editing)}
        >
          SL / TP
        </button>
        <button
          className="btn"
          type="button"
          aria-pressed={p.trailing !== false}
          disabled={!can}
          title="Стоп-лосс подтягивается за ценой. Выключите, чтобы стоп стоял на месте."
          onClick={() =>
            void run(() =>
              actions.updatePosition(p, { trailing: p.trailing === false }),
            )
          }
        >
          Трейлинг
        </button>
        <button
          className="btn"
          type="button"
          disabled={!can}
          onClick={() => void run(() => actions.closePosition(p, 0.5))}
        >
          Закрыть 50%
        </button>
      </div>
      {editing && (
        <form className="edit" noValidate onSubmit={onSave}>
          <label>
            Стоп-лосс
            <input
              className="inp"
              name="sl"
              inputMode="decimal"
              defaultValue={p.sl.toFixed(priceDecimals(p.sl))}
              autoFocus
            />
          </label>
          <label>
            Тейк-профит
            <input
              className="inp"
              name="tp"
              inputMode="decimal"
              defaultValue={p.tp.toFixed(priceDecimals(p.tp))}
            />
          </label>
          <button className="btn primary" type="submit" disabled={busy}>
            Сохранить
          </button>
          {err && <span className="err">{err}</span>}
        </form>
      )}
    </article>
    </div>
  );
}

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

function Chart() {
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
          // Реакция на исход — только у свежих записей, не при каждом открытии вкладки.
          <li
            key={f.id}
            className={`ev ev-${f.kind}${Date.now() - f.at < 2000 ? " fresh" : ""}`}
          >
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
