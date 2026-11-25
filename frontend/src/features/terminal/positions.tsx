// Терминал: живые позиции — прогресс к SL/TP, анимация входа с радара.

import { useEffect, useRef, useState, type FormEvent } from "react";

import { arrow, base, fmtN, pct, price, priceDecimals, tone, usd } from "../../lib/format";

import { reducedMotion } from "../../lib/platform";
import type { Position } from "../../lib/types";
import { actions } from "../../store/actions";
import { positionPnl, useStore } from "../../store";
import { stratName } from "../bots";

import { TraceSteps } from "../trace";
import { useArm } from "../../components/controls";
import { Flash, Gauge } from "../../components/motion";

// ── Живые позиции ─────────────────────────────────────────────────────────────

interface Leaving {
  p: Position;
  pnl: number;
}

export function Positions() {
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
