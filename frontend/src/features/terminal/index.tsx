// Вкладка «Терминал»: баланс, метрики, радар монет, живые позиции, график, лента, календарь.

import { lazy, useEffect, useRef, useState } from "react";

import { arrow, base, fmtN, plural, tone, usd } from "../../lib/format";
import { idleReason } from "../../lib/idle";

import { positionPnl, useStore } from "../../store";
import type { AiState } from "../../lib/types";
import { cfg } from "../bots";
import { STATUS } from "../chainops";
import { Journal } from "../journal";
import { MarketPulse } from "../market";
import { ReadinessChip } from "../setup";

import { Seg } from "../../components/controls";
import { Tween } from "../../components/motion";
import { Q } from "../../components/tooltip";
import { Chart } from "./chart";
import { Equity } from "./equity";
import { Positions } from "./positions";
import { Radar } from "./radar";

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
  // Время последнего цикла: из живого события, а сразу после открытия — из снимка состояния (секунды).
  const cycleAt = useStore((s) => s.cycleAt ?? (s.lastCycle?.at ? s.lastCycle.at * 1000 : null));
  const guardsOff = useStore(
    (s) =>
      (s.bot?.paper ?? true) &&
      cfg<boolean>(s.config, "PAPER_GUARDS_OFF", false),
  );
  const ai = useStore((s) => {
    const id = s.ai?.last || s.bot?.ai;
    const p = id ? s.ai?.providers.find((x) => x.id === id) : undefined;
    return id ? (p ? `${p.name} · ${p.model}` : id) : "";
  });
  const secrets = useStore((s) => s.secrets);
  const aiState = useStore((s) => s.ai);
  const setTab = useStore((s) => s.setTab);
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 5000);
    return () => clearInterval(t);
  }, []);
  const aiOut = aiDown(aiState, now);
  const aiOk = Boolean(aiState?.chain.length) && !aiOut;
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
        : aiOut
          ? `стратегии — AI не отвечает (${aiOut})`
          : ai && ai !== "local"
            ? `AI · ${ai}`
            : "AI, ждёт ответа",
      mode === "local" ? "ok" : aiOut ? "bad" : ai ? "ok" : "warn",
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
  // Огни подключений: биржа, AI и Telegram; нажатие ведёт к карте в «Настройках».
  const light = (id: string) => {
    const st = secrets?.find((x) => x.service === id)?.status;
    return st === "ok" ? "ok" : st === "bad" ? "bad" : "warn";
  };
  const links: [string, string][] = [
    ["Bybit", light("bybit")],
    ["AI", aiOk ? "ok" : "bad"],
    ["Telegram", light("telegram")],
  ];
  return (
    <article className="status" aria-label="Состояние системы">
    <ul className="health">
      {items.map(([k, v, c]) => (
        <li key={k}>
          <span className={`dot ${c}`} />
          <small>{k}</small>
          <b>{v}</b>
        </li>
      ))}
      {secrets && (
        <li className="health-links">
          <small>Связи</small>
          <button
            type="button"
            aria-label={`Подключения: ${links.map(([n, c]) => `${n} ${c === "ok" ? "работает" : "не работает"}`).join(", ")}. Открыть карту`}
            onClick={() => setTab("settings")}
          >
            {links.map(([n, c]) => (
              <span key={n}>
                <span className={`dot ${c}`} />
                {n}
              </span>
            ))}
          </button>
        </li>
      )}
      <ReadinessChip />
      <TrustedChip />
    </ul>
    <Idle />
    </article>
  );
}

/** Все звенья цепочки AI сейчас недоступны — коротко, почему; иначе пусто. */
export function aiDown(ai: AiState | null, now: number): string {
  if (!ai?.chain.length) return "";
  const reasons = ai.chain.map(({ provider }) => {
    const p = ai.providers.find((x) => x.id === provider);
    if (!p) return "";
    const paused = (p.cooldown_until ?? 0) * 1000 > now;
    const fine = p.status === "ok" || p.status === "unknown";
    if (fine && !paused) return "";
    return `${p.name} — ${fine ? "пауза" : (STATUS[p.status]?.[1] ?? p.status)}`;
  });
  return reasons.every(Boolean) ? reasons.join(", ") : "";
}

/** «Почему бот молчит?»: одна причина простыми словами и переход к шагу, где это настраивается. */
function Idle() {
  const wsUp = useStore((s) => s.wsUp);
  const alive = useStore((s) => s.alive);
  const paused = useStore((s) => s.bot?.paused ?? false);
  const cycle = useStore((s) => s.lastCycle);
  const slots = useStore((s) => s.slots);
  const hours = useStore((s) => cfg<string>(s.config, "TRADING_HOURS", ""));
  const openNode = useStore((s) => s.openNode);
  const [hourUtc, setHour] = useState(() => new Date().getUTCHours());
  useEffect(() => {
    const t = setInterval(() => setHour(new Date().getUTCHours()), 60_000);
    return () => clearInterval(t);
  }, []);
  // Нет связи с сервером — об этом уже говорят шапка и строка здоровья.
  const ran = useStore((s) => s.cycleAt !== null || Boolean(s.lastCycle?.at));
  const firstRun = useStore((s) => s.bot?.first_run ?? false);
  const r = wsUp ? idleReason({ alive, paused, cycle, slots, hours, hourUtc, ran, firstRun }) : null;
  if (!r) return null;
  return (
    <p className={`status-why ${r.tone}`} role="status">
      <span className={`dot ${r.tone}`} />
      <b>{r.title}</b>
      <span>{r.detail}</span>
      {r.node && (
        <button type="button" className="link" onClick={() => openNode(r.node!)}>
          где настроить
        </button>
      )}
    </p>
  );
}

/** Прокрутка к блоку другой вкладки: ждём, пока вкладка отрисуется (до ~1 с). */
export const scrollWhenReady = (id: string, tries = 20): void => {
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  else if (tries) setTimeout(() => scrollWhenReady(id, tries - 1), 50);
};

/** Последний пост доверенного канала за сутки: вход через сколько секунд или почему нет. */
function TrustedChip() {
  const t = useStore((s) => s.trusted);
  const openSettings = useStore((s) => s.setTab);
  if (!t || Date.now() / 1000 - t.at > 86_400) return null;
  const what = t.symbol ? `${base(t.symbol)} ${t.side === "buy" ? "LONG" : "SHORT"}` : "монета не разобрана";
  const when = new Date(t.at * 1000).toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" });
  return (
    <li className="health-act">
      <span className={`dot ${t.ok ? "ok" : "warn"}`} />
      <small>Канал</small>
      <button
        type="button"
        title={`${t.ok ? `Вход через ${t.latency_s} с после поста` : `Без сделки: ${t.reason}`} · ${when}. Стратегии, AI и фильтры скана в этих сделках не участвуют.`}
        onClick={() => (openSettings("settings"), scrollWhenReady("tg-channels"))}
      >
        @{t.channel} · {what} · {t.ok ? `вход ${t.latency_s} с` : "пропущен"}
      </button>
    </li>
  );
}

const PathBar = lazy(() => import("../jobs").then((m) => ({ default: m.PathBar })));

/** Терминал: шапка состояния, радар · график · баланс, внизу панель «Позиции / Журнал сделок». */
export function Terminal() {
  const simple = useStore((s) => s.ui === "simple");
  const [dock, setDock] = useState<"pos" | "log">("pos");
  return (
    <section className="view term enter" aria-label="Терминал">
      <HealthStrip />
      {simple && <PathBar />}
      {!simple && <Radar />}
      <div className="term-side">
        <Hero />
      </div>
      {!simple && <Chart />}
      {!simple && <MarketPulse />}
      <div className="term-dock">
        <Seg
          className="sm"
          value={dock}
          options={[
            ["pos", "Позиции"],
            ["log", "Журнал сделок"],
          ]}
          label="Нижняя панель"
          onChange={setDock}
        />
        {dock === "pos" ? <Positions /> : <Journal />}
      </div>
    </section>
  );
}

// ── Баланс и кривая equity ────────────────────────────────────────────────────

function Hero() {
  const balance = useStore((s) => s.balance);
  // Баланс коротко вспыхивает цветом сделки, когда закрытая позиция «приземляется» в него.
  const prevBal = useRef(balance);
  const [balFx, setBalFx] = useState("");
  // Рядом всплывает сумма, на которую он изменился, — видно, сколько принесла сделка.
  const [delta, setDelta] = useState(0);
  useEffect(() => {
    if (balance === prevBal.current) return;
    setBalFx(balance > prevBal.current ? "fl-up" : "fl-down");
    setDelta(balance - prevBal.current);
    prevBal.current = balance;
    const t = setTimeout(() => setBalFx(""), 900);
    const d = setTimeout(() => setDelta(0), 1800);
    return () => {
      clearTimeout(t);
      clearTimeout(d);
    };
  }, [balance]);
  const paper = useStore((s) => s.bot?.paper ?? true);
  const period = useStore((s) => s.period);
  const setPeriod = useStore((s) => s.setPeriod);
  const m = useStore((s) => s.metrics[s.period]);
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
        {delta !== 0 && (
          <span key={balance} className={`bal-d ${tone(delta)}`} aria-hidden="true">
            {arrow(delta)} {usd(delta)}
          </span>
        )}
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
        <OpenPnl />
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
      <DailyStop />
      <Equity />
    </article>
  );
}

/** P&L открытых позиций — отдельно, чтобы тики цены перерисовывали только эту ячейку, а не всю карточку. */
function OpenPnl() {
  const unreal = useStore((s) => Object.values(s.positions).reduce((sum, p) => sum + positionPnl(p).pnl, 0));
  const n = useStore((s) => Object.keys(s.positions).length);
  return (
    <div>
      <span className="label">
        Открытые <Q tip="Сколько вы получили бы, если закрыть открытые позиции прямо сейчас. Меняется вместе с ценой." />
      </span>
      <b className={tone(unreal)}>{n ? `${arrow(unreal)} ${usd(unreal)}` : "$0.00"}</b>
      <small>
        {n} {plural(n, ["позиция", "позиции", "позиций"])}
      </small>
    </div>
  );
}

/** Дневной стоп: при каком капитале бот сегодня перестанет открывать сделки (RiskManager, сброс в 00:00 UTC). */
function DailyStop() {
  const ref = useStore((s) => s.dayRef);
  const limit = useStore((s) => Number(cfg(s.config, "DAILY_LOSS_LIMIT", 0)));
  const off = useStore((s) => (s.bot?.paper ?? true) && cfg<boolean>(s.config, "PAPER_GUARDS_OFF", false));
  const openNode = useStore((s) => s.openNode);
  if (!ref || !(limit > 0)) return null;
  return (
    <p className={`day-stop${off ? " off" : ""}`}>
      <span className="label">
        Дневной стоп{" "}
        <Q tip="Если капитал за сутки упадёт на этот процент от уровня 00:00 UTC, новых сделок до конца дня не будет. Открытые закроются по своим SL/TP." />
      </span>
      {off ? (
        <span>не действует — защиты портфеля выключены (Paper)</span>
      ) : (
        <span>
          входы встанут ниже <b className="num">${fmtN(ref.balance * (1 - limit))}</b> · −{fmtN(limit * 100, 1)}% от уровня 00:00 UTC
        </span>
      )}
      <button type="button" className="link" onClick={() => openNode("risk")}>
        изменить
      </button>
    </p>
  );
}
