// Оболочка приложения: верхняя панель, баннеры защиты, вкладки, предложение сделки, вход.

import { useEffect, useLayoutEffect, useRef, useState, type FormEvent } from "react";

import { isMock } from "./lib/api";
import { base, fmtN, plural, price, usd } from "./lib/format";
import * as platform from "./lib/platform";
import type { Tab } from "./lib/types";
import { Bots, profileName, stratName } from "./features/bots";
import { Journal } from "./features/journal";
import { Settings } from "./features/settings";
import { Setup } from "./features/setup";
import { Terminal, cycleKind } from "./features/terminal";
import { GUARD_TEXT, actions, useStore } from "./store";
import { Icon, TipLayer, Toggle, hideTip } from "./ui";

export function App() {
  const ready = useStore((s) => s.ready);
  const needLogin = useStore((s) => s.needLogin);
  const tab = useStore((s) => s.tab);
  const setup = useStore((s) => s.setup);
  const views = useRef<HTMLElement>(null);
  // Новая вкладка въезжает с той стороны, где стоит в меню.
  const prevTab = useRef(tab);
  const dir = useRef<"fwd" | "back">("fwd");
  if (prevTab.current !== tab) {
    const at = (t: Tab) => TABS.findIndex(([id]) => id === t);
    dir.current = at(tab) > at(prevTab.current) ? "fwd" : "back";
    prevTab.current = tab;
  }
  const [flash, setFlash] = useState(0);

  useEffect(() => {
    void useStore.getState().boot();
  }, []);
  // Ctrl/⌘ K — к поиску параметров из любого места.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        if (useStore.getState().ready) useStore.getState().searchParams();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  useEffect(() => {
    views.current?.scrollTo(0, 0);
    hideTip();
  }, [tab, setup]);

  if (!ready) return <div className="boot">Подключаюсь к боту…</div>;
  if (needLogin) return <Login />;
  return (
    <div className={`app${flash ? " flash" : ""}`} onAnimationEnd={(e) => e.target === e.currentTarget && setFlash(0)}>
      <TopBar onPanic={() => setFlash(Date.now())} />
      <Banners />
      <main className="views" ref={views} onScroll={hideTip} data-dir={dir.current}>
        {setup ? (
          <Setup />
        ) : (
          <>
            {tab === "terminal" && <Terminal />}
            {tab === "bots" && <Bots />}
            {tab === "journal" && <Journal />}
            {tab === "settings" && <Settings />}
          </>
        )}
      </main>
      <Nav />
      <SignalSheet />
      <Toast />
      <TipLayer />
    </div>
  );
}

// ── Верхняя панель ────────────────────────────────────────────────────────────

const CYCLE: Record<string, [string, string, string]> = {
  scan: ["Сканирую", "brand live", ""],
  monitor: ["Слежу", "ok live", ""],
  paused: ["Пауза", "warn", "· новые сделки не открываются"],
  guard: ["Защита", "warn", "· входы остановлены"],
  off: ["Нет связи", "bad", "· показаны последние данные"],
};

function TopBar({ onPanic }: { onPanic: () => void }) {
  const bot = useStore((s) => s.bot);
  const kind = useStore(cycleKind);
  const wsUp = useStore((s) => s.wsUp);
  const alive = useStore((s) => s.alive);
  const used = useStore((s) => s.slots.used);
  const [busy, setBusy] = useState(false);
  const paper = bot?.paper ?? true;
  const paused = bot?.paused ?? false;
  const [txt, dot, more] = CYCLE[kind];
  const detail =
    kind === "scan"
      ? `· ${bot?.profile ? profileName(bot.profile) : "профиль по умолчанию"}`
      : kind === "monitor"
        ? `за ${used} ${plural(used, ["позицией", "позициями", "позициями"])}`
        : more;

  const toggle = async (on: boolean) => {
    setBusy(true);
    if (await (on ? actions.resume() : actions.pause())) await useStore.getState().loadState().catch(() => undefined);
    setBusy(false);
  };

  return (
    <header className="topbar">
      <div className="brand">
        {Icon.logo}
        BitbotBY
      </div>
      <span className={`paper${paper ? "" : " live"}`} title={paper ? "Бумажная торговля: сделки виртуальные" : "Реальная торговля на Bybit"}>
        {paper ? "PAPER" : "LIVE"}
      </span>
      {isMock && <span className="demo">демо</span>}
      <span className="lat" title={wsUp ? "Поток событий подключён" : "Переподключаюсь к серверу"}>
        <span className={`dot ${wsUp && alive ? "ok live" : wsUp ? "warn" : "bad"}`} />
        {wsUp ? (alive ? "онлайн" : "бот молчит") : "нет связи"}
      </span>
      <span className="break" />
      <Toggle className="master" on={!paused} onChange={(on) => void toggle(on)} label="Бот" disabled={busy || !alive}>
        <span>{paused ? "На паузе" : "Работает"}</span>
      </Toggle>
      <span className="cycle">
        <span className={`dot ${dot}`} />
        <span>{txt}</span>
        <span className="more">{detail}</span>
      </span>
      <PanicButton onFire={onPanic} disabled={!alive} />
    </header>
  );
}

/** PANIC под откидной крышкой: поднять крышку, затем удерживать 1.1 с —
 *  закрыть всё по рынку и поставить бота на паузу. Крышка сама опускается через 5 с. */
function PanicButton({ onFire, disabled }: { onFire: () => void; disabled: boolean }) {
  const [holding, setHolding] = useState(false);
  const [open, setOpen] = useState(false);
  // Крышка нужна там, где легко задеть экран пальцем; с мышью хватает удержания.
  const covered = useRef(typeof matchMedia === "function" && matchMedia("(pointer: coarse)").matches).current;
  const timer = useRef<ReturnType<typeof setTimeout>>();
  const cover = useRef<ReturnType<typeof setTimeout>>();
  const t0 = useRef(0);
  const say = useStore((s) => s.say);
  useEffect(() => () => clearTimeout(cover.current), []);
  const lift = () => {
    setOpen(true);
    platform.haptic("tap");
    say("Крышка поднята — удерживайте PANIC 1 секунду");
    clearTimeout(cover.current);
    cover.current = setTimeout(() => setOpen(false), 5000);
  };
  const start = () => {
    if (disabled) return;
    if (covered && !open) return lift();
    clearTimeout(cover.current);
    t0.current = Date.now();
    setHolding(true);
    timer.current = setTimeout(() => {
      timer.current = undefined;
      setHolding(false);
      setOpen(false);
      platform.haptic("warning");
      if (!platform.reducedMotion()) onFire();
      void actions.panic().then(() => useStore.getState().loadState().catch(() => undefined));
    }, 1100);
  };
  const end = () => {
    if (!timer.current) return;
    clearTimeout(timer.current);
    timer.current = undefined;
    setHolding(false);
    cover.current = setTimeout(() => setOpen(false), 5000);
    if (Date.now() - t0.current < 600) say("Удерживайте PANIC 1 секунду, чтобы закрыть всё");
  };
  return (
    <button
      className={`panic${holding ? " holding" : ""}${covered ? " covered" : ""}${open ? " open" : ""}`}
      type="button"
      disabled={disabled}
      aria-expanded={covered ? open : undefined}
      title={
        open || !covered
          ? "Удерживайте 1 секунду: закрыть все позиции по рынку и поставить бота на паузу"
          : "PANIC под крышкой: нажмите, чтобы поднять, затем удерживайте 1 секунду"
      }
      onPointerDown={(e) => {
        e.preventDefault();
        start();
      }}
      onPointerUp={end}
      onPointerLeave={end}
      onPointerCancel={end}
      onKeyDown={(e) => {
        if ((e.key === " " || e.key === "Enter") && !e.repeat) {
          e.preventDefault();
          start();
        }
      }}
      onKeyUp={(e) => (e.key === " " || e.key === "Enter") && end()}
      onClick={(e) => e.preventDefault()}
    >
      {Icon.panic}
      <span>
        PANIC<span className="long"> CLOSE</span>
      </span>
      <span className="panic-cover" aria-hidden="true" />
    </button>
  );
}

// ── Баннеры: защита и потеря связи ────────────────────────────────────────────

const untilText = (until?: number) =>
  until ? ` до ${new Date(until * 1000).toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" })}` : "";

function Banners() {
  const guards = useStore((s) => s.guards);
  const alive = useStore((s) => s.alive);
  const wsUp = useStore((s) => s.wsUp);
  const active = Object.values(guards).filter((g) => g.active);
  return (
    <div className="banners">
      {(!alive || !wsUp) && (
        <div className="banner off" role="status">
          {Icon.offline}
          <div>
            <b>{wsUp ? "Бот не отвечает" : "Нет связи с сервером бота"}</b>
            <p>Показаны последние полученные данные. Кнопки управления заблокированы, пока связь не вернётся. Стоп-лоссы на бирже продолжают работать.</p>
          </div>
        </div>
      )}
      {active.map((g) => (
        <div className="banner guard" role="status" key={g.kind}>
          {Icon.shield}
          <div>
            <b>
              {GUARD_TEXT[g.kind] ?? "Сработала защита"}
              {untilText(g.until)}
            </b>
            {g.detail && <p>{g.detail}</p>}
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Нижняя навигация (на десктопе — боковая) ──────────────────────────────────

const TABS: [Tab, string, JSX.Element][] = [
  ["terminal", "Терминал", Icon.terminal],
  ["bots", "Боты", Icon.bots],
  ["journal", "Журнал", Icon.journal],
  ["settings", "Настройки", Icon.settings],
];

function Nav() {
  const tab = useStore((s) => s.tab);
  const setTab = useStore((s) => s.setTab);
  const pending = useStore((s) => s.pending.length);
  const nav = useRef<HTMLElement>(null);
  const [ind, setInd] = useState<React.CSSProperties>({});
  useLayoutEffect(() => {
    const move = () => {
      const b = nav.current?.querySelector<HTMLElement>('button[aria-current="page"]');
      if (b) setInd({ width: b.offsetWidth, height: b.offsetHeight, transform: `translate(${b.offsetLeft}px,${b.offsetTop}px)` });
    };
    move();
    const ro = new ResizeObserver(move);
    if (nav.current) ro.observe(nav.current);
    return () => ro.disconnect();
  }, [tab]);
  return (
    <nav className="nav" ref={nav} aria-label="Разделы">
      <span className="nav-ind" style={ind} aria-hidden="true" />
      {TABS.map(([id, label, icon]) => (
        <button
          key={id}
          type="button"
          aria-current={tab === id ? "page" : undefined}
          className={id === "terminal" && pending && tab !== "terminal" ? "ping" : ""}
          onClick={() => {
            platform.haptic("tap");
            setTab(id);
          }}
        >
          {icon}
          {label}
        </button>
      ))}
    </nav>
  );
}

// ── Предложение сделки (бот ждёт подтверждения) ───────────────────────────────

function SignalSheet() {
  const sig = useStore((s) => s.pending[0]);
  const tab = useStore((s) => s.tab);
  const balance = useStore((s) => s.balance);
  const risk = useStore((s) => Number(s.config.RISK_PER_TRADE?.value ?? 0));
  const say = useStore((s) => s.say);
  const [now, setNow] = useState(Date.now() / 1000);
  const [busy, setBusy] = useState(false);
  const total = useRef(60);

  useEffect(() => {
    if (!sig) return;
    total.current = Math.max(1, sig.expires_at - Date.now() / 1000);
    setNow(Date.now() / 1000);
    if (useStore.getState().tab !== "terminal") say("Бот нашёл сделку и ждёт подтверждения — откройте «Терминал»");
    const t = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(t);
  }, [sig?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!sig || tab !== "terminal") return null;
  const left = Math.max(0, Math.ceil(sig.expires_at - now));
  const answer = async (yes: boolean) => {
    setBusy(true);
    await actions.answerSignal(sig.id, yes);
    setBusy(false);
  };
  const stat = (s: { win_rate: number; trades: number; ev: number }) =>
    s.trades ? (
      <>
        прибыльных <b>{fmtN(s.win_rate * 100, 1)}%</b> · в среднем <b>{usd(s.ev)}</b> за сделку · {s.trades}{" "}
        {plural(s.trades, ["сделка", "сделки", "сделок"])}
      </>
    ) : (
      "нет данных"
    );
  return (
    <div className="sheet" role="dialog" aria-labelledby="sheetTitle" key={sig.id}>
      <h3 id="sheetTitle">
        <span className="coin" style={{ width: 30, height: 30 }}>
          {base(sig.symbol)}
        </span>
        <span>
          Открыть {base(sig.symbol)} {sig.side === "long" ? "LONG" : "SHORT"}?
        </span>
        <span className="cd">
          <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
            <circle cx="11" cy="11" r="9" fill="none" style={{ stroke: "var(--line)" }} strokeWidth="3" />
            <circle
              cx="11"
              cy="11"
              r="9"
              fill="none"
              style={{ stroke: "var(--brand)", transition: "stroke-dashoffset 1s linear" }}
              strokeWidth="3"
              strokeDasharray="56.5"
              strokeDashoffset={56.5 * (1 - left / total.current)}
              strokeLinecap="round"
            />
          </svg>
          {left} с
        </span>
      </h3>
      <p className="sub">
        Вход {price(sig.entry)} · SL {price(sig.stop_loss)} · TP {price(sig.take_profit)} · уверенность {sig.confidence.toFixed(2)}
      </p>
      <dl>
        <dt>Стратегия</dt>
        <dd>{sig.strategy ? stratName(sig.strategy) : "—"}</dd>
        {sig.reasoning && (
          <>
            <dt>Почему</dt>
            <dd>{sig.reasoning}</dd>
          </>
        )}
        <dt>В live</dt>
        <dd>{stat(sig.live)}</dd>
        <dt>Бэктест</dt>
        <dd>{stat(sig.backtest)}</dd>
        {risk > 0 && (
          <>
            <dt>Риск</dt>
            <dd>
              до <b>${fmtN(balance * risk)}</b> при срабатывании стопа ({fmtN(risk * 100, 1)}% баланса)
            </dd>
          </>
        )}
        <dt>Не ответите</dt>
        <dd>{sig.on_timeout_execute ? "бот откроет сделку сам" : "сделка будет пропущена"}</dd>
      </dl>
      <div className="row">
        <button className="btn" type="button" disabled={busy} onClick={() => void answer(false)}>
          Пропустить
        </button>
        <button className="btn primary" type="button" disabled={busy} onClick={() => void answer(true)}>
          Открыть сделку
        </button>
      </div>
    </div>
  );
}

// ── Тост ──────────────────────────────────────────────────────────────────────

function Toast() {
  const toast = useStore((s) => s.toast);
  const [show, setShow] = useState(false);
  useEffect(() => {
    if (!toast) return;
    setShow(true);
    const t = setTimeout(() => setShow(false), 3200);
    return () => clearTimeout(t);
  }, [toast]);
  return (
    <div className={`toast${show ? " show" : ""}`} role="status">
      {toast?.text}
    </div>
  );
}

// ── Вход: код из Telegram (десктоп и браузер) ─────────────────────────────────

function Login() {
  const requestCode = useStore((s) => s.requestCode);
  const verifyCode = useStore((s) => s.verifyCode);
  const [server, setServer] = useState(platform.apiBase());
  const [sent, setSent] = useState(false);
  const [busy, setBusy] = useState(false);
  const desktop = platform.kind === "desktop";
  const miniapp = platform.kind === "miniapp";

  const request = async (e: FormEvent) => {
    e.preventDefault();
    if (desktop) platform.setApiBase(server.trim());
    setBusy(true);
    setSent(await requestCode());
    setBusy(false);
  };
  const verify = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const code = String(new FormData(e.currentTarget).get("code") ?? "").replace(/\D/g, "");
    setBusy(true);
    await verifyCode(code);
    setBusy(false);
  };

  return (
    <div className="login">
      <div className="card">
        <h1>
          {Icon.logo}
          BitbotBY
        </h1>
        {miniapp ? (
          <p>Не удалось войти через Telegram. Закройте приложение и откройте его заново из своего бота.</p>
        ) : !sent ? (
          <form onSubmit={request} style={{ display: "contents" }}>
            <p>Управлять ботом может только владелец. Мы пришлём 6-значный код в ваш Telegram-бот.</p>
            {desktop && (
              <label className="kf">
                <span style={{ fontSize: 12, color: "var(--ink-2)" }}>Адрес сервера бота</span>
                <input className="inp" value={server} onChange={(e) => setServer(e.target.value)} placeholder="https://мой-сервер:8080" required />
              </label>
            )}
            <button className="btn primary" type="submit" disabled={busy}>
              Получить код в Telegram
            </button>
          </form>
        ) : (
          <form onSubmit={verify} style={{ display: "contents" }}>
            <p>Код отправлен в Telegram. Он действует 5 минут.</p>
            <input className="inp code" name="code" inputMode="numeric" autoComplete="one-time-code" maxLength={6} autoFocus aria-label="Код из Telegram" />
            <button className="btn primary" type="submit" disabled={busy}>
              Войти
            </button>
            <button className="btn" type="button" disabled={busy} onClick={() => setSent(false)}>
              Отправить код ещё раз
            </button>
          </form>
        )}
      </div>
      <Toast />
    </div>
  );
}
