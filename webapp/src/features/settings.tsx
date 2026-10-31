// Вкладка «Настройки»: основное, AI-провайдеры, подключения (ключи), все параметры.

import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";

import { api } from "../lib/api";
import { OWN_HOME, type SettingsPane } from "../lib/where";
import { fmtN, plural } from "../lib/format";
import { inHours } from "../lib/idle";
import type { BotNode, ConfigHistoryItem, ConfigValues, SecretItem, SettingDef, TgStatus } from "../lib/types";
import { actions, useStore, type PatchResult } from "../store";
import { Icon, Q, Seg, Toggle, useArm } from "../ui";
import { cfg } from "./bots";

type Pane = SettingsPane;

/** Прокрутка к разделу страницы настроек (вместо переключения «папок»). */
const goTo = (pane: Pane | "map") => {
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  document.getElementById(`set-${pane}`)?.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
};

/** Заголовок раздела страницы настроек: номер, название, одна фраза о том, что внутри. */
function SecHead({ n, title, sub }: { n: string; title: string; sub: string }) {
  return (
    <div className="set-h">
      <span className="set-n">{n}</span>
      <div>
        <h2>{title}</h2>
        <p>{sub}</p>
      </div>
    </div>
  );
}

/** Одна страница вместо разделов-папок: карта подключений, основное, ключи и (в «Профи») все параметры. */
export function Settings() {
  const simple = useStore((s) => s.ui === "simple");
  const jump = useStore((s) => s.paramSearch);
  // Cmd/Ctrl+K: сразу к «Все параметры» с курсором в поиске.
  useEffect(() => {
    if (jump !== null) setTimeout(() => goTo("all"), 0);
  }, [jump]);
  // Подсвечиваем пилюлю раздела, который сейчас на экране.
  const [active, setActive] = useState<Pane | "map">("map");
  // Ключ, открытый в боковой панели: с карты или из списка связок.
  const [key, setKey] = useState<string | null>(null);
  useEffect(() => {
    const root = document.querySelector("main.views");
    const seen = new Map<string, number>();
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) seen.set(e.target.id.replace("set-", ""), e.isIntersecting ? e.intersectionRatio : 0);
        const top = [...seen.entries()].sort((a, b) => b[1] - a[1])[0];
        if (top && top[1] > 0) setActive(top[0] as Pane | "map");
      },
      { root, threshold: [0, 0.25, 0.5, 0.75, 1] },
    );
    for (const id of ["map", "main", "conn", "all"]) {
      const el = document.getElementById(`set-${id}`);
      if (el) io.observe(el);
    }
    return () => io.disconnect();
  }, [simple]);
  const chips: [Pane | "map", string][] = [
    ["map", "Карта"],
    ["main", "Уведомления и аккаунт"],
    ["conn", "Ключи и доступ"],
    ...(simple ? [] : ([["all", "Все параметры"]] as [Pane, string][])),
  ];
  return (
    <section className="view enter" aria-label="Настройки">
      <nav className="set-jump" aria-label="Разделы настроек">
        {chips.map(([id, label]) => (
          <button key={id} type="button" className={`btn${active === id ? " on" : ""}`} aria-current={active === id ? "true" : undefined} onClick={() => goTo(id)}>
            {label}
          </button>
        ))}
      </nav>
      <ConnMap go={goTo} openKey={setKey} />
      <div id="set-main" className="set-sec">
        <SecHead n="01" title="Уведомления и аккаунт" sub="Что присылать в Telegram и когда молчать, счёт, перезапуск и компьютер с ботом" />
        <Main openKey={setKey} />
      </div>
      <div id="set-conn" className="set-sec">
        <SecHead n="02" title="Ключи и доступ" sub="Ключи хранятся на сервере бота зашифрованными — приложение видит только последние 4 символа" />
        <Keys open={key} setOpen={setKey} />
      </div>
      {!simple && (
        <div id="set-all" className="set-sec">
          <SecHead n="03" title="Все параметры" sub="Любой параметр бота: поиск, история изменений, копирование и сброс" />
          <AllParams go={goTo} />
        </div>
      )}
    </section>
  );
}

// ══ Основное ═════════════════════════════════════════════════════════════════

function Main({ openKey }: { openKey: (service: string) => void }) {
  return (
    <div className="set-pane view enter">
      <div className="cols">
        <div className="col">
          <AlertsCard />
          <TgChannelsCard />
          <p className="note-soon moved">
            Часы торговли и монеты теперь в узле «Сканер».{" "}
            <button className="btn" type="button" onClick={() => useStore.getState().openNode("scan")}>
              Цепочка → Сканер
            </button>
          </p>
        </div>
        <div className="col">
          <AccountCard openKey={openKey} />
          <ComputerCard />
        </div>
      </div>
    </div>
  );
}

interface Launcher {
  local: boolean;
  autostart: boolean | null;
  keep_awake: boolean | null;
  phone: boolean | null;
  at: number | null;
  /** Помощник лаунчера на компьютере жив и забирает запросы. */
  helper: boolean;
  result: { id: string; ok: boolean; message: string; at: number } | null;
}

const MACHINE = [
  ["autostart", "Запуск при входе в систему", "autostart"],
  ["keep_awake", "Не давать компьютеру уснуть", "awake"],
  ["phone", "MiniApp в Telegram (доступ с телефона)", "phone"],
] as const;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Компьютер, на котором работает бот. Приложение в Docker кладёт запрос в общую папку,
 * помощник лаунчера выполняет только разрешённое; телефон отсюда можно лишь выключить.
 */
function ComputerCard() {
  const [st, setSt] = useState<Launcher | null>(null);
  const [busy, setBusy] = useState("");
  const [copied, setCopied] = useState("");
  const load = () =>
    api<Launcher>("GET", "/api/launcher")
      .then((v) => (setSt(v), v))
      .catch(() => null);
  useEffect(() => {
    void load();
  }, []);
  if (!st?.local) return null;
  const say = useStore.getState().say;
  const copy = async (cmd: string) => {
    try {
      await navigator.clipboard.writeText(cmd);
      setCopied(cmd);
    } catch {
      setCopied("");
    }
  };
  const toggle = async (setting: string, on: boolean) => {
    setBusy(setting);
    try {
      const { id } = await api<{ id: string }>("POST", "/api/launcher", { setting, on });
      // Помощник смотрит папку раз в 2 с; выключение телефона ещё и перезапускает приложение.
      for (let i = 0; i < 40; i++) {
        await sleep(1500);
        const v = await load();
        if (v?.result?.id === id) return say(v.result.message || (v.result.ok ? "Готово" : "Не получилось"));
      }
      say("Компьютер не ответил. Проверьте, что бот запущен ярлыком BitbotBY.");
    } catch (e) {
      say(e instanceof Error ? e.message : "Не получилось.");
    } finally {
      setBusy("");
    }
  };
  return (
    <article className="card">
      <div className="card-h">
        <h2>Компьютер</h2>
        <Q tip="Бот работает в Docker на этом компьютере. Тумблеры передают просьбу помощнику лаунчера на компьютере — он умеет только эти действия. Доступ с телефона отсюда можно лишь выключить: включение открывает приложение в интернет, поэтому только командой на самом компьютере." />
      </div>
      {st.at === null && <p className="note-soon">Лаунчер ещё не сообщил о себе — перезапустите ярлыком BitbotBY, и здесь появится состояние.</p>}
      {st.at !== null && !st.helper && (
        <p className="note-soon">Помощник на компьютере не отвечает — переключать можно командой в Терминале. Он запускается вместе с ботом ярлыком BitbotBY.</p>
      )}
      <ul className="machine">
        {MACHINE.map(([k, label, cmd]) => {
          const on = st[k];
          const can = st.helper && on !== null && (k !== "phone" || on);
          const next = `python3 launcher/launch.py ${cmd} ${on ? "off" : "on"}`;
          return (
            <li key={k}>
              <span className={`dot${on ? " ok" : ""}`} />
              <span>
                {label}: <b>{busy === k ? "меняю…" : on === null ? "неизвестно" : on ? "включено" : "выключено"}</b>
              </span>
              <span className="machine-act">
                {can ? (
                  <Toggle on={Boolean(on)} label={label} disabled={busy !== ""} onChange={(v) => void toggle(k, v)} />
                ) : (
                  <button className="btn mono" type="button" title="Скопировать команду" onClick={() => void copy(next)}>
                    {copied === next ? "Скопировано ✓" : next}
                  </button>
                )}
              </span>
              {k === "phone" && !on && st.helper && <p className="machine-note">Включается только на компьютере: это открывает приложение в интернет.</p>}
            </li>
          );
        })}
      </ul>
      <p className="note-soon">Нет доступа к Терминалу? Ссылку входа присылает бот в Telegram по команде /app.</p>
    </article>
  );
}

const HOURS = [
  ["", "24/7"],
  ["1-10", "Азия"],
  ["8-20", "Европа"],
  ["14-22", "США"],
  ["22-6", "Ночь"],
] as const;

/** Живёт в узле «Сканер» вкладки «Бот». */
export function HoursCard() {
  const hours = useStore((s) => cfg<string>(s.config, "TRADING_HOURS", ""));
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 60_000);
    return () => clearInterval(t);
  }, []);
  const on = (i: number) => inHours(hours, i);
  const h = now.getUTCHours();
  const custom = !HOURS.some(([v]) => v === hours);
  return (
    <article className="card">
      <div className="card-h">
        <h2>Часы торговли</h2>
        <Q tip="Бот открывает новые сделки только в эти часы (UTC). Открытые позиции сопровождаются круглосуточно." />
        <span className="label sp">
          сейчас {String(h).padStart(2, "0")}:{String(now.getUTCMinutes()).padStart(2, "0")} UTC
        </span>
      </div>
      <Seg
        className="full"
        value={custom ? null : hours}
        options={HOURS}
        onChange={(v) => void actions.patchConfig({ TRADING_HOURS: v })}
        label="Часы торговли"
      />
      {custom && <p className="note-soon" style={{ marginTop: 8 }}>Свой диапазон: {hours} UTC (из Telegram /hours или «Все параметры»).</p>}
      <div className="hours-bar">
        {Array.from({ length: 24 }, (_, i) => (
          <i key={i} className={`${on(i) ? "on" : ""}${i === h ? " now" : ""}`} title={`${String(i).padStart(2, "0")}:00 UTC`} />
        ))}
      </div>
      <div className="hours-axis">
        <span>00</span>
        <span>06</span>
        <span>12</span>
        <span>18</span>
        <span>24 UTC</span>
      </div>
    </article>
  );
}

const STATUS: Record<SecretItem["status"], [string, string]> = {
  ok: ["ok", "работает"],
  warn: ["warn", "есть проблема"],
  bad: ["bad", "не работает"],
  unset: ["", "не настроен"],
  unchecked: ["", "не проверен"],
};

const LIST_NAME = { forced: "Всегда сканировать", excluded: "Исключить из скана" } as const;

/** Живёт в узле «Сканер» вкладки «Бот». */
export function SymbolsCard() {
  const symbols = useStore((s) => s.symbols);
  return (
    <article className="card">
      <div className="card-h">
        <h2>Монеты</h2>
        <Q tip="«Всегда сканировать» — бот проверяет эти монеты в каждом цикле, даже если их нет в топе по объёму (/add в Telegram). «Исключить» — бот никогда ими не торгует (/remove)." />
      </div>
      {(["forced", "excluded"] as const).map((l) => (
        <SymbolList key={l} list={l} items={symbols[l]} />
      ))}
    </article>
  );
}

function SymbolList({ list, items }: { list: "forced" | "excluded"; items: string[] }) {
  const symbols = useStore((s) => s.symbols);
  const [err, setErr] = useState("");
  const other = list === "forced" ? "excluded" : "forced";
  const save = (next: string[], otherNext = symbols[other]) =>
    actions.putSymbols({ ...symbols, [list]: next, [other]: otherNext } as typeof symbols);
  const add = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const form = e.currentTarget;
    const v = String(new FormData(form).get("t") ?? "")
      .trim()
      .toUpperCase()
      .replace(/\/?USDT$/, "");
    if (!/^[A-Z0-9]{2,10}$/.test(v)) return setErr("Введите тикер из 2–10 латинских букв или цифр, например PEPE.");
    const sym = `${v}/USDT`;
    if (items.includes(sym)) return setErr(`${v} уже в этом списке.`);
    setErr("");
    if (
      await save(
        [...items, sym],
        symbols[other].filter((s) => s !== sym),
      )
    )
      form.reset();
  };
  return (
    <div className="symlist">
      <h3>
        {LIST_NAME[list]} <span className="pill">{items.length}</span>
      </h3>
      <div className="syms">
        {items.length ? (
          items.map((s) => (
            <span className="sym" key={s}>
              {s.replace("/USDT", "")}
              <button type="button" aria-label={`Убрать ${s} из списка «${LIST_NAME[list]}»`} onClick={() => void save(items.filter((x) => x !== s))}>
                ✕
              </button>
            </span>
          ))
        ) : (
          <span className="muted" style={{ fontSize: 12 }}>
            Список пуст
          </span>
        )}
      </div>
      <form className="add" noValidate onSubmit={add}>
        <input className="inp" name="t" placeholder={list === "forced" ? "Тикер, например SOL" : "Тикер, например HYPE"} autoComplete="off" aria-label={`Добавить монету в «${LIST_NAME[list]}»`} />
        <button className={`btn${list === "forced" ? " primary" : ""}`} type="submit">
          {list === "forced" ? "Добавить" : "Исключить"}
        </button>
        {err && <span className="err">{err}</span>}
      </form>
    </div>
  );
}

// Тихие часы выбираются по местному времени, а бот хранит их в UTC (QUIET_HOURS).
const TZ = Math.round(-new Date().getTimezoneOffset() / 60);
const shiftHour = (h: number, by: number) => (((h + by) % 24) + 24) % 24;
const shiftRange = (r: string, by: number) => (r ? r.split("-").map((h) => shiftHour(Number(h), by)).join("-") : "");
const QUIET = [
  ["", "Без тишины"],
  ["23-8", "23–08"],
  ["0-7", "00–07"],
  ["22-9", "22–09"],
] as const;
const LEVEL: Record<string, string> = { critical: "приходит всегда", trade: "сделки", info: "сведения" };

function AlertsCard() {
  const alerts = useStore((s) => s.alerts);
  const quietUtc = useStore((s) => cfg<string>(s.config, "QUIET_HOURS", ""));
  const quietLocal = shiftRange(quietUtc, TZ);
  const preset = QUIET.find(([v]) => v === quietLocal)?.[0] ?? null;
  return (
    <article className="card">
      <div className="card-h">
        <h2>Уведомления в Telegram</h2>
        <Q tip="Те же виды, что в меню /alerts. Пауза не даёт одному и тому же уведомлению сыпаться чаще, чем раз в N минут." />
      </div>
      <div className="field">
        <div className="field-h">
          <span>
            Тихие часы, по вашему времени{" "}
            <Q tip="В эти часы приходят только уведомления с пометкой «приходит всегда»: бот не отвечает, просадка, дневной лимит, всплеск волатильности, позиция без защиты. Сделки и сигналы — утром." />
          </span>
          {quietUtc && <span className="pill sp">{preset === null ? `свои: ${quietLocal.replace("-", "–")}` : `UTC ${quietUtc.replace("-", "–")}`}</span>}
        </div>
        <Seg className="full" value={preset} options={QUIET} label="Тихие часы" onChange={(v) => void actions.patchConfig({ QUIET_HOURS: shiftRange(v, -TZ) })} />
      </div>
      <div className="list-tgl">
        {alerts.map((a) => (
          <div className={`alert-row${a.enabled ? "" : " off"}`} key={a.kind}>
            <div>
              <b>{a.label}</b>
              {a.level && <span className={`lvl ${a.level}`}>{LEVEL[a.level]}</span>}
            </div>
            <Toggle on={a.enabled} label={a.label} onChange={(on) => void actions.patchAlert(a.kind, { enabled: on })} />
            <label className="cool">
              <span>Не чаще раза в</span>
              <span className="numf">
                <input
                  key={a.cooldown_min}
                  type="number"
                  min="0"
                  max="1440"
                  step="1"
                  defaultValue={a.cooldown_min}
                  inputMode="numeric"
                  disabled={!a.enabled}
                  aria-label={`Пауза для «${a.label}», минут`}
                  onBlur={(e) => {
                    const v = Math.round(Number(e.target.value));
                    if (v !== a.cooldown_min) void actions.patchAlert(a.kind, { cooldown_min: v });
                  }}
                  onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
                />
                <span className="u">мин</span>
              </span>
              <span>{a.cooldown_min ? "" : "каждый раз"}</span>
            </label>
          </div>
        ))}
      </div>
    </article>
  );
}

// ── Каналы Telegram: какие читать и каким доверять ────────────────────────────

const TRUST_PARAMS = ["TELEGRAM_TRUSTED_RISK", "TELEGRAM_TRUSTED_SLOTS", "TELEGRAM_TRUSTED_SL", "TELEGRAM_TRUSTED_DEDUP_MIN", "TELEGRAM_TRUSTED_UNTRUST"];
/** Как channel_names на сервере: «@Chan, https://t.me/chan2» → ["chan", "chan2"]. */
const chanList = (raw: unknown): string[] => [
  ...new Set(
    String(raw ?? "")
      .split(/[,\s]+/)
      .map((c) => c.replace(/^(?:https?:\/\/)?(?:t\.me\/|telegram\.me\/)?@?/, "").replace(/\/+$/, "").toLowerCase())
      .filter(Boolean),
  ),
];

/** Каналы, которые бот читает, и доверенные: пост доверенного канала сразу становится сделкой. */
function TgChannelsCard() {
  const config = useStore((s) => s.config);
  const schema = useStore((s) => s.schema);
  const paper = useStore((s) => s.bot?.paper ?? true);
  const [add, setAdd] = useState("");
  const on = cfg<boolean>(config, "TELEGRAM_TRUSTED_TRADING", false);
  const watched = chanList(cfg(config, "TELEGRAM_SENTIMENT_CHANNELS", ""));
  const trusted = chanList(cfg(config, "TELEGRAM_TRUSTED_CHANNELS", ""));
  const all = [...new Set([...watched, ...trusted])];
  const trust = (ch: string, yes: boolean) =>
    void actions.patchConfig({ TELEGRAM_TRUSTED_CHANNELS: (yes ? [...trusted, ch] : trusted.filter((c) => c !== ch)).join(",") });
  const remove = (ch: string) =>
    void actions.patchConfig({
      TELEGRAM_SENTIMENT_CHANNELS: watched.filter((c) => c !== ch).join(","),
      TELEGRAM_TRUSTED_CHANNELS: trusted.filter((c) => c !== ch).join(","),
    });
  const addChannel = (e: FormEvent) => {
    e.preventDefault();
    const fresh = chanList(add).filter((c) => !watched.includes(c));
    if (fresh.length) void actions.patchConfig({ TELEGRAM_SENTIMENT_CHANNELS: [...watched, ...fresh].join(",") });
    setAdd("");
  };
  const params = TRUST_PARAMS.map((k) => schema.find((d) => d.key === k)).filter((d): d is SettingDef => Boolean(d));
  return (
    <article className="card tgch" id="tg-channels">
      <div className="card-h">
        <h2>Каналы Telegram</h2>
        <Q tip="Посты всех каналов идут в оценку настроения по монетам текущего скана. Доверенные — ещё и сразу в сделку. Вход в Telegram — на карточке ключа «Памп-каналы Telegram»." />
      </div>
      <div className="alert-row">
        <div>
          <b>Сделки по доверенным каналам</b>
          <small>в любом профиле, мимо сканера, стратегий и AI</small>
        </div>
        <Toggle on={on} label="Сделки по доверенным каналам" onChange={(v) => void actions.patchConfig({ TELEGRAM_TRUSTED_TRADING: v })} />
      </div>
      <p className="kf-h">
        Пост доверенного канала сразу становится сделкой: стратегии и AI его не проверяют и не закрывают — только стоп, тейк и трейлинг. Защиты остаются: стоп
        всегда, свой риск и слоты, пауза, дневной лимит, спред, одна сделка на монету, режим счёта — сейчас {paper ? "Paper" : "Live"}.
      </p>
      <div className="list-tgl">
        {all.map((ch) => (
          <div className={`alert-row${trusted.includes(ch) ? "" : " off"}`} key={ch}>
            <div>
              <b>@{ch}</b>
              <small>{trusted.includes(ch) ? "Доверять: входить по постам" : "только настроение"}</small>
            </div>
            <span className="tgch-a">
              <Toggle on={trusted.includes(ch)} label={`Доверять @${ch}: входить по постам`} onChange={(v) => trust(ch, v)} />
              <button type="button" className="btn" aria-label={`Убрать @${ch}`} onClick={() => remove(ch)}>
                Убрать
              </button>
            </span>
          </div>
        ))}
      </div>
      <form className="tgch-add" onSubmit={addChannel}>
        <input className="inp" aria-label="Добавить канал" placeholder="@канал или ссылка t.me/…" value={add} onChange={(e) => setAdd(e.target.value)} />
        <button className="btn" type="submit" disabled={!add.trim()}>
          Добавить
        </button>
      </form>
      <div className="tgch-params">
        {params.map((d) => (
          <label className="cool" key={d.key}>
            <span>{d.label}</span>
            <span className="numf">
              <input
                key={String(config[d.key]?.value)}
                type="number"
                inputMode="decimal"
                min={d.min === null ? undefined : Number(shown(d, d.min))}
                max={d.max === null ? undefined : Number(shown(d, d.max))}
                defaultValue={String(shown(d, config[d.key]?.value ?? d.default))}
                onBlur={(e) => {
                  const v = Number(e.target.value.replace(",", "."));
                  const raw = d.type === "pct" ? +(v / 100).toFixed(6) : v;
                  if (e.target.value && Number.isFinite(v) && raw !== config[d.key]?.value) void actions.patchConfig({ [d.key]: raw });
                }}
                onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
              />
              <span className="u">{unitOf(d)}</span>
            </span>
          </label>
        ))}
      </div>
    </article>
  );
}

type Link = "ok" | "warn" | "bad" | "off";
interface MapNode {
  id: string;
  label: string;
  sub: string;
  st: Link;
  act?: () => void;
}

/** Карта подключений: что питает бота слева, бот в центре, что он отдаёт справа. По живым связям бегут точки. */
function ConnMap({ go, openKey }: { go: (pane: Pane | "map") => void; openKey: (service: string) => void }) {
  const secrets = useStore((s) => s.secrets);
  const ai = useStore((s) => s.ai);
  const note = useStore((s) => s.secretsNote);
  const wsUp = useStore((s) => s.wsUp);
  const alive = useStore((s) => s.alive);
  const paper = useStore((s) => s.bot?.paper ?? true);
  // Где работает бот: лаунчер на компьютере отвечает local=true, на сервере его нет.
  const [local, setLocal] = useState<boolean | null>(null);
  useEffect(() => {
    api<{ local: boolean }>("GET", "/api/launcher")
      .then((v) => setLocal(v.local))
      .catch(() => setLocal(false));
  }, []);
  const lastAi = ai?.providers.find((p) => p.id === ai.last);
  const by = (id: string) => secrets?.find((s) => s.service === id);
  const link = (item?: SecretItem): Link => {
    const c = item ? STATUS[item.status][0] : "";
    return c === "ok" ? "ok" : c === "bad" ? "bad" : c === "warn" ? "warn" : "off";
  };
  const sub = (item?: SecretItem) => {
    if (!item) return "не подключено";
    const last4 = item.fields.find((f) => f.secret && f.last4)?.last4;
    return `${STATUS[item.status][1]}${last4 ? ` · ключ ••••${last4}` : ""}`;
  };
  const toKey = (service: string) => () => (by(service) ? openKey(service) : go("conn"));
  const data = (secrets ?? []).filter((s) => s.group === "Данные и соцсети");
  const dOk = data.filter((s) => s.status === "ok").length;
  const tg = link(by("telegram"));
  const server: Link = wsUp && alive ? "ok" : wsUp ? "warn" : "bad";
  const src: MapNode[] = [
    { id: "bybit", label: "Bybit", sub: sub(by("bybit")), st: link(by("bybit")), act: toKey("bybit") },
    {
      id: "ai",
      label: "AI-провайдеры",
      sub: ai?.chain.length ? `в цепочке ${ai.chain.length}${lastAi ? ` · отвечает ${lastAi.name}` : ""}` : "цепочка пуста",
      st: ai?.chain.length ? "ok" : "bad",
      act: () => useStore.getState().openNode("ai"),
    },
    { id: "telegram", label: "Telegram-бот", sub: sub(by("telegram")), st: tg, act: toKey("telegram") },
    { id: "data", label: "Данные и соцсети", sub: `подключено ${dOk} из ${data.length}`, st: dOk ? "ok" : "off", act: () => go("conn") },
  ];
  const out: MapNode[] = [
    {
      id: "trades",
      label: paper ? "Сделки · Paper" : "Сделки · Live",
      sub: paper ? "виртуальные, без реальных денег" : "реальные деньги на Bybit",
      st: server === "ok" ? (paper ? "ok" : "warn") : "off",
      act: () => useStore.getState().setTab("journal"),
    },
    { id: "notify", label: "Уведомления", sub: tg === "ok" ? "сделки и защиты — в Telegram" : "подключите Telegram", st: tg === "ok" ? "ok" : "off", act: () => go("main") },
    { id: "app", label: "Приложение", sub: "MiniApp, браузер, десктоп", st: server === "ok" ? "ok" : "warn" },
  ];
  // Обязательное для работы: бот на связи, биржа, AI и Telegram; источники данных — по желанию.
  const need = [
    { label: "бот", ok: server === "ok" },
    { label: "Bybit", ok: src[0].st === "ok" },
    { label: "AI", ok: src[1].st === "ok" },
    { label: "Telegram", ok: src[2].st === "ok" },
  ];
  const ready = need.filter((n) => n.ok).length;
  const missing = need.filter((n) => !n.ok).map((n) => n.label);
  const lines = (nodes: MapNode[], dir: "in" | "out") =>
    nodes.map((n, i) => {
      const y = ((i + 0.5) / nodes.length) * 100;
      const d = dir === "in" ? `M0 ${y} C 55 ${y}, 45 50, 100 50` : `M0 50 C 55 50, 45 ${y}, 100 ${y}`;
      return (
        <g key={n.id}>
          <path className={`cm-base ${n.st}`} d={d} />
          <path className={`cm-dots ${n.st}`} d={d} pathLength={100} />
        </g>
      );
    });
  const node = (n: MapNode) => {
    const body = (
      <>
        <span className={`dot ${n.st === "ok" ? "ok live" : n.st === "off" ? "" : n.st}`} />
        <b>{n.label}</b>
        <small>{n.sub}</small>
      </>
    );
    return (
      <li key={n.id}>
        {n.act ? (
          <button type="button" className={`cm-node ${n.st}`} onClick={n.act}>
            {body}
          </button>
        ) : (
          <div className={`cm-node ${n.st}`}>{body}</div>
        )}
      </li>
    );
  };
  return (
    <article className="card cmap" id="set-map">
      <div className="cm-head">
        <div>
          <span className="label">Система</span>
          <h2>{missing.length ? `Осталось подключить: ${missing.join(", ")}` : "Всё подключено"}</h2>
          <p>Что питает бота → бот → что он отдаёт. Нажмите на любой узел карты, чтобы настроить его.</p>
        </div>
        <div className="learn-meter" style={{ "--p": Math.round((ready / need.length) * 100) } as React.CSSProperties} role="img" aria-label={`Подключено ${ready} из ${need.length}`}>
          <b className="num">
            {ready}
            <small>/{need.length}</small>
          </b>
          <span>подключено</span>
        </div>
      </div>
      <div className="cm">
        <ul className="cm-col">{src.map(node)}</ul>
        <svg className="cm-links" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
          {lines(src, "in")}
        </svg>
        <div className={`cm-hub ${server}`}>
          <span className="cm-hub-ring">{Icon.logo}</span>
          <b>BitbotBY</b>
          <small>{wsUp ? (alive ? "работает" : "бот не отвечает") : "нет связи с сервером"}</small>
          {local !== null && <small className="cm-where">{local ? "на этом компьютере" : "на сервере"}</small>}
        </div>
        <svg className="cm-links" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
          {lines(out, "out")}
        </svg>
        <ul className="cm-col">{out.map(node)}</ul>
      </div>
      {!secrets && <p className="note-soon">{note || "Статус ключей недоступен."}</p>}
    </article>
  );
}

function AccountCard({ openKey }: { openKey: (service: string) => void }) {
  const paper = useStore((s) => s.bot?.paper ?? true);
  const secrets = useStore((s) => s.secrets);
  const risk = useStore((s) => cfg(s.config, "RISK_PER_TRADE", 0.02));
  const slots = useStore((s) => cfg(s.config, "MAX_POSITIONS", 3));
  const balance = useStore((s) => s.balance);
  const [box, setBox] = useState(false);
  const [busy, setBusy] = useState(false);
  const [restartArmed, armRestart] = useArm();
  const bybitOk = !secrets || secrets.find((s) => s.service === "bybit")?.status === "ok";
  const pinSet = !secrets || Boolean(secrets.find((s) => s.service === "server")?.fields.find((f) => f.env === "TRADING_MODE_PIN")?.set);

  const goLive = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const pin = String(new FormData(e.currentTarget).get("pin") ?? "");
    setBusy(true);
    if (await actions.accountMode(false, pin)) setBox(false);
    setBusy(false);
  };
  const toPaper = async () => {
    setBusy(true);
    await actions.accountMode(true);
    setBusy(false);
  };

  return (
    <article className="card">
      <div className="card-h">
        <h2>Счёт и бот</h2>
      </div>
      <div className="row-tgl">
        <span>
          <span className={`paper${paper ? "" : " live"}`}>{paper ? "PAPER" : "LIVE"}</span> {paper ? "Виртуальные сделки" : "Реальные деньги на Bybit"}
        </span>
        <span className={`pill ${paper ? "ok" : "bad"}`}>
          <span className={`dot ${paper ? "ok" : "bad"}`} />
          активно
        </span>
      </div>
      <p className="muted" style={{ fontSize: 12, margin: "10px 0" }}>
        {paper
          ? "Деньги не тратятся. Переход на реальные деньги защищён PIN и подтверждением на сервере."
          : "Бот торгует реальными деньгами. Вернуться в Paper можно в любой момент: открытые позиции на бирже останутся, бот продолжит их сопровождать."}
      </p>
      {paper ? (
        <button className="btn danger" type="button" onClick={() => setBox(!box)}>
          Перейти на Live…
        </button>
      ) : (
        <button className="btn" type="button" disabled={busy} onClick={() => void toPaper()}>
          Вернуться в Paper
        </button>
      )}
      {paper && box && (
        <div className="confirm-box">
          {!bybitOk ? (
            <>
              <b>Сначала подключите Bybit.</b>
              <span>Ключ Bybit не проверен — бот не сможет выставлять ордера.</span>
              <div className="btn-row">
                <button className="btn primary" type="button" onClick={() => openKey("bybit")}>
                  Открыть ключ Bybit
                </button>
                <button className="btn" type="button" onClick={() => setBox(false)}>
                  Отмена
                </button>
              </div>
            </>
          ) : !pinSet ? (
            <>
              <b>Сначала задайте PIN.</b>
              <span>Без PIN переход на реальные деньги невозможен. Он хранится в «Ключи и доступ → Пароли и PIN», его же спрашивает Telegram-бот.</span>
              <div className="btn-row">
                <button className="btn primary" type="button" onClick={() => openKey("server")}>
                  Задать PIN
                </button>
                <button className="btn" type="button" onClick={() => setBox(false)}>
                  Отмена
                </button>
              </div>
            </>
          ) : (
            <form onSubmit={goLive} style={{ display: "contents" }}>
              <b>Бот начнёт торговать реальными деньгами.</b>
              <span>
                До {slots} {plural(slots, ["сделки", "сделок", "сделок"])} сразу, {fmtN(risk * 100, 1)}% баланса на сделку (≈ ${fmtN(balance * risk, 0)}).
                Бэктест пока не нашёл у стратегий устойчивого преимущества.
              </span>
              <label className="kf">
                <span style={{ fontSize: 12, color: "var(--ink-2)" }}>PIN</span>
                <input className="inp" name="pin" type="password" inputMode="numeric" autoComplete="off" autoFocus />
              </label>
              <div className="btn-row">
                <button className="btn danger" type="submit" disabled={busy}>
                  Перейти на Live
                </button>
                <button className="btn" type="button" onClick={() => setBox(false)}>
                  Отмена
                </button>
              </div>
            </form>
          )}
        </div>
      )}
      <div className="btn-row" style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--line-soft)" }}>
        <button className={`btn${restartArmed ? " armed" : ""}`} type="button" onClick={() => armRestart() && void actions.restart()}>
          {restartArmed ? "Точно перезапустить?" : "Перезапустить бота"}
        </button>
      </div>
    </article>
  );
}

// ══ Ключи API ════════════════════════════════════════════════════════════════

const KEY_INFO: Record<string, [string, string?]> = {
  bybit: [
    "Нужен для торговли и баланса. Создайте ключ только с правами «чтение» и «торговля», без вывода средств, и привяжите его к IP сервера.",
    "https://www.bybit.com/app/user/api-management",
  ],
  groq: ["Бесплатный тир, самый быстрый.", "https://console.groq.com"],
  gemini: ["Бесплатный тир с дневной квотой.", "https://aistudio.google.com"],
  anthropic: ["Лучшее качество анализа, платный.", "https://console.anthropic.com"],
  deepseek: ["Самый дешёвый из платных.", "https://platform.deepseek.com"],
  openai: ["Модели ChatGPT.", "https://platform.openai.com"],
  openrouter: ["Много моделей по одному ключу, есть бесплатные.", "https://openrouter.ai/keys"],
  nvidia: ["Бесплатные модели NVIDIA NIM.", "https://build.nvidia.com"],
  telegram: ["Токен от @BotFather и ваш chat_id (его покажет @userinfobot). Управлять ботом может только этот чат.", "https://t.me/BotFather"],
  news: ["Новости для оценки настроения рынка.", "https://newsapi.org"],
  finnhub: ["Календарь FOMC, CPI, NFP для макро-паузы. Без ключа пауза не работает.", "https://finnhub.io"],
  glassnode: ["On-chain метрики, например запас стейблкоинов.", "https://glassnode.com"],
  reddit: ["Посты r/CryptoCurrency для анализа настроения моделью CryptoBERT.", "https://www.reddit.com/prefs/apps"],
  twitter: ["Настроение по твитам, только профиль Meme. Два способа: официальный платный API или бесплатный обход через аккаунты X. Если указаны оба, работает официальный API.", "https://developer.twitter.com"],
  telethon: ["Посты каналов в реальном времени, во всех профилях: монеты текущего скана ищутся точно ($WIF, #WIF, WIF/USDT). API ID и API Hash — на my.telegram.org, затем «Войти»: код придёт в Telegram.", "https://my.telegram.org"],
  pumpportal: ["Темп запуска новых токенов на pump.fun — признак мем-сезона.", "https://pumpportal.fun"],
  server: ["PIN для перехода в Live, доступ к Redis и API дашборда, секрет вебхука алертов."],
};

/** Что питает каждый ключ: [что именно, шаг цепочки]. Шаг в панели ключа — переход в «Цепочку». */
const FEEDS: Record<string, [string, BotNode?][]> = {
  bybit: [["цены и свечи", "scan"], ["ордера и баланс", "order"]],
  telegram: [["уведомления"], ["управление из чата"]],
  news: [["настроение рынка", "scan"]],
  finnhub: [["макро-пауза: FOMC, CPI, NFP", "risk"]],
  glassnode: [["on-chain контекст", "scan"]],
  reddit: [["настроение · CryptoBERT", "scan"]],
  twitter: [["настроение · Meme", "scan"]],
  telethon: [["каналы Telegram · все профили", "scan"]],
  pumpportal: [["детектор мем-сезона", "scan"]],
  server: [["переход в Live", "order"], ["дашборд и вебхуки"]],
};
const STEP: Partial<Record<BotNode, string>> = { scan: "Сканер", ai: "AI-цепочка", risk: "Риск", order: "Ордер" };

/** Ключи как связки: сервис → что он питает в боте. Форма ключа открывается панелью справа, список остаётся на виду. */
function Keys({ open, setOpen }: { open: string | null; setOpen: (service: string | null) => void }) {
  const secrets = useStore((s) => s.secrets);
  const note = useStore((s) => s.secretsNote);
  const ai = useStore((s) => s.ai);
  const closeBtn = useRef<HTMLButtonElement>(null);
  // Ключи AI-провайдеров настраиваются в узле AI цепочки — здесь одна строка-связка.
  const groups = useMemo(() => [...new Set((secrets ?? []).map((s) => s.group))].filter((g) => g !== "AI-модели"), [secrets]);
  const aiKeys = (secrets ?? []).filter((s) => s.group === "AI-модели");
  const cur = secrets?.find((s) => s.service === open);
  // Панель уезжает с прежним содержимым, а не пустой.
  const last = useRef(cur);
  if (cur) last.current = cur;
  const shown = last.current;
  const isOpen = Boolean(cur);
  useEffect(() => {
    if (!isOpen) return;
    closeBtn.current?.focus({ preventScroll: true });
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(null);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isOpen, setOpen]);
  const toStep = (step: BotNode) => {
    setOpen(null);
    useStore.getState().openNode(step);
  };
  const row = (id: string, c: string, name: string, sub: string, feeds: [string, BotNode?][], act: () => void) => (
    <li key={id}>
      <button type="button" className={`kl-row ${c}`} onClick={act}>
        <span className="kl-src">
          <span className={`dot ${c === "ok" ? "ok live" : c}`} />
          <b>{name}</b>
          <small>{sub}</small>
        </span>
        <span className="kl-wire" aria-hidden="true" />
        <span className="kl-to">
          {feeds.map(([what, step]) => (
            <span className="kl-chip" key={what}>
              {step && <em>{STEP[step]}</em>}
              {what}
            </span>
          ))}
        </span>
        <span className="kl-go" aria-hidden="true">
          ›
        </span>
      </button>
    </li>
  );
  const keyRow = (s: SecretItem) => {
    const [c, t] = STATUS[s.status];
    const last4 = s.fields.find((f) => f.secret && f.last4)?.last4;
    return row(s.service, c, s.name, `${t}${last4 ? ` · ••••${last4}` : ""}`, FEEDS[s.service] ?? [], () => setOpen(s.service));
  };
  const aiOk = aiKeys.filter((s) => s.status === "ok").length;
  return (
    <div className="set-pane view enter">
      {!secrets && <p className="empty">{note || "Ключи недоступны."}</p>}
      <div className="klist">
        {secrets && (
          <section className="kl-g">
            <h3>AI-модели</h3>
            <ul>
              {row(
                "ai",
                ai?.chain.length ? "ok" : "bad",
                "AI-провайдеры",
                `ключей ${aiOk} из ${aiKeys.length}${ai?.chain.length ? ` · в цепочке ${ai.chain.length}` : ""}`,
                [["решение о сделке", "ai"]],
                () => toStep("ai"),
              )}
            </ul>
          </section>
        )}
        {groups.map((g) => (
          <section className="kl-g" key={g}>
            <h3>{g}</h3>
            <ul>{secrets!.filter((s) => s.group === g).map(keyRow)}</ul>
          </section>
        ))}
      </div>
      <button type="button" className={`pipe-scrim${isOpen ? " open" : ""}`} tabIndex={-1} aria-hidden="true" onClick={() => setOpen(null)} />
      <aside className={`pipe-panel${isOpen ? " open" : ""}`} role="dialog" aria-label={shown ? `Ключ «${shown.name}»` : "Ключ"} aria-hidden={!isOpen}>
        {shown && (
          <>
            <div className="pipe-panel-h">
              <span className={`dot ${STATUS[shown.status][0]}`} />
              <b>{shown.name}</b>
              <span className="label">{shown.group}</span>
              <button ref={closeBtn} type="button" className="btn pipe-close" aria-label="Закрыть ключ" onClick={() => setOpen(null)}>
                ✕
              </button>
            </div>
            <div className="pipe-body" key={shown.service}>
              {(FEEDS[shown.service] ?? []).length > 0 && (
                <div className="kl-feeds">
                  <span className="label">Питает в боте</span>
                  <div>
                    {FEEDS[shown.service].map(([what, step]) =>
                      step ? (
                        <button type="button" className="kl-chip" key={what} onClick={() => toStep(step)}>
                          <em>{STEP[step]}</em>
                          {what} →
                        </button>
                      ) : (
                        <span className="kl-chip" key={what}>
                          {what}
                        </span>
                      ),
                    )}
                  </div>
                </div>
              )}
              <KeyCard item={shown} />
              <div className="secnote">
                {Icon.lock}
                <div>
                  <b>Как хранятся ключи.</b> Ключ уходит на сервер вашего бота и хранится там зашифрованным. Приложение больше не показывает
                  его целиком — только последние 4 символа. Перед сохранением сервер проверяет ключ запросом к сервису. Новый ключ начнёт
                  работать после перезапуска бота.
                </div>
              </div>
            </div>
          </>
        )}
      </aside>
    </div>
  );
}

/** Тестовая сеть Bybit — отдельные ключи testnet, ордера без реальных денег. */
function TestnetRow() {
  const on = useStore((s) => cfg<boolean>(s.config, "TESTNET", false));
  return (
    <div className="row-tgl">
      <span>
        Тестовая сеть Bybit (testnet){" "}
        <Q tip="Ордера идут на testnet.bybit.com — нужны отдельные ключи testnet. Заработает после перезапуска бота." />
      </span>
      <Toggle on={on} label="Тестовая сеть Bybit" onChange={(v) => void actions.patchConfig({ TESTNET: v })} />
    </div>
  );
}

/** Проверка формата до отправки: ловит очевидные опечатки, сам ключ проверяет сервер. */
function checkField(env: string, label: string, secret: boolean, v: string): string {
  if (/CHAT_ID|API_ID$/.test(env)) return /^-?\d+$/.test(v) ? "" : `${label} — это число, без букв и пробелов.`;
  if (env === "TELEGRAM_SENTIMENT_PHONE") return /^\+\d{8,15}$/.test(v) ? "" : "Телефон нужен в формате +375291234567.";
  if (env === "TRADING_MODE_PIN") return /^\d{4,12}$/.test(v) ? "" : "PIN — от 4 до 12 цифр.";
  if (secret && v.length < 8) return `«${label}» слишком короткий — скопируйте его целиком.`;
  return "";
}

/** Код привязки: 6 цифр, их видно только здесь. */
const linkCode = () => String(crypto.getRandomValues(new Uint32Array(1))[0] % 1_000_000).padStart(6, "0");

/** Привязка Telegram: вы отправляете боту код из приложения, мы находим ваш chat id и подставляем его в поле. */
function ChatFinder({ token, hasToken, onFound }: { token: string; hasToken: boolean; onFound: (id: string) => void }) {
  const [busy, setBusy] = useState(false);
  const [code] = useState(linkCode);
  const [chats, setChats] = useState<{ id: string; name: string }[] | null>(null);
  const [note, setNote] = useState("");
  const find = async () => {
    setBusy(true);
    const found = await actions.findTelegramChat(token.trim(), code);
    setBusy(false);
    if (!found) return;
    setChats(found);
    if (found.length === 1) {
      onFound(found[0].id);
      setNote(`Нашёл: ${found[0].name || "ваш чат"} (${found[0].id}). Он подставлен в поле, осталось сохранить.`);
    } else setNote(found.length ? "Код прислали из нескольких чатов — выберите свой:" : `Код пока не пришёл: отправьте боту ${code} и нажмите ещё раз.`);
  };
  return (
    <div className="chatfind">
      <p className="kf-h">
        1. Вставьте токен бота (@BotFather). 2. Откройте своего бота в Telegram и отправьте ему код <b className="num">{code}</b>. 3. Нажмите кнопку — бот
        привяжется к чату, приславшему этот код.
      </p>
      <button className="btn" type="button" disabled={busy || (!token.trim() && !hasToken)} onClick={() => void find()}>
        {busy ? (
          <>
            <span className="spin" />
            Ищу…
          </>
        ) : (
          "Найти мой chat id"
        )}
      </button>
      {note && <p className="note-soon" aria-live="polite">{note}</p>}
      {chats && chats.length > 1 && (
        <div className="chatpick">
          {chats.map((c) => (
            <button key={c.id} type="button" className="chip-btn" onClick={() => (onFound(c.id), setNote(`Выбран ${c.name || c.id}. Осталось сохранить.`))}>
              {c.name || c.id}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

const TG_STATE: Record<TgStatus["state"], [string, string]> = {
  ok: ["ok", "читает каналы"],
  off: ["", "нет ключей"],
  no_session: ["warn", "нужен вход"],
  revoked: ["bad", "вход отозван"],
  error: ["warn", "нет связи"],
  stopped: ["", "остановлено"],
  unknown: ["", "бот молчит"],
};
const TG_KEYS = ["TELEGRAM_SENTIMENT_API_ID", "TELEGRAM_SENTIMENT_API_HASH", "TELEGRAM_SENTIMENT_PHONE"];

/** Вход в Telegram кодом из приложения: сессия хранится на сервере зашифрованной, бот код не просит никогда. */
function TgLogin({ item }: { item: SecretItem }) {
  const [step, setStep] = useState<"idle" | "code" | "password">("idle");
  const [via, setVia] = useState<"app" | "sms">("app");
  const [code, setCode] = useState("");
  const [pw, setPw] = useState("");
  const [busy, setBusy] = useState(false);
  const [rt, setRt] = useState<TgStatus | null>(null);
  const ready = TG_KEYS.every((env) => item.fields.find((f) => f.env === env)?.set);
  const phone = item.fields.find((f) => f.env === "TELEGRAM_SENTIMENT_PHONE")?.value;
  useEffect(() => void actions.tgStatus().then(setRt), [item.signed_in, item.checked_at]);
  const run = async (path: "login" | "code" | "password", body?: Record<string, string>) => {
    setBusy(true);
    const r = await actions.tgLogin(path, body);
    setBusy(false);
    if (!r) return;
    if (r.step === "code") return setVia(r.via), setStep("code");
    if (r.step === "password") return setStep("password");
    setStep("idle");
    setCode("");
    setPw("");
  };
  const [c, t] = TG_STATE[rt?.state ?? "unknown"];
  return (
    <div className="chatfind tglogin">
      <div className="tg-h">
        <b>Вход в Telegram: {item.signed_in ? "выполнен" : "не выполнен"}</b>
        {rt && (
          <span className={`pill ${c}`}>
            <span className={`dot ${c}`} />
            {t}
          </span>
        )}
      </div>
      {rt && ready && !["unknown", "no_session"].includes(rt.state) && <p className="kf-h">{rt.message}</p>}
      {rt?.state === "ok" && <TgWindow rt={rt} />}
      {step === "idle" && (
        <>
          <p className="kf-h">
            {!ready
              ? "Сначала сохраните API ID, API Hash и телефон — потом войдите кодом из Telegram."
              : item.signed_in
                ? "Войти заново нужно, только если Telegram отозвал вход или вы сменили аккаунт."
                : "Telegram пришлёт код в приложение Telegram (или по SMS) — введите его здесь. Сессия хранится на сервере зашифрованной, повторно код не понадобится."}
          </p>
          <button className="btn" type="button" disabled={busy || !ready} onClick={() => void run("login")}>
            {busy ? (
              <>
                <span className="spin" />
                Запрашиваю код…
              </>
            ) : item.signed_in ? (
              "Войти заново"
            ) : (
              "Войти"
            )}
          </button>
        </>
      )}
      {step !== "idle" && (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void (step === "code" ? run("code", { code }) : run("password", { password: pw }));
          }}
        >
          {step === "code" ? (
            <div className="kf">
              <label htmlFor="tg-code">
                <span>Код из {via === "app" ? "Telegram (чат «Telegram»)" : "SMS"}</span>
                {phone && <span className="saved">{phone}</span>}
              </label>
              <input className="inp num" id="tg-code" inputMode="numeric" autoComplete="one-time-code" spellCheck={false} value={code} onChange={(e) => setCode(e.target.value)} />
            </div>
          ) : (
            <div className="kf">
              <label htmlFor="tg-pw">Облачный пароль Telegram</label>
              <small className="kf-h">У аккаунта включена двухэтапная проверка. Пароль нужен один раз для входа и не сохраняется.</small>
              <input className="inp" id="tg-pw" type="password" autoComplete="off" value={pw} onChange={(e) => setPw(e.target.value)} />
            </div>
          )}
          <div className="kc-a">
            <button className="btn primary" type="submit" disabled={busy || (step === "code" ? code.trim().length < 4 : !pw)}>
              {busy ? (
                <>
                  <span className="spin" />
                  Проверяю…
                </>
              ) : (
                "Подтвердить"
              )}
            </button>
            <button className="btn" type="button" disabled={busy} onClick={() => setStep("idle")}>
              Отмена
            </button>
          </div>
        </form>
      )}
    </div>
  );
}

/** Что видно в окне свежих постов: «нет упоминаний» отдельно от нейтрального настроения. */
function TgWindow({ rt }: { rt: TgStatus }) {
  const top = Object.entries(rt.mentions ?? {});
  const failed = Object.entries(rt.failed ?? {});
  const posts = rt.posts ?? 0;
  const chans = rt.channels?.length ?? 0;
  return (
    <>
      <p className="kf-h">
        За {rt.window_min ?? 60} мин: {posts} {plural(posts, ["пост", "поста", "постов"])} из {chans} {plural(chans, ["канала", "каналов", "каналов"])} ·{" "}
        {top.length ? `упоминания: ${top.map(([c, n]) => `${c} ${n}`).join(" · ")}` : "монеты текущего скана не упоминались"}
      </p>
      {failed.length > 0 && <p className="kf-h warn">Не читаются: {failed.map(([c, why]) => `${c} — ${why}`).join("; ")}</p>}
    </>
  );
}

export function KeyCard({ item }: { item: SecretItem }) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [model, setModel] = useState(item.model?.value ?? "");
  const [shown, setShown] = useState<Record<string, boolean>>({});
  const [msg, setMsg] = useState<[string, string] | null>(null);
  const [busy, setBusy] = useState(false);
  const [delArmed, armDel] = useArm();
  const [c, t] = STATUS[item.status];
  const [desc, url] = KEY_INFO[item.service] ?? [""];
  const has = item.fields.some((f) => f.set);
  useEffect(() => setModel(item.model?.value ?? ""), [item.model?.value]);

  const save = async () => {
    const filled = Object.fromEntries(Object.entries(values).filter(([, v]) => v.trim()).map(([k, v]) => [k, v.trim()]));
    const modelChanged = Boolean(item.model && model.trim() && model.trim() !== (item.model.value ?? ""));
    if (!Object.keys(filled).length && !modelChanged)
      return setMsg([has ? "" : "bad", has ? "Ничего не изменилось. Чтобы заменить ключ, вставьте новый." : "Вставьте ключ, затем нажмите «Проверить и сохранить»."]);
    for (const f of item.fields) {
      const e = filled[f.env] ? checkField(f.env, f.label, f.secret, filled[f.env]) : "";
      if (e) return setMsg(["bad", e]);
    }
    setBusy(true);
    setMsg(null);
    const ok = await actions.putSecret(item.service, filled, modelChanged ? model.trim() : undefined);
    setBusy(false);
    if (ok) {
      setValues({});
      setShown({});
    }
  };

  return (
    <article className="kcard" id={`key-${item.service}`}>
      <div className="kc-h">
        <b>{item.name}</b>
        <span className={`pill ${c}`}>
          <span className={`dot ${c}`} />
          {t}
        </span>
      </div>
      <p className="kc-d">
        {desc}{" "}
        {url && (
          <a href={url} target="_blank" rel="noopener noreferrer">
            Где взять ↗
          </a>
        )}
      </p>
      {item.fields.filter((f) => !(item.service === "twitter" && f.env === "TWSCRAPE_ACCOUNTS")).map((f) => {
        const id = `k-${f.env}`;
        const tail = f.set ? (f.secret ? (f.last4 ? `сохранён ••••${f.last4}` : "задан") : (f.value ?? "задан")) : f.optional ? "необязательно" : "не задан";
        return (
          <div className="kf" key={f.env}>
            <label htmlFor={id}>
              <span>{f.label}</span>
              <span className="saved">{tail}</span>
            </label>
            {f.hint && <small className="kf-h">{f.hint}</small>}
            {f.secret ? (
              <div className="pw">
                <input
                  className="inp"
                  id={id}
                  type={shown[f.env] ? "text" : "password"}
                  autoComplete="off"
                  spellCheck={false}
                  placeholder={f.set ? "Вставьте новый, чтобы заменить" : (f.placeholder ?? "Вставьте ключ")}
                  value={values[f.env] ?? ""}
                  onChange={(e) => setValues({ ...values, [f.env]: e.target.value })}
                />
                <button type="button" aria-pressed={Boolean(shown[f.env])} aria-label={`Показать: ${f.label}`} onClick={() => setShown({ ...shown, [f.env]: !shown[f.env] })}>
                  {Icon.eye}
                </button>
              </div>
            ) : (
              <input
                className="inp"
                id={id}
                autoComplete="off"
                spellCheck={false}
                placeholder={f.value ?? ""}
                value={values[f.env] ?? ""}
                onChange={(e) => setValues({ ...values, [f.env]: e.target.value })}
              />
            )}
          </div>
        );
      })}
      {item.service === "twitter" && <XAccounts item={item} />}
      {item.service === "telethon" && <TgLogin item={item} />}
      {item.model && (
        <div className="kf">
          <label htmlFor={`k-${item.model.env}`}>
            <span>Модель</span>
            <span className="saved">{item.model.env}</span>
          </label>
          <input className="inp" id={`k-${item.model.env}`} spellCheck={false} placeholder="по умолчанию" value={model} onChange={(e) => setModel(e.target.value)} />
        </div>
      )}
      {item.service === "bybit" && <TestnetRow />}
      {item.service === "telegram" && (
        <ChatFinder
          token={values.TELEGRAM_BOT_TOKEN ?? ""}
          hasToken={item.fields.some((f) => f.env === "TELEGRAM_BOT_TOKEN" && f.set)}
          onFound={(id) => setValues((v) => ({ ...v, TELEGRAM_CHAT_ID: id }))}
        />
      )}
      <div className="kc-a">
        <button className="btn primary" type="button" disabled={busy} onClick={() => void save()}>
          {busy ? (
            <>
              <span className="spin" />
              Проверяю…
            </>
          ) : (
            "Проверить и сохранить"
          )}
        </button>
        {has && (
          <button
            className={`btn danger${delArmed ? " armed" : ""}`}
            type="button"
            disabled={busy}
            onClick={() => armDel() && void actions.deleteSecret(item.service)}
          >
            {delArmed ? "Точно удалить?" : "Удалить"}
          </button>
        )}
      </div>
      <p className={`kc-msg ${msg ? msg[0] : c}`} aria-live="polite">
        {msg ? msg[1] : item.message}
      </p>
    </article>
  );
}

// ══ Все параметры ════════════════════════════════════════════════════════════

const NUMERIC = ["int", "float", "pct"];
export const shown = (d: SettingDef, v: unknown) => (d.type === "pct" ? +(Number(v) * 100).toFixed(4) : v);
const unitOf = (d: SettingDef) => (d.type === "pct" ? "%" : d.unit);

export function fmtVal(d: SettingDef, v: unknown): string {
  if (d.type === "bool") return v ? "вкл" : "выкл";
  if (d.type === "choice") return d.options.find((o) => o.value === v)?.label ?? String(v);
  if (d.type === "str") return v ? String(v) : "пусто";
  const n = Number(shown(d, v));
  return `${Math.abs(n) >= 10000 ? fmtN(n, 0) : n}${unitOf(d) ? ` ${unitOf(d)}` : ""}`;
}

const SOURCE: Record<string, string> = {
  live: "задано в приложении",
  ui: "задано в приложении",
  env: "из файла .env",
  default: "",
};

function AllParams({ go }: { go: (pane: Pane) => void }) {
  const schema = useStore((s) => s.schema);
  const config = useStore((s) => s.config);
  const jump = useStore((s) => s.paramSearch);
  const [q, setQ] = useState(jump ?? "");
  useEffect(() => {
    if (jump === null) return;
    setQ(jump);
    useStore.getState().clearParamSearch();
    setTimeout(() => document.getElementById("param-search")?.focus(), 60);
  }, [jump]);
  const [pending, setPending] = useState<Record<string, unknown>>({});
  const [bad, setBad] = useState<Record<string, string>>({});
  const [restartKeys, setRestartKeys] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [open, setOpen] = useState<Set<string>>(() => new Set(schema.slice(0, 1).map((d) => d.group)));

  const groups = useMemo(() => {
    const out: [string, SettingDef[]][] = [];
    for (const d of schema) {
      const g = out.find(([name]) => name === d.group);
      if (g) g[1].push(d);
      else out.push([d.group, [d]]);
    }
    return out;
  }, [schema]);
  const needle = q.trim().toLowerCase();
  // Поиск ищет по всем параметрам; без поиска — основные или все.
  const hit = (d: SettingDef) =>
    needle ? `${d.label} ${d.key} ${d.hint}`.toLowerCase().includes(needle) : !(d.key in OWN_HOME) && (advanced || d.basic || d.key in pending);
  const found = schema.filter(hit).length;
  const elsewhere = schema.filter((d) => d.key in OWN_HOME).length;

  const change = (d: SettingDef, raw: unknown) => {
    const cur = config[d.key]?.value;
    let v = raw;
    let err = "";
    if (NUMERIC.includes(d.type)) {
      const x = parseFloat(String(raw).replace(",", "."));
      const u = unitOf(d) ? ` ${unitOf(d)}` : "";
      const min = d.min === null ? null : Number(shown(d, d.min));
      const max = d.max === null ? null : Number(shown(d, d.max));
      if (!Number.isFinite(x)) err = "Введите число.";
      else if (min !== null && x < min) err = `Не меньше ${min}${u}.`;
      else if (max !== null && x > max) err = `Не больше ${max}${u}.`;
      else if (d.type === "int" && !Number.isInteger(x)) err = "Нужно целое число.";
      v = d.type === "pct" ? +(x / 100).toFixed(6) : x;
    }
    const nextBad = { ...bad };
    const nextPending = { ...pending };
    if (err) {
      nextBad[d.key] = err;
      delete nextPending[d.key];
    } else {
      delete nextBad[d.key];
      const same = typeof v === "number" && typeof cur === "number" ? Math.abs(v - cur) < 1e-9 : v === cur;
      if (same) delete nextPending[d.key];
      else nextPending[d.key] = v;
    }
    setBad(nextBad);
    setPending(nextPending);
  };

  const [fx, setFx] = useState<Record<string, "ok" | "bad">>({});
  const flashKeys = (keys: string[], kind: "ok" | "bad") => {
    setFx(Object.fromEntries(keys.map((k) => [k, kind])));
    setTimeout(() => setFx({}), 1500);
  };
  const apply = async () => {
    setBusy(true);
    const keys = Object.keys(pending);
    const r: PatchResult | null = await actions.patchConfig(pending);
    setBusy(false);
    if (!r) return flashKeys(keys, "bad");
    flashKeys(keys, "ok");
    setPending({});
    setRestartKeys((k) => [...new Set([...k, ...r.pending_restart])]);
  };
  const restart = async () => {
    if (await actions.restart()) setRestartKeys([]);
  };

  const nPending = Object.keys(pending).length;
  const nBad = Object.keys(bad).length;

  return (
    <div className="set-pane view enter">
      <div className="cfg-top">
        <input id="param-search" className="inp" type="search" placeholder="Поиск: плечо, стоп, RISK_PER_TRADE…  (Ctrl/⌘ K)" aria-label="Поиск параметра" autoComplete="off" value={q} onChange={(e) => setQ(e.target.value)} />
        <span className="label">{needle ? `найдено ${found}` : `${found} из ${schema.length}`}</span>
      </div>
      <div className="row-tgl">
        <span>
          Показать расширенные <Q tip="По умолчанию видны основные параметры. Расширенные — тонкая настройка фильтров, выхода, обучения и сервера." />
        </span>
        <Toggle on={advanced} label="Показать расширенные параметры" onChange={setAdvanced} />
      </div>
      <p className="cfg-legend">
        <span className="tag hot">сразу</span> применяется без перезапуска · <span className="tag rst">перезапуск</span> вступит в силу после
        перезапуска бота. Ключи — в «Подключения», ключи AI — в Бот → AI-цепочка, Paper/Live — в «Основное».
      </p>
      {!needle && elsewhere > 0 && (
        <p className="cfg-legend">
          {elsewhere} {plural(elsewhere, ["настройка", "настройки", "настроек"])} здесь не показано: у них есть свои разделы (риск, открытие сделок, режим, часы, AI,
          обучение). Найти любую и перейти к ней можно поиском.
        </p>
      )}
      <div className="cfg">
        {groups.map(([g, defs]) => {
          const rows = defs.filter(hit);
          if (!rows.length) return null;
          const changed = rows.filter((d) => d.key in pending).length;
          return (
            <details
              className="grp"
              key={g}
              open={Boolean(needle) || open.has(g)}
              onToggle={(e) => {
                if (needle) return;
                const next = new Set(open);
                if ((e.currentTarget as HTMLDetailsElement).open) next.add(g);
                else next.delete(g);
                setOpen(next);
              }}
            >
              <summary>
                <span>{g}</span>
                <span className="cnt">{rows.length}</span>
                {changed > 0 && <span className="chgb">изменено {changed}</span>}
              </summary>
              {rows.map((d) =>
                OWN_HOME[d.key] ? (
                  <HomeRow key={d.key} d={d} go={go} />
                ) : (
                <ParamRow key={d.key} d={d} value={d.key in pending ? pending[d.key] : config[d.key]?.value} source={config[d.key]?.source ?? "default"} changed={d.key in pending} err={bad[d.key]} fx={fx[d.key]} onChange={(v) => change(d, v)} onReset={() => change(d, rawOf(d, d.default))} />
                ),
              )}
            </details>
          );
        })}
      </div>
      <SettingsTools schema={schema} config={config} onImport={(v) => Object.entries(v).forEach(([k, x]) => {
        const d = schema.find((s) => s.key === k);
        if (d && !d.readonly) change(d, rawOf(d, x));
      })} />
      {(nPending > 0 || nBad > 0) && (
        <div className="applybar">
          <span>
            {nPending > 0 && (
              <>
                Изменено <b>{nPending}</b> {plural(nPending, ["параметр", "параметра", "параметров"])}
              </>
            )}
            {nBad > 0 && <span className="down">{`${nPending ? " · " : ""}исправьте ошибки: ${nBad}`}</span>}
          </span>
          <span className="sp">
            <button
              className="btn"
              type="button"
              onClick={() => {
                setPending({});
                setBad({});
              }}
            >
              Отменить
            </button>
            <button className="btn primary" type="button" disabled={busy || nBad > 0 || !nPending} onClick={() => void apply()}>
              Применить
            </button>
          </span>
        </div>
      )}
      {!nPending && !nBad && restartKeys.length > 0 && (
        <div className="applybar">
          <span>
            <b>{restartKeys.length}</b> {plural(restartKeys.length, ["настройка вступит", "настройки вступят", "настроек вступят"])} в силу после
            перезапуска бота
          </span>
          <span className="sp">
            <button className="btn primary" type="button" onClick={() => void restart()}>
              Перезапустить бота
            </button>
          </span>
        </div>
      )}
    </div>
  );
}

/** Параметр с собственным разделом: вместо второго поля — ссылка туда. */
function HomeRow({ d, go }: { d: SettingDef; go: (pane: Pane) => void }) {
  const home = OWN_HOME[d.key];
  const setTab = useStore((s) => s.setTab);
  const openNode = useStore((s) => s.openNode);
  const open = () => {
    if (home.tab === "settings" && home.pane) go(home.pane);
    else if (home.node) openNode(home.node);
    else setTab(home.tab);
  };
  return (
    <div className="prm home">
      <div className="prm-l">
        <label>{d.label}</label>
        <code>{d.key}</code>
        {d.hint && <p>{d.hint}</p>}
      </div>
      <div className="prm-c">
        <span className="tag hot">{home.where}</span>
        <button className="btn" type="button" onClick={open}>
          Открыть
        </button>
      </div>
    </div>
  );
}

/** Значение в том виде, в каком поле его показывает (проценты — в %, а не долей). */
const rawOf = (d: SettingDef, v: unknown): unknown => (d.type === "pct" && typeof v === "number" ? +(v * 100).toFixed(6) : v);

const sameVal = (a: unknown, b: unknown): boolean =>
  typeof a === "number" && typeof b === "number" ? Math.abs(a - b) < 1e-9 : a === b;

function ParamRow(props: { d: SettingDef; value: unknown; source: string; changed: boolean; err?: string; fx?: "ok" | "bad"; onChange: (v: unknown) => void; onReset: () => void }) {
  const { d, value } = props;
  const id = `c-${d.key}`;
  let control: JSX.Element;
  if (d.type === "bool") control = <Toggle on={Boolean(value)} label={d.label} disabled={d.readonly} onChange={props.onChange} />;
  else if (d.type === "choice")
    control = (
      <select className="sel" id={id} value={String(value ?? "")} disabled={d.readonly} onChange={(e) => props.onChange(e.target.value)}>
        {d.options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    );
  else if (d.type === "str")
    control = <input className="inp" id={id} value={String(value ?? "")} spellCheck={false} autoComplete="off" disabled={d.readonly} onChange={(e) => props.onChange(e.target.value)} />;
  else
    control = (
      <NumField id={id} d={d} value={Number(shown(d, value))} disabled={d.readonly} onChange={props.onChange} />
    );
  return (
    <div className={`prm${props.changed ? " chg" : ""}${props.err ? " bad" : ""}${props.fx ? ` fx-${props.fx}` : ""}`}>
      <div className="prm-l">
        <label htmlFor={id}>{d.label}</label>
        <code>{d.key}</code>
        {d.hint && <p>{d.hint}</p>}
        <p className="def">
          по умолчанию: {fmtVal(d, d.default)}
          {SOURCE[props.source] ? ` · ${SOURCE[props.source]}` : ""}
        </p>
      </div>
      <div className="prm-c">
        {d.readonly ? <span className="tag rst">только .env</span> : <span className={`tag ${d.hot ? "hot" : "rst"}`}>{d.hot ? "сразу" : "перезапуск"}</span>}
        {control}
        {props.fx === "ok" && (
          <svg className="prm-tick" viewBox="0 0 24 24" width="18" height="18" aria-label="Применено" role="img">
            <path d="M5 12.5l4.5 4.5L19 7.5" />
          </svg>
        )}
        {!d.readonly && !sameVal(value, d.default) && (
          <button className="btn reset" type="button" title={`Вернуть по умолчанию: ${fmtVal(d, d.default)}`} aria-label={`${d.label}: вернуть по умолчанию`} onClick={props.onReset}>
            ↺
          </button>
        )}
      </div>
      {props.err && <p className="err">{props.err}</p>}
    </div>
  );
}

/** Число редактируется как текст: промежуточное «0.» или пустое поле не сбрасывается. */
export function NumField({ id, d, value, disabled, onChange }: { id: string; d: SettingDef; value: number; disabled: boolean; onChange: (v: string) => void }) {
  const [text, setText] = useState(String(value));
  useEffect(() => {
    if (parseFloat(text.replace(",", ".")) !== value) setText(String(value));
  }, [value]); // eslint-disable-line react-hooks/exhaustive-deps
  const step = d.type === "pct" ? (d.step ?? 0.0001) * 100 : (d.step ?? (d.type === "int" ? 1 : 0.01));
  return (
    <div className="numf">
      <input
        id={id}
        type="number"
        inputMode="decimal"
        step={step}
        value={text}
        disabled={disabled}
        onChange={(e) => {
          setText(e.target.value);
          onChange(e.target.value);
        }}
      />
      <span className="u">{unitOf(d)}</span>
    </div>
  );
}

// ── Журнал изменений, копирование и вставка набора настроек ─────────────────────

const when = (ts: number): string =>
  new Date(ts * 1000).toLocaleString("ru-RU", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

function SettingsTools({ schema, config, onImport }: { schema: SettingDef[]; config: ConfigValues; onImport: (v: Record<string, unknown>) => void }) {
  const say = useStore((s) => s.say);
  const [hist, setHist] = useState<ConfigHistoryItem[] | null>(null);
  const [paste, setPaste] = useState<string | null>(null);
  const [err, setErr] = useState("");
  const [resetAsk, setResetAsk] = useState(false);
  const val = (d: SettingDef, v: unknown): string => (v === null || v === undefined ? "—" : fmtVal(d, v));

  const loadHistory = () => void api<ConfigHistoryItem[]>("GET", "/api/config/history").then(setHist).catch(() => setHist([]));
  // Набор — только то, что вы поменяли: значения по умолчанию и «только .env» не копируются.
  const mine = () =>
    Object.fromEntries(schema.filter((d) => !d.readonly && config[d.key] && !sameVal(config[d.key].value, d.default)).map((d) => [d.key, config[d.key].value]));
  const copy = async () => {
    const text = JSON.stringify(mine(), null, 2);
    try {
      await navigator.clipboard.writeText(text);
      say(`Скопировано настроек: ${Object.keys(mine()).length}`);
    } catch {
      setPaste(text);
      say("Не удалось скопировать — набор показан ниже, скопируйте вручную.");
    }
  };
  const doImport = () => {
    try {
      const v = JSON.parse(paste ?? "");
      if (!v || typeof v !== "object" || Array.isArray(v)) throw new Error();
      const known = Object.keys(v).filter((k) => schema.some((d) => d.key === k && !d.readonly));
      onImport(Object.fromEntries(known.map((k) => [k, v[k]])));
      setErr("");
      setPaste(null);
      say(`Загружено в форму: ${known.length}. Проверьте и нажмите «Применить».`);
    } catch {
      setErr("Это не набор настроек: вставьте JSON, скопированный из приложения.");
    }
  };

  return (
    <article className="card">
      <div className="card-h">
        <h2>Набор настроек и журнал</h2>
      </div>
      <div className="btn-row">
        <button className="btn" type="button" onClick={() => void copy()}>
          Скопировать мои настройки
        </button>
        <button className="btn" type="button" onClick={() => setPaste(paste === null ? "" : null)}>
          Вставить набор…
        </button>
        <button className="btn" type="button" aria-expanded={hist !== null} onClick={() => (hist ? setHist(null) : loadHistory())}>
          {hist ? "Скрыть журнал" : "Что менялось"}
        </button>
        <button className="btn danger" type="button" onClick={() => setResetAsk(true)}>
          Сбросить всё к .env
        </button>
      </div>
      {resetAsk && (
        <div className="confirm-box">
          <span>
            Все настройки, которые вы меняли из приложения и Telegram, вернутся к значениям из <code>.env</code>: режим, пауза, символы, риск, стратегии.
            Экспериментальные стратегии снова выключатся. Ключи API не затрагиваются.
          </span>
          <div className="btn-row">
            <button className="btn" type="button" onClick={() => setResetAsk(false)}>
              Отмена
            </button>
            <button
              className="btn danger"
              type="button"
              onClick={() => {
                setResetAsk(false);
                void actions.resetConfig();
              }}
            >
              Да, сбросить
            </button>
          </div>
        </div>
      )}
      {paste !== null && (
        <div className="paste">
          <textarea className="inp" rows={6} spellCheck={false} aria-label="Набор настроек (JSON)" value={paste} onChange={(e) => setPaste(e.target.value)} />
          {err && <p className="err">{err}</p>}
          <button className="btn primary" type="button" disabled={!paste.trim()} onClick={doImport}>
            Загрузить в форму
          </button>
        </div>
      )}
      {hist && (
        <ul className="hist">
          {hist.length === 0 && <li className="empty">Из приложения пока ничего не меняли.</li>}
          {hist.map((h) => (
            <li key={h.ts}>
              <time>{when(h.ts)}</time>
              <div>
                {h.changes.map((c) => {
                  const d = schema.find((s) => s.key === c.key);
                  return (
                    <p key={c.key}>
                      <b>{c.label}</b>: {d ? val(d, c.old) : String(c.old)} → {d ? val(d, c.new) : String(c.new)}
                    </p>
                  );
                })}
              </div>
            </li>
          ))}
        </ul>
      )}
    </article>
  );
}

// ── X / Twitter: бесплатный обход официального API ────────────────────────────

const X_FIELDS: [keyof XForm, string, boolean][] = [
  ["login", "Логин в X", false],
  ["password", "Пароль от X", true],
  ["email", "Почта аккаунта", false],
  ["email_password", "Пароль от почты", true],
];
type XForm = { login: string; password: string; email: string; email_password: string };
const X_EMPTY: XForm = { login: "", password: "", email: "", email_password: "" };

/** Аккаунты отдельными полями: каждое значение хранится зашифрованным, паролей приложение не показывает. */
function XAccounts({ item }: { item: SecretItem }) {
  const [form, setForm] = useState<XForm>(X_EMPTY);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [delArmed, setDelArmed] = useState<string | null>(null);
  const official = item.fields.find((f) => f.env === "TWITTER_BEARER_TOKEN")?.set;
  const accounts = item.accounts ?? [];
  const missing = X_FIELDS.filter(([k]) => !form[k].trim()).map(([, l]) => l);
  const add = async () => {
    if (missing.length) return setErr(`Заполните: ${missing.join(", ").toLowerCase()}.`);
    if (!form.email.includes("@")) return setErr("«Почта аккаунта» — это адрес вида имя@почта.ru.");
    setErr("");
    setBusy(true);
    const ok = await actions.addXAccount(form);
    setBusy(false);
    if (ok) setForm(X_EMPTY);
  };
  return (
    <details className="risk-box" open={accounts.length > 0}>
      <summary>
        Без официального API — бесплатно, на ваш страх и риск
        {accounts.length > 0 && <span className="pill ok">аккаунтов: {accounts.length}</span>}
      </summary>
      <div className="honest">
        <div>
          <b>Это обход правил X.</b> Бот заходит в X под обычными аккаунтами и читает твиты как человек. X может заблокировать такие аккаунты, поэтому заведите
          для этого <b>отдельные</b> аккаунты, а не основной, и не используйте почту, которая вам важна. Пароль от почты нужен только затем, чтобы
          библиотека сама прочитала код подтверждения при входе в X.
          {official ? " Сейчас задан официальный API, и он имеет приоритет: эти аккаунты не используются." : ""}
        </div>
      </div>
      {accounts.length > 0 && (
        <ul className="xacc">
          {accounts.map((a) => (
            <li key={a.login}>
              <span>
                <b>{a.login}</b>
                <small>почта {a.email}</small>
              </span>
              <button
                className={`btn danger${delArmed === a.login ? " armed" : ""}`}
                type="button"
                onClick={() => {
                  if (delArmed === a.login) {
                    setDelArmed(null);
                    void actions.removeXAccount(a.login);
                  } else {
                    setDelArmed(a.login);
                    setTimeout(() => setDelArmed(null), 2600);
                  }
                }}
              >
                {delArmed === a.login ? "Точно?" : "Удалить"}
              </button>
            </li>
          ))}
        </ul>
      )}
      <div className="xform">
        {X_FIELDS.map(([k, label, secret]) => (
          <div className="kf" key={k}>
            <label htmlFor={`x-${k}`}>{label}</label>
            <input
              className="inp"
              id={`x-${k}`}
              type={secret ? "password" : k === "email" ? "email" : "text"}
              autoComplete="off"
              spellCheck={false}
              value={form[k]}
              onChange={(e) => setForm({ ...form, [k]: e.target.value })}
            />
          </div>
        ))}
      </div>
      {err && <p className="kc-msg bad">{err}</p>}
      <div className="kc-a">
        <button className="btn primary" type="button" disabled={busy} onClick={() => void add()}>
          {busy ? "Сохраняю…" : "Добавить аккаунт"}
        </button>
      </div>
    </details>
  );
}
