// Вкладка «Настройки»: основное, AI-провайдеры, подключения (ключи), все параметры.

import { useEffect, useMemo, useState, type FormEvent } from "react";

import { fmtN, plural } from "../lib/format";
import type { SecretItem, SettingDef } from "../lib/types";
import { actions, useStore, type PatchResult } from "../store";
import { Icon, Q, Seg, Toggle, useArm } from "../ui";
import { AiPane } from "./ai";
import { cfg } from "./bots";

type Pane = "main" | "ai" | "conn" | "all";

export function Settings() {
  const [pane, setPane] = useState<Pane>("main");
  return (
    <section className="view enter" aria-label="Настройки">
      <Seg
        className="full"
        value={pane}
        options={[
          ["main", "Основное"],
          ["ai", "AI"],
          ["conn", "Подключения"],
          ["all", "Параметры"],
        ]}
        onChange={setPane}
        label="Раздел настроек"
      />
      {pane === "main" && <Main go={setPane} />}
      {pane === "ai" && <AiPane />}
      {pane === "conn" && <Keys />}
      {pane === "all" && <AllParams />}
    </section>
  );
}

// ══ Основное ═════════════════════════════════════════════════════════════════

function Main({ go }: { go: (pane: Pane) => void }) {
  return (
    <div className="set-pane view enter">
      <div className="cols">
        <div className="col">
          <HoursCard />
          <SymbolsCard />
          <AlertsCard />
        </div>
        <div className="col">
          <ConnCard go={go} />
          <AccountCard toKeys={() => go("conn")} />
          <HelpCard />
        </div>
      </div>
    </div>
  );
}

const HOURS = [
  ["", "24/7"],
  ["1-10", "Азия"],
  ["8-20", "Европа"],
  ["14-22", "США"],
  ["22-6", "Ночь"],
] as const;

function HoursCard() {
  const hours = useStore((s) => cfg<string>(s.config, "TRADING_HOURS", ""));
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 60_000);
    return () => clearInterval(t);
  }, []);
  const [a, b] = hours ? hours.split("-").map(Number) : [0, 24];
  const on = (i: number) => !hours || (a < b ? i >= a && i < b : i >= a || i < b);
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

function SymbolsCard() {
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

function AlertsCard() {
  const alerts = useStore((s) => s.alerts);
  return (
    <article className="card">
      <div className="card-h">
        <h2>Уведомления в Telegram</h2>
        <Q tip="Те же виды, что в меню /alerts. Пауза не даёт одному и тому же уведомлению сыпаться чаще, чем раз в N минут." />
      </div>
      <div className="list-tgl">
        {alerts.map((a) => (
          <div className={`alert-row${a.enabled ? "" : " off"}`} key={a.kind}>
            <div>
              <b>{a.label}</b>
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

function ConnCard({ go }: { go: (pane: Pane) => void }) {
  const secrets = useStore((s) => s.secrets);
  const ai = useStore((s) => s.ai);
  const lastAi = ai?.providers.find((p) => p.id === ai.last);
  const note = useStore((s) => s.secretsNote);
  const wsUp = useStore((s) => s.wsUp);
  const alive = useStore((s) => s.alive);
  const by = (id: string) => secrets?.find((s) => s.service === id);
  const okIn = (group: string) => {
    const g = (secrets ?? []).filter((s) => s.group === group);
    return [g.filter((s) => s.status === "ok").length, g.length];
  };
  const row = (label: string, item?: SecretItem) => {
    const [c, t] = item ? STATUS[item.status] : ["", "—"];
    const last4 = item?.fields.find((f) => f.secret && f.last4)?.last4;
    return (
      <div key={label}>
        <span className={`dot ${c}`} />
        {label}
        <small>
          {t}
          {last4 ? ` · ключ ••••${last4}` : ""}
        </small>
      </div>
    );
  };
  const [dOk, dN] = okIn("Данные и соцсети");
  return (
    <article className="card">
      <div className="card-h">
        <h2>Подключения</h2>
      </div>
      <div className="conn">
        <div>
          <span className={`dot ${wsUp && alive ? "ok" : "bad"}`} />
          Сервер бота
          <small>{wsUp ? (alive ? "работает" : "бот не отвечает") : "нет связи"}</small>
        </div>
        {secrets ? (
          <>
            {row("Bybit API", by("bybit"))}
            {row("Telegram-бот", by("telegram"))}
            <div>
              <span className={`dot ${ai?.chain.length ? "ok" : "bad"}`} />
              AI-провайдеры
              <small>
                в цепочке {ai?.chain.length ?? 0}
                {lastAi ? ` · отвечает ${lastAi.name}` : ""}
              </small>
            </div>
            <div>
              <span className={`dot ${dOk ? "ok" : ""}`} />
              Источники данных
              <small>
                подключено {dOk} из {dN}
              </small>
            </div>
          </>
        ) : (
          <p className="note-soon">{note || "Статус ключей недоступен."}</p>
        )}
      </div>
      <div className="btn-row" style={{ marginTop: 12 }}>
        <button className="btn" type="button" onClick={() => go("ai")}>
          AI-провайдеры
        </button>
        <button className="btn" type="button" onClick={() => go("conn")}>
          Ключи и доступ
        </button>
      </div>
    </article>
  );
}

function AccountCard({ toKeys }: { toKeys: () => void }) {
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
                <button className="btn primary" type="button" onClick={toKeys}>
                  Открыть ключи
                </button>
                <button className="btn" type="button" onClick={() => setBox(false)}>
                  Отмена
                </button>
              </div>
            </>
          ) : !pinSet ? (
            <>
              <b>Сначала задайте PIN.</b>
              <span>Без PIN переход на реальные деньги невозможен. Задайте его в «Ключи API → Сервер и доступ» — его же спрашивает Telegram-бот.</span>
              <div className="btn-row">
                <button className="btn primary" type="button" onClick={toKeys}>
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

const HELP: [string, string[]][] = [
  [
    "О боте",
    [
      "BitbotBY торгует на бирже Bybit сам. Каждый цикл он сканирует рынок, оставляет самые живые монеты, проверяет их стратегиями и AI, а затем — рисками. Сделку он открывает, только если все проверки пройдены, и сразу ставит стоп-лосс и тейк-профит на бирже.",
      "Пока включён Paper, все сделки виртуальные.",
    ],
  ],
  [
    "Режимы решения",
    [
      "Local — только индикаторы, без внешних сервисов. AI — решает языковая модель. SAC — нейросеть, обученная на истории; нужна обученная модель. Hybrid — SAC и AI должны согласиться, иначе бот пропускает сделку.",
      "Chronos немного повышает или понижает уверенность сигнала по прогнозу цены.",
    ],
  ],
  [
    "Настройки",
    [
      "Частые настройки — во вкладке «Боты». В «Настройках» четыре раздела: «Основное» — часы, монеты, уведомления и счёт; «AI» — какие модели решают и в каком порядке; «Подключения» — ключи биржи, Telegram и данных; «Параметры» — все остальные настройки бота. Метка «сразу» значит, что бот применит изменение без остановки, «перезапуск» — после перезапуска.",
      "Значения из приложения важнее значений в файле .env.",
    ],
  ],
  [
    "Команды Telegram",
    [
      "Всё, что есть в приложении, доступно и в Telegram-боте: /status /pnl /pos /pause /resume /closeall /mode /scan /risk /lev /hours /add /remove /symbols /strategies /provider /alerts /trainn /settings /help.",
    ],
  ],
  [
    "Монеты и профили",
    [
      "Профиль рынка задаёт, что бот сканирует: Bluechip — крупнейшие монеты, Altcoin — ликвидные альткоины, Meme — мем-монеты с сигналами из соцсетей.",
      "«Всегда сканировать» добавляет монеты к скану, «Исключить» убирает их навсегда.",
    ],
  ],
  [
    "Риски и защита",
    [
      "Бот рискует заданным процентом баланса на сделку. Защита: дневной лимит убытка, остановка при предельной просадке, пауза при всплеске волатильности и вокруг макро-новостей, фильтры корреляции и ликвидности. PANIC закрывает всё по рынку.",
      "Бэктест пока не нашёл у стратегий устойчивого преимущества — держите Paper, пока статистика в live этого не подтвердит.",
    ],
  ],
  [
    "Безопасность",
    [
      "Ключи хранятся на сервере в зашифрованном виде, приложение показывает только последние 4 символа. Ключ Bybit создавайте без права вывода средств и с привязкой к IP сервера.",
      "Переход на реальные деньги защищён PIN. Управлять ботом может только ваш Telegram-аккаунт.",
    ],
  ],
];

function HelpCard() {
  return (
    <article className="card help">
      <div className="card-h">
        <h2>Справка</h2>
      </div>
      {HELP.map(([title, paras]) => (
        <details key={title}>
          <summary>{title}</summary>
          <div className="hb">
            {paras.map((p) => (
              <p key={p}>{p}</p>
            ))}
          </div>
        </details>
      ))}
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
  twitter: ["Настроение по твитам, только профиль Meme. Нужен платный тариф API или аккаунты для twscrape.", "https://developer.twitter.com"],
  telethon: ["Чтение публичных каналов через Telethon, только профиль Meme. При первом входе Telegram пришлёт код.", "https://my.telegram.org"],
  pumpportal: ["Темп запуска новых токенов на pump.fun — признак мем-сезона.", "https://pumpportal.fun"],
  server: ["PIN для перехода в Live, доступ к Redis и API дашборда, секрет вебхука алертов."],
};

function Keys() {
  const secrets = useStore((s) => s.secrets);
  const note = useStore((s) => s.secretsNote);
  // Ключи AI-провайдеров — в разделе «AI», здесь биржа, Telegram, данные и доступ.
  const groups = useMemo(() => [...new Set((secrets ?? []).map((s) => s.group))].filter((g) => g !== "AI-модели"), [secrets]);
  return (
    <div className="set-pane view enter">
      <div className="secnote">
        {Icon.lock}
        <div>
          <b>Как хранятся ключи.</b> Ключ уходит на сервер вашего бота и хранится там зашифрованным. Приложение больше не показывает его
          целиком — только последние 4 символа. Перед сохранением сервер проверяет ключ запросом к сервису. Новый ключ начнёт работать после
          перезапуска бота.
        </div>
      </div>
      {!secrets && <p className="empty">{note || "Ключи недоступны."}</p>}
      {groups.map((g) => (
        <section className="kgroup" key={g}>
          <h3>{g}</h3>
          <div className="kgrid">
            {secrets!
              .filter((s) => s.group === g)
              .map((s) => (
                <KeyCard key={s.service} item={s} />
              ))}
          </div>
        </section>
      ))}
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

function KeyCard({ item }: { item: SecretItem }) {
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
    <article className="kcard">
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
      {item.fields.map((f) => {
        const id = `k-${f.env}`;
        const tail = f.set ? (f.secret ? (f.last4 ? `сохранён ••••${f.last4}` : "задан") : (f.value ?? "задан")) : f.optional ? "необязательно" : "не задан";
        return (
          <div className="kf" key={f.env}>
            <label htmlFor={id}>
              <span>{f.label}</span>
              <span className="saved">{tail}</span>
            </label>
            {f.secret ? (
              <div className="pw">
                <input
                  className="inp"
                  id={id}
                  type={shown[f.env] ? "text" : "password"}
                  autoComplete="off"
                  spellCheck={false}
                  placeholder={f.set ? "Вставьте новый, чтобы заменить" : "Вставьте ключ"}
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
      {item.model && (
        <div className="kf">
          <label htmlFor={`k-${item.model.env}`}>
            <span>Модель</span>
            <span className="saved">{item.model.env}</span>
          </label>
          <input className="inp" id={`k-${item.model.env}`} spellCheck={false} placeholder="по умолчанию" value={model} onChange={(e) => setModel(e.target.value)} />
        </div>
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
const shown = (d: SettingDef, v: unknown) => (d.type === "pct" ? +(Number(v) * 100).toFixed(4) : v);
const unitOf = (d: SettingDef) => (d.type === "pct" ? "%" : d.unit);

function fmtVal(d: SettingDef, v: unknown): string {
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

function AllParams() {
  const schema = useStore((s) => s.schema);
  const config = useStore((s) => s.config);
  const [q, setQ] = useState("");
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
    needle ? `${d.label} ${d.key} ${d.hint}`.toLowerCase().includes(needle) : advanced || d.basic || d.key in pending;
  const found = schema.filter(hit).length;

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

  const apply = async () => {
    setBusy(true);
    const r: PatchResult | null = await actions.patchConfig(pending);
    setBusy(false);
    if (!r) return;
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
        <input className="inp" type="search" placeholder="Поиск: плечо, стоп, RISK_PER_TRADE…" aria-label="Поиск параметра" autoComplete="off" value={q} onChange={(e) => setQ(e.target.value)} />
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
        перезапуска бота. Ключи — в «Подключения» и «AI», Paper/Live — в «Основное».
      </p>
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
              {rows.map((d) => (
                <ParamRow key={d.key} d={d} value={d.key in pending ? pending[d.key] : config[d.key]?.value} source={config[d.key]?.source ?? "default"} changed={d.key in pending} err={bad[d.key]} onChange={(v) => change(d, v)} />
              ))}
            </details>
          );
        })}
      </div>
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

function ParamRow(props: { d: SettingDef; value: unknown; source: string; changed: boolean; err?: string; onChange: (v: unknown) => void }) {
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
    <div className={`prm${props.changed ? " chg" : ""}${props.err ? " bad" : ""}`}>
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
      </div>
      {props.err && <p className="err">{props.err}</p>}
    </div>
  );
}

/** Число редактируется как текст: промежуточное «0.» или пустое поле не сбрасывается. */
function NumField({ id, d, value, disabled, onChange }: { id: string; d: SettingDef; value: number; disabled: boolean; onChange: (v: string) => void }) {
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
