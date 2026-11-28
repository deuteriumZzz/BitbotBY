// Настройки → Подключения: ключи сервисов (биржа, AI, данные) и проверка каждого.

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { CalendarDots, CaretRight, Cube, Fire, Key, Newspaper } from "@phosphor-icons/react";

import type { BotNode, SecretItem } from "../../lib/types";
import { actions } from "../../store/actions";
import { useStore } from "../../store";
import { Toggle, useArm } from "../../components/controls";
import { IRow } from "../../components/irow";
import { brandLogo } from "../../components/brand";
import { Icon } from "../../components/icon";
import { Q } from "../../components/tooltip";
import { cfg } from "../bots";

import { ChatFinder, TgLogin } from "./telegram";
import { STATUS } from "./trading";
import { XAccounts, XWay, XWayKind, xWay } from "./x";

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
  telegram: ["Токен от @BotFather и ваш chat_id (его покажет @userinfobot). Управлять ботом может только этот чат.", "https://t.me/BotFather"],
  news: ["Новости для оценки настроения рынка.", "https://newsapi.org"],
  finnhub: ["Календарь FOMC, CPI, NFP для макро-паузы. Без ключа пауза не работает.", "https://finnhub.io"],
  glassnode: ["On-chain метрики, например запас стейблкоинов.", "https://glassnode.com"],
  reddit: ["Посты r/CryptoCurrency для анализа настроения моделью CryptoBERT.", "https://www.reddit.com/prefs/apps"],
  twitter: ["Настроение по твитам, только профиль Meme. Подключается одним из двух способов; если заданы оба, работает официальный API."],
  telethon: ["Посты каналов в реальном времени, во всех профилях: монеты текущего скана ищутся точно ($WIF, #WIF, WIF/USDT). API ID и API Hash — на my.telegram.org, затем «Войти»: код придёт в Telegram.", "https://my.telegram.org"],
  pumpportal: ["Темп запуска новых токенов на pump.fun — признак мем-сезона.", "https://pumpportal.fun"],
  server: ["PIN для перехода в Live, доступ к Redis и API дашборда, секрет вебхука алертов."],
};

/** Что питает каждый ключ: [что именно, шаг цепочки]. Шаг в панели ключа — переход в «Цепочку».
 * Сигналы для решения ведут в AI-цепочку: там карточка «Сигналы для решения» и её узел на холсте. */
export const FEEDS: Record<string, [string, BotNode?][]> = {
  bybit: [["цены и свечи", "scan"], ["ордера и баланс", "order"]],
  telegram: [["уведомления"], ["управление из чата"]],
  news: [["новости и заголовки", "ai"]],
  finnhub: [["макро-пауза: FOMC, CPI, NFP", "risk"], ["фильтр рынка · макро", "ai"]],
  glassnode: [["фильтр рынка · стейблкоины", "ai"]],
  reddit: [["фильтр рынка · CryptoBERT", "ai"]],
  twitter: [["настроение · Meme", "ai"]],
  telethon: [["настроение и упоминания монет", "ai"], ["доверенные — сделка по посту", "order"]],
  pumpportal: [["детектор мем-сезона", "scan"]],
  server: [["переход в Live", "order"], ["дашборд и вебхуки"]],
};

export const STEP: Partial<Record<BotNode, string>> = { scan: "Сканер", ai: "AI", risk: "Риск", order: "Ордер" };

/** Ключи как связки: сервис → что он питает в боте. Форма ключа открывается панелью справа, список остаётся на виду. */
/** Эти ключи открываются с карты подключений — в списке их не повторяем. */
// Ключи, у которых свой дом строкой KeyLine: Bybit — «Ордер», Telegram-бот — уведомления, вход в Telegram — каналы.
const ON_MAP = new Set(["bybit", "telegram", "telethon"]);

/** only — какие группы показать списком (по умолчанию все); панель ключа открывается для любого. */
export function Keys({ open, setOpen, only }: { open: string | null; setOpen: (service: string | null) => void; only?: string[] }) {
  const secrets = useStore((s) => s.secrets);
  const note = useStore((s) => s.secretsNote);
  const closeBtn = useRef<HTMLButtonElement>(null);
  // Ключи AI-провайдеров настраиваются в узле AI цепочки — здесь одна строка-связка.
  const groups = useMemo(
    () => [...new Set((secrets ?? []).map((s) => s.group))].filter((g) => g !== "AI-модели" && (!only || only.includes(g))),
    [secrets, only],
  );
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
  // Строка источника: иконка, название, состояние; что он питает в боте — внутри, по клику.
  const row = (id: string, c: string, name: string, sub: string, act: () => void) => (
    <li key={id}>
      <button type="button" className={`kl-row irow ${c === "ok" ? "ok" : c}`} onClick={act}>
        <span className="irow-ico" aria-hidden="true">
          {keyIcon(id)}
        </span>
        <span className="irow-t">
          <b>{name}</b>
          <small>{sub}</small>
        </span>
        <span className="irow-c kl-go" aria-hidden="true">
          <CaretRight />
        </span>
      </button>
    </li>
  );
  const keyRow = (s: SecretItem) => {
    const [c, t] = STATUS[s.status];
    const last4 = s.fields.find((f) => f.secret && f.last4)?.last4;
    const way = s.service === "twitter" ? xWay(s) : "";
    return row(s.service, c, s.name, `${t}${way ? ` · ${way}` : ""}${last4 ? ` · ••••${last4}` : ""}`, () => setOpen(s.service));
  };
  return (
    <div className={only?.length === 0 ? "" : "set-pane view enter"}>
      {!secrets && only?.length !== 0 && <p className="empty">{note || "Ключи недоступны."}</p>}
      <div className="klist">
        {/* Bybit, Telegram и AI — на карте выше, их ключи открываются оттуда. */}
        {groups.map((g) => {
          const items = secrets!.filter((s) => s.group === g && !ON_MAP.has(s.service));
          return items.length ? (
            <section className="kl-g" key={g}>
              <h3>{g}</h3>
              <ul>{items.map(keyRow)}</ul>
            </section>
          ) : null;
        })}
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
                        <span className="kl-chip" key={what}>
                          <em>{STEP[step]}</em>
                          {what}
                        </span>
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
  // У X два способа подключения: открыт тот, которым бот читает X сейчас.
  const [way, setWay] = useState<XWayKind>(() => (xWay(item) === "официальный API" ? "api" : "acc"));
  const xAcc = item.service === "twitter" && way === "acc";
  useEffect(() => setModel(item.model?.value ?? ""), [item.model?.value]);

  const save = async () => {
    const filled = Object.fromEntries(Object.entries(values).filter(([, v]) => v.trim()).map(([k, v]) => [k, v.trim()]));
    const modelChanged = Boolean(item.model && model.trim() && model.trim() !== (item.model.value ?? ""));
    if (!Object.keys(filled).length && !modelChanged) {
      if (!has) return setMsg(["bad", "Вставьте ключ, затем нажмите «Проверить и сохранить»."]);
      // Полей не меняли — проверяем то, что уже сохранено.
      setBusy(true);
      setMsg(null);
      await actions.checkSecret(item.service);
      return setBusy(false);
    }
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
      {item.service === "twitter" && <XWay item={item} way={way} setWay={setWay} />}
      {!xAcc && item.fields.filter((f) => !(item.service === "twitter" && f.env === "TWSCRAPE_ACCOUNTS")).map((f) => {
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
      {xAcc && <XAccounts item={item} />}
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
      <div className="kc-a" hidden={xAcc}>
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
        {msg ? msg[1] : item.message?.includes(".env") ? "Сохранён, ещё не проверен — нажмите «Проверить и сохранить»." : item.message}
      </p>
    </article>
  );
}

/** Один ключ строкой там, где он работает (Bybit — в «Ордере», Telegram-бот — в уведомлениях):
 * статус и кнопка; форма ключа выезжает панелью. */
const ICON: Record<string, ReactNode> = {
  news: <Newspaper />,
  glassnode: <Cube />,
  finnhub: <CalendarDots />,
  pumpportal: <Fire />,
};
export const keyIcon = (id: string): ReactNode => brandLogo(id) ?? ICON[id] ?? <Key />;

export function KeyLine({ service, label }: { service: string; label?: string }) {
  const item = useStore((s) => s.secrets?.find((x) => x.service === service));
  const [open, setOpen] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  if (!item) return null;
  const [c, text] = STATUS[item.status];
  const last4 = item.fields.find((f) => f.secret && f.last4)?.last4;
  // Ключ сохранён, но не проверен (давний перенос или сеть не ответила) — проверяем одной кнопкой.
  const unsure = item.status === "warn" || item.status === "unchecked";
  // Старые ключи помечены «Перенесено из .env»: файла .env больше нет, пишем по сути.
  const msg = item.message?.includes(".env") ? "сохранён, ещё не проверен" : item.message;
  const check = async () => {
    setBusy(true);
    await actions.checkSecret(service);
    setBusy(false);
  };
  return (
    <>
      <IRow
        icon={keyIcon(service)}
        title={label ?? item.name}
        sub={`${unsure && msg ? msg : text}${last4 ? ` · ••••${last4}` : ""}`}
        tone={c === "ok" ? "ok" : c === "bad" ? "bad" : c === "warn" ? "warn" : undefined}
      >
        {unsure && (
          <button className="btn primary" type="button" disabled={busy} onClick={() => void check()}>
            {busy ? "Проверяю…" : "Проверить"}
          </button>
        )}
        <button className="btn" type="button" onClick={() => setOpen(service)}>
          {item.status === "unset" ? "Подключить" : "Заменить"}
        </button>
      </IRow>
      <Keys open={open} setOpen={setOpen} only={[]} />
    </>
  );
}
