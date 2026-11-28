// Настройки → Подключения: Telegram — вход аккаунта, каналы настроения и доверенные каналы.

import { useEffect, useState, type FormEvent } from "react";

import { plural } from "../../lib/format";

import type { SecretItem, TgStatus } from "../../lib/types";
import { actions } from "../../store/actions";
import { useStore } from "../../store";
import { Toggle } from "../../components/controls";
import { Q } from "../../components/tooltip";
import { cfg } from "../bots";
import { ParamList } from "../params";
import { TG_STATUS_POLL_MS } from "../../constants";

// ── Каналы Telegram: какие читать и каким доверять ────────────────────────────

const TRUST_PARAMS = ["TELEGRAM_TRUSTED_RISK", "TELEGRAM_TRUSTED_SLOTS", "TELEGRAM_TRUSTED_SL", "TELEGRAM_TRUSTED_DEDUP_MIN", "TELEGRAM_TRUSTED_UNTRUST"];

/** Как channel_names на сервере: «@Chan, https://t.me/chan2» → ["chan", "chan2"]. */
export const chanList = (raw: unknown): string[] => [
  ...new Set(
    String(raw ?? "")
      .split(/[,\s]+/)
      .map((c) => c.replace(/^(?:https?:\/\/)?(?:t\.me\/|telegram\.me\/)?@?/, "").replace(/\/+$/, "").toLowerCase())
      .filter(Boolean),
  ),
];

/** Каналы, которые бот читает, и доверенные: пост доверенного канала сразу становится сделкой. */
const ago = (ts: number): string => {
  const m = Math.round((Date.now() / 1000 - ts) / 60);
  return m < 1 ? "только что" : m < 60 ? `${m} мин назад` : m < 1440 ? `${Math.round(m / 60)} ч назад` : `${Math.round(m / 1440)} дн. назад`;
};

/** Что бот делает с каналом прямо сейчас: [тон точки, строка]. */
function chanState(rt: TgStatus | null, ch: string): [string, string] {
  if (!rt || rt.state === "unknown") return ["", "бот ещё не сообщал о каналах"];
  if (rt.state !== "ok") return ["warn", `не читается: ${TG_STATE[rt.state][1]}`];
  const c = rt.per_channel?.[ch];
  if (!c) return ["", "добавлен — бот подключит его в течение минуты"];
  if (c.state === "failed") return ["bad", `не читается: ${c.why}`];
  if (c.state === "connecting") return ["", "подключается…"];
  const win = rt.window_min ?? 60;
  const last = c.last_at ? `, последний ${ago(c.last_at)}` : "";
  return ["ok", c.posts ? `читается · ${c.posts} ${plural(c.posts, ["пост", "поста", "постов"])} за ${win} мин${last}` : `читается · за ${win} мин постов нет${last}`];
}

export function TgChannelsCard({ openKey }: { openKey: (service: string) => void }) {
  const config = useStore((s) => s.config);
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
  // Что бот делает с каждым каналом — обновляется раз в 30 с, пока карточка на экране.
  const [rt, setRt] = useState<TgStatus | null>(null);
  const listKey = all.join(",");
  useEffect(() => {
    const load = () => void actions.tgStatus().then(setRt);
    load();
    const t = setInterval(load, TG_STATUS_POLL_MS);
    return () => clearInterval(t);
  }, [listKey]);
  const reading = all.filter((ch) => rt?.per_channel?.[ch]?.state === "reading").length;
  // Две роли: доверенные торгуют по постам (и тоже дают настроение), остальные — только сбор информации.
  const tradeList = all.filter((ch) => trusted.includes(ch));
  const infoList = all.filter((ch) => !trusted.includes(ch));
  const row = (ch: string) => {
    const [tone, line] = chanState(rt, ch);
    const title = rt?.per_channel?.[ch]?.title;
    const link = /^-?\d+$/.test(ch) ? null : `https://t.me/${ch}`;
    return (
      <div className="alert-row" key={ch}>
        <div className="tgch-c">
          <b>
            <span className={`dot ${tone}`} />
            {title || (link ? `@${ch}` : `канал ${ch}`)}
          </b>
          <small>
            {link ? (
              <a href={link} target="_blank" rel="noopener noreferrer">
                @{ch} ↗
              </a>
            ) : (
              `id ${ch}`
            )}{" "}
            · <span className={tone === "bad" || tone === "warn" ? "tgch-bad" : ""}>{line}</span>
          </small>
        </div>
        <span className="tgch-a">
          <Toggle on={trusted.includes(ch)} label={`Доверять @${ch}: входить по постам`} onChange={(v) => trust(ch, v)} />
          <button type="button" className="btn" aria-label={`Убрать @${ch}`} onClick={() => remove(ch)}>
            Убрать
          </button>
        </span>
      </div>
    );
  };
  return (
    <article className="card tgch" id="tg-channels">
      <div className="card-h">
        <h2>Каналы Telegram</h2>
        <Q tip="Обычный канал даёт боту настроение и упоминания монет. Доверенный ещё и открывает сделку по посту. Тумблер «Доверять» переключает роль." />
      </div>
      <div className="tgch-h">
        <b>
          Отслеживаем: {all.length} {plural(all.length, ["канал", "канала", "каналов"])}
          {rt?.state === "ok" && ` · читаются ${reading}`}
        </b>
        {rt && ["off", "no_session", "revoked"].includes(rt.state) && (
          <button type="button" className="btn" onClick={() => openKey("telethon")}>
            Вход в Telegram →
          </button>
        )}
      </div>

      <section className="tgch-g" aria-label="Доверенные каналы: торговля по постам">
        <div className="tgch-gh">
          <b>Торгуем по постам — доверенные · {tradeList.length}</b>
          <small>пост сразу становится сделкой и тоже идёт в настроение</small>
        </div>
        <div className="alert-row">
          <div>
            <b>Сделки по доверенным каналам</b>
            <small>в любом профиле, мимо сканера, стратегий и AI</small>
          </div>
          <Toggle on={on} label="Сделки по доверенным каналам" onChange={(v) => void actions.patchConfig({ TELEGRAM_TRUSTED_TRADING: v })} />
        </div>
        <p className="kf-h">
          Стратегии и AI такой вход не проверяют и не закрывают — только стоп, тейк и трейлинг. Защиты остаются: стоп всегда, свой риск и слоты, пауза, дневной
          лимит, спред, одна сделка на монету, режим счёта — сейчас {paper ? "Paper" : "Live"}.
        </p>
        <div className="list-tgl">
          {tradeList.map(row)}
          {!tradeList.length && <p className="kf-h">Доверенных нет — включите «Доверять» у канала из сбора информации ниже.</p>}
        </div>
        <ParamList keys={TRUST_PARAMS} />
      </section>

      <section className="tgch-g" aria-label="Каналы только для сбора информации">
        <div className="tgch-gh">
          <b>Только сбор информации · {infoList.length}</b>
          <small>посты дают настроение и упоминания монет текущего скана — обычным путём, через все проверки</small>
        </div>
        <div className="list-tgl">
          {infoList.map(row)}
          {!infoList.length && <p className="kf-h">{all.length ? "Все каналы доверенные." : "Каналов нет — добавьте первый ниже."}</p>}
        </div>
        <form className="tgch-add" onSubmit={addChannel}>
          <input className="inp" aria-label="Добавить канал для сбора информации" placeholder="@канал или ссылка t.me/…" value={add} onChange={(e) => setAdd(e.target.value)} />
          <button className="btn" type="submit" disabled={!add.trim()}>
            Добавить
          </button>
        </form>
      </section>
    </article>
  );
}

/** Код привязки: 6 цифр, их видно только здесь. */
const linkCode = () => String(crypto.getRandomValues(new Uint32Array(1))[0] % 1_000_000).padStart(6, "0");

/** Привязка Telegram: вы отправляете боту код из приложения, мы находим ваш chat id и подставляем его в поле. */
export function ChatFinder({ token, hasToken, onFound }: { token: string; hasToken: boolean; onFound: (id: string) => void }) {
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
export function TgLogin({ item }: { item: SecretItem }) {
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
