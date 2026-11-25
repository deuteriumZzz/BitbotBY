// Настройки → Подключения: аккаунты X (Twitter) для чтения без официального API.

import { useState } from "react";

import type { SecretItem } from "../../lib/types";
import { actions } from "../../store/actions";

// ── X / Twitter: бесплатный обход официального API ────────────────────────────

/** Каким способом бот читает X: официальный API главнее аккаунтов. */
export function xWay(s: SecretItem): string {
  if (s.fields.find((f) => f.env === "TWITTER_BEARER_TOKEN")?.set) return "официальный API";
  const n = s.accounts?.length ?? 0;
  return n ? `через аккаунты X: ${n}` : "";
}

export type XWayKind = "acc" | "api";

/** Два способа подключить X одним переключателем и строка о том, какой работает сейчас. */
export function XWay({ item, way, setWay }: { item: SecretItem; way: XWayKind; setWay: (w: XWayKind) => void }) {
  return (
    <>
      <p className="kc-d">
        Сейчас бот читает X: <b>{xWay(item) || "никак, выберите способ"}</b>. Изменения заработают после перезапуска бота.
      </p>
      <div className="seg full" role="group" aria-label="Способ подключения X">
        <button type="button" aria-pressed={way === "acc"} onClick={() => setWay("acc")}>
          Через аккаунты
        </button>
        <button type="button" aria-pressed={way === "api"} onClick={() => setWay("api")}>
          Официальный API
        </button>
      </div>
      {way === "api" && (
        <p className="kc-d">
          Надёжно и без риска блокировки аккаунтов.{" "}
          <a href="https://developer.twitter.com" target="_blank" rel="noopener noreferrer">
            Где взять ↗
          </a>
        </p>
      )}
    </>
  );
}

const X_FIELDS: [keyof XForm, string, boolean][] = [
  ["login", "Логин в X", false],
  ["password", "Пароль от X", true],
  ["email", "Почта аккаунта", false],
  ["email_password", "Пароль от почты", true],
];

type XForm = { login: string; password: string; email: string; email_password: string };

const X_EMPTY: XForm = { login: "", password: "", email: "", email_password: "" };

/** Аккаунты отдельными полями: каждое значение хранится зашифрованным, паролей приложение не показывает. */
export function XAccounts({ item }: { item: SecretItem }) {
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
    <>
      <div className="honest">
        <div>
          <b>Бесплатно, но против правил X.</b> Бот входит в X под вашими аккаунтами и ищет твиты так же, как сайт x.com в браузере, через открытую
          библиотеку{" "}
          <a href="https://github.com/vladkens/twscrape" target="_blank" rel="noopener noreferrer">
            twscrape ↗
          </a>
          . X может заблокировать такие аккаунты, поэтому берите <b>отдельные</b>, а не основной.
          {official ? " Сейчас задан официальный API, и он главнее: эти аккаунты не используются." : ""}
        </div>
      </div>
      <details className="risk-box" open={!accounts.length}>
        <summary>Как подготовить аккаунт</summary>
        <ol className="x-steps">
          <li>Заведите новый аккаунт X на отдельную почту, не на ту, что вам важна.</li>
          <li>Войдите в него в браузере и пройдите проверки X (телефон, капча), чтобы вход не был заблокирован.</li>
          <li>Выключите на нём двухфакторную защиту X: бот не умеет вводить код из приложения.</li>
          <li>
            В почте включите доступ по IMAP и создайте «пароль приложения» (Gmail, Яндекс и Mail.ru требуют его вместо обычного). По нему бот прочитает код,
            который X присылает при входе.
          </li>
          <li>Добавьте аккаунт ниже и перезапустите бота. Лучше 2–3 аккаунта: когда у одного кончится лимит запросов, бот возьмёт следующий.</li>
        </ol>
      </details>
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
    </>
  );
}
