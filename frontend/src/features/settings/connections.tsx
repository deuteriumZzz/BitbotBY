// Настройки → Подключения: карта подключений и карточка аккаунта биржи.

import { useEffect, useRef, useState, type FormEvent } from "react";
import { ArrowClockwise, CurrencyDollar, Flask, SignOut } from "@phosphor-icons/react";


import { fmtN, plural } from "../../lib/format";
import * as platform from "../../lib/platform";

import { actions } from "../../store/actions";
import { useStore } from "../../store";
import { useArm } from "../../components/controls";
import { cfg } from "../bots";
import { IRow } from "../../components/irow";
import { KeyLine, Keys } from "./keys";


/** Paper ↔ Live с защитой (ключ Bybit, PIN, подтверждение сервера). Живёт в шапке у значка PAPER/LIVE. */
export function AccountSwitch({ openKey }: { openKey: (service: string) => void }) {
  const paper = useStore((s) => s.bot?.paper ?? true);
  const secrets = useStore((s) => s.secrets);
  const risk = useStore((s) => cfg(s.config, "RISK_PER_TRADE", 0.02));
  const slots = useStore((s) => cfg(s.config, "MAX_POSITIONS", 3));
  const balance = useStore((s) => s.balance);
  const [box, setBox] = useState(false);
  const [busy, setBusy] = useState(false);
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
    <>
      <IRow
        icon={paper ? <Flask /> : <CurrencyDollar />}
        tone={paper ? "ok" : "bad"}
        title={paper ? "Paper — виртуальные сделки" : "Live — реальные деньги"}
        tip={
          paper
            ? "Деньги не тратятся. Переход на реальные деньги защищён PIN и подтверждением на сервере."
            : "Бот торгует реальными деньгами. Вернуться в Paper можно в любой момент: открытые позиции на бирже останутся, бот продолжит их сопровождать."
        }
        sub={paper ? "деньги не тратятся" : "бот торгует реальными деньгами"}
      >
        {paper ? (
          <button className="btn danger" type="button" onClick={() => setBox(!box)}>
            На Live…
          </button>
        ) : (
          <button className="btn" type="button" disabled={busy} onClick={() => void toPaper()}>
            В Paper
          </button>
        )}
      </IRow>
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
              <span>Без PIN переход на реальные деньги невозможен. Задаётся кнопкой ниже.</span>
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
    </>
  );
}

export function AccountCard() {
  const [restartArmed, armRestart] = useArm();
  const [logoutArmed, armLogout] = useArm();
  return (
    <article className="card">
      <div className="card-h">
        <h2>Счёт и бот</h2>
      </div>
      <KeyLine service="bybit" />
      <IRow icon={<ArrowClockwise />} title="Перезапуск бота" sub="нужен после смены ключей">
        <button className={`btn${restartArmed ? " armed" : ""}`} type="button" onClick={() => armRestart() && void actions.restart()}>
          {restartArmed ? "Точно?" : "Перезапустить"}
        </button>
      </IRow>
      {/* В MiniApp вход — через Telegram, выходить некуда; без сессии (вход не настроен) — тоже. */}
      {platform.kind !== "miniapp" && platform.getToken() && (
        <IRow icon={<SignOut />} title="Это устройство">
          <button className={`btn${logoutArmed ? " armed" : ""}`} type="button" onClick={() => armLogout() && void useStore.getState().logout()}>
            {logoutArmed ? "Точно выйти?" : "Выйти"}
          </button>
        </IRow>
      )}
    </article>
  );
}

/** Значок PAPER/LIVE в шапке — он же переключатель счёта: по клику окошко с переходом (защита та же). */
export function AccountBadge() {
  const paper = useStore((s) => s.bot?.paper ?? true);
  const [open, setOpen] = useState(false);
  const [key, setKey] = useState<string | null>(null);
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && !key && setOpen(false);
    const onDown = (e: PointerEvent) => !key && box.current && !box.current.contains(e.target as Node) && setOpen(false);
    window.addEventListener("keydown", onKey);
    window.addEventListener("pointerdown", onDown);
    return () => (window.removeEventListener("keydown", onKey), window.removeEventListener("pointerdown", onDown));
  }, [open, key]);
  return (
    <div className="acct" ref={box}>
      <button
        type="button"
        className={`paper${paper ? "" : " live"}`}
        aria-haspopup="dialog"
        aria-expanded={open}
        title={paper ? "Бумажная торговля: сделки виртуальные. Нажмите, чтобы перейти на Live" : "Реальная торговля на Bybit. Нажмите, чтобы вернуться в Paper"}
        onClick={() => setOpen(!open)}
      >
        {paper ? <Flask weight="bold" /> : <CurrencyDollar weight="bold" />}
        {paper ? "PAPER" : "LIVE"}
      </button>
      {open && (
        <div className="acct-pop" role="dialog" aria-label="Счёт: Paper или Live">
          <AccountSwitch openKey={setKey} />
        </div>
      )}
      <Keys open={key} setOpen={setKey} only={[]} />
    </div>
  );
}
