// Настройки → Подключения: карта подключений и карточка аккаунта биржи.

import { useState, type FormEvent } from "react";


import { fmtN, plural } from "../../lib/format";
import * as platform from "../../lib/platform";

import { actions } from "../../store/actions";
import { useStore } from "../../store";
import { useArm } from "../../components/controls";
import { cfg } from "../bots";
import { KeyLine } from "./keys";


export function AccountCard({ openKey }: { openKey: (service: string) => void }) {
  const paper = useStore((s) => s.bot?.paper ?? true);
  const secrets = useStore((s) => s.secrets);
  const risk = useStore((s) => cfg(s.config, "RISK_PER_TRADE", 0.02));
  const slots = useStore((s) => cfg(s.config, "MAX_POSITIONS", 3));
  const balance = useStore((s) => s.balance);
  const [box, setBox] = useState(false);
  const [busy, setBusy] = useState(false);
  const [restartArmed, armRestart] = useArm();
  const [logoutArmed, armLogout] = useArm();
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
      <KeyLine service="bybit" />
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
      <div className="btn-row" style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--line-soft)" }}>
        <button className={`btn${restartArmed ? " armed" : ""}`} type="button" onClick={() => armRestart() && void actions.restart()}>
          {restartArmed ? "Точно перезапустить?" : "Перезапустить бота"}
        </button>
        {/* В MiniApp вход — через Telegram, выходить некуда; без сессии (вход не настроен) — тоже. */}
        {platform.kind !== "miniapp" && platform.getToken() && (
          <button className={`btn${logoutArmed ? " armed" : ""}`} type="button" onClick={() => armLogout() && void useStore.getState().logout()}>
            {logoutArmed ? "Точно выйти?" : "Выйти на этом устройстве"}
          </button>
        )}
      </div>
    </article>
  );
}
