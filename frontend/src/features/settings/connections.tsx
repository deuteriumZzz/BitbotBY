// Настройки → Подключения: карта подключений и карточка аккаунта биржи.

import { useEffect, useState, type FormEvent } from "react";

import { aiLine } from "../ai";
import { api } from "../../lib/api";

import { fmtN, plural } from "../../lib/format";
import * as platform from "../../lib/platform";

import type { SecretItem } from "../../lib/types";
import { actions } from "../../store/actions";
import { useStore } from "../../store";
import { useArm } from "../../components/controls";
import { Icon } from "../../components/icon";
import { cfg } from "../bots";

import { Pane } from "./index";
import { STATUS } from "./trading";
import { EP } from "../../api/endpoints";

type Link = "ok" | "warn" | "bad" | "off";

interface MapNode {
  id: string;
  label: string;
  sub: string;
  st: Link;
  act?: () => void;
}

/** Карта подключений: что питает бота слева, бот в центре, что он отдаёт справа. По живым связям бегут точки. */
export function ConnMap({ go, openKey }: { go: (pane: Pane | "map") => void; openKey: (service: string) => void }) {
  const secrets = useStore((s) => s.secrets);
  const ai = useStore((s) => s.ai);
  const note = useStore((s) => s.secretsNote);
  const wsUp = useStore((s) => s.wsUp);
  const alive = useStore((s) => s.alive);
  const paper = useStore((s) => s.bot?.paper ?? true);
  // Где работает бот: лаунчер на компьютере отвечает local=true, на сервере его нет.
  const [local, setLocal] = useState<boolean | null>(null);
  useEffect(() => {
    api<{ local: boolean }>("GET", EP.launcher)
      .then((v) => setLocal(v.local))
      .catch(() => setLocal(false));
  }, []);
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
      label: "AI",
      sub: aiLine(ai),
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
