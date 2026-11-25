// Настройки → Основное: часы торговли и списки монет (принудительные, исключённые).

import { useEffect, useState, type FormEvent } from "react";

import { inHours } from "../../lib/idle";
import type { SecretItem } from "../../lib/types";
import { actions } from "../../store/actions";
import { useStore } from "../../store";
import { Seg } from "../../components/controls";
import { Q } from "../../components/tooltip";
import { cfg } from "../bots";

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
      {custom && <p className="note-soon" style={{ marginTop: 8 }}>Свой диапазон: {hours} UTC (задаётся в «Все параметры»).</p>}
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

export const STATUS: Record<SecretItem["status"], [string, string]> = {
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
        <Q tip="«Всегда сканировать» — бот проверяет эти монеты в каждом цикле, даже если их нет в топе по объёму. «Исключить» — бот никогда ими не торгует." />
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
