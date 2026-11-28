// Вкладка «Настройки»: подключения, Telegram, счёт и компьютер, расширенные параметры.

import { useEffect, useState } from "react";

import { type SettingsPane } from "../../lib/where";

import { useStore } from "../../store";

import { OpsCard } from "../bots";
import { ReportsCard } from "../reports";

import { AlertsCard } from "./alerts";
import { ComputerCard } from "./computer";
import { AccountCard, ConnMap } from "./connections";
import { Keys } from "./keys";
import { AllParams } from "./params";
import { TgChannelsCard } from "./telegram";

export type Pane = SettingsPane;

/** Прокрутка к разделу страницы настроек (вместо переключения «папок»). */
const goTo = (pane: Pane | "map") => {
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  document.getElementById(`set-${pane}`)?.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
};

/** Заголовок раздела страницы настроек: номер, название, одна фраза о том, что внутри. */
function SecHead({ n, title }: { n: string; title: string }) {
  return (
    <div className="set-h">
      <span className="set-n">{n}</span>
      <h2>{title}</h2>
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
  // Переход к ключу из других вкладок («Подключить →» в «Сигналах для решения»).
  const keyJump = useStore((s) => s.keyJump);
  useEffect(() => {
    if (!keyJump) return;
    setKey(keyJump);
    useStore.setState({ keyJump: null });
  }, [keyJump]);
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
    for (const id of ["map", "conn", "main", "acct", "all"]) {
      const el = document.getElementById(`set-${id}`);
      if (el) io.observe(el);
    }
    return () => io.disconnect();
  }, [simple]);
  const chips: [Pane | "map", string][] = [
    ["map", "Карта"],
    ["conn", "Подключения"],
    ["main", "Telegram"],
    ["acct", "Счёт и компьютер"],
    ["all", simple ? "Основные параметры" : "Расширенные"],
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
      <div id="set-conn" className="set-sec">
        <SecHead n="01" title="Подключения" />
        <Keys open={key} setOpen={setKey} />
      </div>
      <div id="set-main" className="set-sec">
        <SecHead n="02" title="Telegram" />
        <div className="set-pane view enter">
          <div className="cols">
            <div className="col">
              <AlertsCard />
            </div>
            <div className="col">
              <TgChannelsCard openKey={setKey} />
            </div>
          </div>
        </div>
      </div>
      <div id="set-acct" className="set-sec">
        <SecHead n="03" title="Счёт и компьютер" />
        <div className="set-pane view enter">
          <div className="cols">
            <div className="col">
              <AccountCard openKey={setKey} />
            </div>
            <div className="col">
              <ComputerCard />
              {!simple && (
                <details className="card dev">
                  <summary>Для разработчика</summary>
                  <OpsCard />
                  <ReportsCard dev />
                </details>
              )}
            </div>
          </div>
        </div>
      </div>
      <div id="set-all" className="set-sec">
        <SecHead n="04" title={simple ? "Основные параметры" : "Расширенные параметры"} />
        <AllParams go={goTo} basicOnly={simple} />
      </div>
    </section>
  );
}
