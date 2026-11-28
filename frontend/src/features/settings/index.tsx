// Вкладка «Настройки»: только то, без чего бот не работает. Всё про поведение бота —
// в блоках вкладки «Стратегия»; параметры целиком — поиском (Ctrl/⌘ K).

import { useEffect, useState } from "react";

import { type SettingsPane } from "../../lib/where";

import { useStore } from "../../store";

import { ComputerCard } from "./computer";
import { AccountCard } from "./connections";
import { Keys } from "./keys";
import { TelegramPanel } from "./tgpanel";
import { AllParams } from "./params";

export type Pane = SettingsPane;

const goTo = (pane: Pane | "map") => {
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  document.getElementById(`set-${pane}`)?.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
};

export function Settings() {
  const simple = useStore((s) => s.ui === "simple");
  const jump = useStore((s) => s.paramSearch);
  // Параметры показываются только по поиску: Ctrl/⌘ K или кнопка внизу.
  const [params, setParams] = useState(false);
  useEffect(() => {
    if (jump === null) return;
    setParams(true);
    setTimeout(() => goTo("all"), 0);
  }, [jump]);
  // Ключ, открытый в боковой панели (с карты или переходом из другой вкладки).
  const [key, setKey] = useState<string | null>(null);
  const keyJump = useStore((s) => s.keyJump);
  useEffect(() => {
    if (!keyJump) return;
    setKey(keyJump);
    useStore.setState({ keyJump: null });
  }, [keyJump]);
  return (
    <section className="view enter set-page" aria-label="Настройки">
      <Keys open={key} setOpen={setKey} only={[]} />
      <div className="cols">
        {/* В «Профи» Telegram — блок на схеме; в «Простом» схемы нет, поэтому здесь. */}
        {simple ? (
          <>
            <div className="col" id="set-acct">
              <AccountCard openKey={setKey} />
              <ComputerCard />
            </div>
            <div className="col" id="set-main">
              <TelegramPanel />
            </div>
          </>
        ) : (
          <>
            <div className="col" id="set-acct">
              <AccountCard openKey={setKey} />
            </div>
            <div className="col">
              <ComputerCard />
            </div>
          </>
        )}
      </div>
      {!simple && (
        <div id="set-all" className="set-params">
          {params ? (
            <>
              <AllParams go={goTo} />
              <button className="btn" type="button" onClick={() => setParams(false)}>
                Скрыть параметры
              </button>
            </>
          ) : (
            <button className="btn" type="button" onClick={() => setParams(true)}>
              Найти параметр
            </button>
          )}
        </div>
      )}
    </section>
  );
}
