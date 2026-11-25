// Настройки → Основное: карточка «Компьютер» — автозапуск, не спать, доступ с телефона (только при запуске лаунчером).

import { useEffect, useState } from "react";

import { api } from "../../lib/api";

import { useStore } from "../../store";
import { Toggle } from "../../components/controls";
import { Q } from "../../components/tooltip";

interface Launcher {
  local: boolean;
  autostart: boolean | null;
  keep_awake: boolean | null;
  phone: boolean | null;
  at: number | null;
  /** Помощник лаунчера на компьютере жив и забирает запросы. */
  helper: boolean;
  result: { id: string; ok: boolean; message: string; at: number } | null;
}

const MACHINE = [
  ["autostart", "Запуск при входе в систему", "autostart"],
  ["keep_awake", "Не давать компьютеру уснуть", "awake"],
  ["phone", "MiniApp в Telegram (доступ с телефона)", "phone"],
] as const;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Компьютер, на котором работает бот. Приложение в Docker кладёт запрос в общую папку,
 * помощник лаунчера выполняет только разрешённое; телефон отсюда можно лишь выключить.
 */
export function ComputerCard() {
  const [st, setSt] = useState<Launcher | null>(null);
  const [busy, setBusy] = useState("");
  const [copied, setCopied] = useState("");
  const load = () =>
    api<Launcher>("GET", "/api/launcher")
      .then((v) => (setSt(v), v))
      .catch(() => null);
  useEffect(() => {
    void load();
  }, []);
  if (!st?.local) return null;
  const say = useStore.getState().say;
  const copy = async (cmd: string) => {
    try {
      await navigator.clipboard.writeText(cmd);
      setCopied(cmd);
    } catch {
      setCopied("");
    }
  };
  const toggle = async (setting: string, on: boolean) => {
    setBusy(setting);
    try {
      const { id } = await api<{ id: string }>("POST", "/api/launcher", { setting, on });
      // Помощник смотрит папку раз в 2 с; выключение телефона ещё и перезапускает приложение.
      for (let i = 0; i < 40; i++) {
        await sleep(1500);
        const v = await load();
        if (v?.result?.id === id) return say(v.result.message || (v.result.ok ? "Готово" : "Не получилось"));
      }
      say("Компьютер не ответил. Проверьте, что бот запущен ярлыком BitbotBY.");
    } catch (e) {
      say(e instanceof Error ? e.message : "Не получилось.");
    } finally {
      setBusy("");
    }
  };
  return (
    <article className="card">
      <div className="card-h">
        <h2>Компьютер</h2>
        <Q tip="Бот работает в Docker на этом компьютере. Тумблеры передают просьбу помощнику лаунчера на компьютере — он умеет только эти действия. Доступ с телефона отсюда можно лишь выключить: включение открывает приложение в интернет, поэтому только командой на самом компьютере." />
      </div>
      {st.at === null && <p className="note-soon">Лаунчер ещё не сообщил о себе — перезапустите ярлыком BitbotBY, и здесь появится состояние.</p>}
      {st.at !== null && !st.helper && (
        <p className="note-soon">Помощник на компьютере не отвечает — переключать можно командой в Терминале. Он запускается вместе с ботом ярлыком BitbotBY.</p>
      )}
      <ul className="machine">
        {MACHINE.map(([k, label, cmd]) => {
          const on = st[k];
          const can = st.helper && on !== null && (k !== "phone" || on);
          const next = `./bitbot ${cmd} ${on ? "off" : "on"}`;
          return (
            <li key={k}>
              <span className={`dot${on ? " ok" : ""}`} />
              <span>
                {label}: <b>{busy === k ? "меняю…" : on === null ? "неизвестно" : on ? "включено" : "выключено"}</b>
              </span>
              <span className="machine-act">
                {can ? (
                  <Toggle on={Boolean(on)} label={label} disabled={busy !== ""} onChange={(v) => void toggle(k, v)} />
                ) : (
                  <button className="btn mono" type="button" title="Скопировать команду" onClick={() => void copy(next)}>
                    {copied === next ? "Скопировано ✓" : next}
                  </button>
                )}
              </span>
              {k === "phone" && !on && st.helper && <p className="machine-note">Включается только на компьютере: это открывает приложение в интернет.</p>}
            </li>
          );
        })}
      </ul>
      <p className="note-soon">Нет доступа к Терминалу? Ссылку входа присылает бот в Telegram по команде /app.</p>
    </article>
  );
}
