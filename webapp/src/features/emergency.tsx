// Аварийное управление: уровни 1–4 и сброс. Каждое действие — в два шага:
// сначала бот показывает, что будет сделано, затем второе нажатие подтверждает.
// Уровень только растёт; понизить можно лишь сбросом (торговля остаётся на паузе).

import { useEffect, useState } from "react";

import { api, command } from "../lib/api";
import { useStore } from "../store";
import { Icon, useArm } from "../ui";

interface EmergencyState {
  level: number;
  by?: string;
  ts?: string;
  reason?: string;
  levels: Record<string, string>;
}

interface Preview {
  action: string;
  steps?: string[];
  reset?: boolean;
}

export function EmergencyCard() {
  const [state, setState] = useState<EmergencyState | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [armed, arm] = useArm();

  const load = () =>
    api<EmergencyState>("GET", "/api/emergency")
      .then(setState)
      .catch(() => setErr("Не удалось получить аварийный уровень."));
  useEffect(() => {
    void load();
  }, []);

  const ask = async (action: string) => {
    setErr("");
    try {
      const p = await api<Preview>("POST", "/api/emergency", { action });
      setPreview({ ...p, action });
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Не получилось.");
    }
  };

  const confirm = async () => {
    if (!preview || !arm()) return; // первое нажатие только взводит кнопку
    setBusy(true);
    try {
      await command("POST", "/api/emergency", { action: preview.action, confirm: true });
      useStore.getState().say(preview.reset ? "Авария снята. Бот на паузе — продолжите вручную." : "Аварийный уровень включён.");
      setPreview(null);
      await load();
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Не получилось.");
    } finally {
      setBusy(false);
    }
  };

  const level = state?.level ?? 0;
  return (
    <article className="card">
      <div className="card-h">
        <h2>Аварийное управление</h2>
        <span className="pill">{level === 0 ? "норма" : `уровень ${level}`}</span>
      </div>
      {level > 0 && state && (
        <div className="honest">
          {Icon.warn}
          <div>
            <b>Уровень {level}.</b> {state.levels[String(level)]}
            {state.reason ? ` · ${state.reason}` : ""}
          </div>
        </div>
      )}
      {!preview && state && (
        <div className="btn-row">
          {[1, 2, 3, 4]
            .filter((n) => n > level)
            .map((n) => (
              <button key={n} type="button" className="btn danger" title={state.levels[String(n)]} onClick={() => void ask(String(n))}>
                Уровень {n}
              </button>
            ))}
          {level > 0 && (
            <button type="button" className="btn" onClick={() => void ask("reset")}>
              Сбросить
            </button>
          )}
        </div>
      )}
      {preview && (
        <div>
          <p className="note-soon">{preview.reset ? "Авария будет снята, торговля останется на паузе." : "Будет выполнено:"}</p>
          {!preview.reset && (
            <ol>
              {(preview.steps ?? []).map((s) => (
                <li key={s}>{s.replace(/^\d+\.\s*/, "")}</li>
              ))}
            </ol>
          )}
          <div className="btn-row">
            <button type="button" className={`btn danger${armed ? " armed" : ""}`} disabled={busy} onClick={() => void confirm()}>
              {armed ? "Нажмите ещё раз" : "Подтвердить"}
            </button>
            <button type="button" className="btn" disabled={busy} onClick={() => setPreview(null)}>
              Отмена
            </button>
          </div>
        </div>
      )}
      {err && <p className="err">{err}</p>}
    </article>
  );
}
