// Настройки: сброс, история изменений и прочие инструменты.

import { useState } from "react";

import { api } from "../../lib/api";

import type { ConfigHistoryItem, ConfigValues, SettingDef } from "../../lib/types";
import { actions } from "../../store/actions";
import { useStore } from "../../store";

import { fmtVal, sameVal } from "./params";
import { EP } from "../../api/endpoints";

// ── Журнал изменений, копирование и вставка набора настроек ─────────────────────

const when = (ts: number): string =>
  new Date(ts * 1000).toLocaleString("ru-RU", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

export function SettingsTools({ schema, config, onImport }: { schema: SettingDef[]; config: ConfigValues; onImport: (v: Record<string, unknown>) => void }) {
  const say = useStore((s) => s.say);
  const [hist, setHist] = useState<ConfigHistoryItem[] | null>(null);
  const [paste, setPaste] = useState<string | null>(null);
  const [err, setErr] = useState("");
  const [resetAsk, setResetAsk] = useState(false);
  const val = (d: SettingDef, v: unknown): string => (v === null || v === undefined ? "—" : fmtVal(d, v));

  const loadHistory = () => void api<ConfigHistoryItem[]>("GET", EP.configHistory).then(setHist).catch(() => setHist([]));
  // Набор — только то, что вы поменяли: значения по умолчанию и «только .env» не копируются.
  const mine = () =>
    Object.fromEntries(schema.filter((d) => !d.readonly && config[d.key] && !sameVal(config[d.key].value, d.default)).map((d) => [d.key, config[d.key].value]));
  const copy = async () => {
    const text = JSON.stringify(mine(), null, 2);
    try {
      await navigator.clipboard.writeText(text);
      say(`Скопировано настроек: ${Object.keys(mine()).length}`);
    } catch {
      setPaste(text);
      say("Не удалось скопировать — набор показан ниже, скопируйте вручную.");
    }
  };
  const doImport = () => {
    try {
      const v = JSON.parse(paste ?? "");
      if (!v || typeof v !== "object" || Array.isArray(v)) throw new Error();
      const known = Object.keys(v).filter((k) => schema.some((d) => d.key === k && !d.readonly));
      onImport(Object.fromEntries(known.map((k) => [k, v[k]])));
      setErr("");
      setPaste(null);
      say(`Загружено в форму: ${known.length}. Проверьте и нажмите «Применить».`);
    } catch {
      setErr("Это не набор настроек: вставьте JSON, скопированный из приложения.");
    }
  };

  return (
    <article className="card">
      <div className="card-h">
        <h2>Набор настроек и журнал</h2>
      </div>
      <div className="btn-row">
        <button className="btn" type="button" onClick={() => void copy()}>
          Скопировать мои настройки
        </button>
        <button className="btn" type="button" onClick={() => setPaste(paste === null ? "" : null)}>
          Вставить набор…
        </button>
        <button className="btn" type="button" aria-expanded={hist !== null} onClick={() => (hist ? setHist(null) : loadHistory())}>
          {hist ? "Скрыть журнал" : "Что менялось"}
        </button>
        <button className="btn danger" type="button" onClick={() => setResetAsk(true)}>
          Сбросить всё к .env
        </button>
      </div>
      {resetAsk && (
        <div className="confirm-box">
          <span>
            Все настройки, которые вы меняли в приложении, вернутся к значениям из <code>.env</code>: режим, пауза, символы, риск, стратегии.
            Экспериментальные стратегии снова выключатся. Ключи API не затрагиваются.
          </span>
          <div className="btn-row">
            <button className="btn" type="button" onClick={() => setResetAsk(false)}>
              Отмена
            </button>
            <button
              className="btn danger"
              type="button"
              onClick={() => {
                setResetAsk(false);
                void actions.resetConfig();
              }}
            >
              Да, сбросить
            </button>
          </div>
        </div>
      )}
      {paste !== null && (
        <div className="paste">
          <textarea className="inp" rows={6} spellCheck={false} aria-label="Набор настроек (JSON)" value={paste} onChange={(e) => setPaste(e.target.value)} />
          {err && <p className="err">{err}</p>}
          <button className="btn primary" type="button" disabled={!paste.trim()} onClick={doImport}>
            Загрузить в форму
          </button>
        </div>
      )}
      {hist && (
        <ul className="hist">
          {hist.length === 0 && <li className="empty">Из приложения пока ничего не меняли.</li>}
          {hist.map((h) => (
            <li key={h.ts}>
              <time>{when(h.ts)}</time>
              <div>
                {h.changes.map((c) => {
                  const d = schema.find((s) => s.key === c.key);
                  return (
                    <p key={c.key}>
                      <b>{c.label}</b>: {d ? val(d, c.old) : String(c.old)} → {d ? val(d, c.new) : String(c.new)}
                    </p>
                  );
                })}
              </div>
            </li>
          ))}
        </ul>
      )}
    </article>
  );
}
