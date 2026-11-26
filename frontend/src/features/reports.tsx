// Отчёты бота (только чтение): сводка по журналу решений, разбор одного решения
// «данные → решение → исполнение → итог», проверка целостности базы сделок.

import { useState } from "react";

import { api } from "../lib/api";
import { Seg } from "../components/controls";
import { EP } from "../api/endpoints";

type Hours = 6 | 24 | 168;
const HOURS: readonly (readonly [Hours, string])[] = [
  [6, "6 ч"],
  [24, "Сутки"],
  [168, "Неделя"],
];

export function ReportsCard() {
  const [hours, setHours] = useState<Hours>(24);
  const [decision, setDecision] = useState("");
  const [text, setText] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  const run = async (path: string) => {
    setBusy(true);
    setErr("");
    try {
      setText((await api<{ text: string }>("GET", path)).text);
    } catch (e) {
      setText("");
      setErr(e instanceof Error ? e.message : "Не удалось построить отчёт.");
    } finally {
      setBusy(false);
    }
  };

  const id = decision.trim();
  return (
    <article className="card rep">
      <div className="card-h">
        <h2>Отчёты</h2>
      </div>
      <div className="btn-row">
        <Seg value={hours} options={HOURS} onChange={setHours} label="Период сводки" />
        <button type="button" className="btn" disabled={busy} onClick={() => void run(EP.reportsDigest(hours))}>
          Сводка
        </button>
        <button type="button" className="btn" disabled={busy} onClick={() => void run(EP.reportsDbcheck)}>
          Проверить базу
        </button>
      </div>
      <form
        className="btn-row rep-audit"
        onSubmit={(e) => {
          e.preventDefault();
          if (id) void run(EP.reportsAudit(id));
        }}
      >
        <input className="inp" value={decision} onChange={(e) => setDecision(e.target.value)} placeholder="ID решения из журнала" aria-label="ID решения" />
        <button type="submit" className="btn" disabled={busy || !id}>
          Разобрать решение
        </button>
      </form>
      {err && <p className="err">{err}</p>}
      {text && <pre className="job-out">{text}</pre>}
    </article>
  );
}
