// Шаги решения по узлам конвейера (src/decision_trace.py): «Почему открыта»
// в терминале и журнале, итог цикла на узлах вкладки «Бот».

import type { BotNode, TraceStep } from "../lib/types";

export const NODE_TITLE: Record<BotNode, string> = {
  scan: "Сканер",
  strat: "Стратегии",
  ai: "AI-цепочка",
  sac: "SAC",
  risk: "Риск",
  order: "Ордер",
  track: "Сопровождение",
};

/** ▲ — узел пропустил дальше, ▼ — остановил, – — пропущен или не участвует. */
export const mark = (ok: boolean | null | undefined): [string, string, string] =>
  ok === true ? ["▲", "ok", "пройдено"] : ok === false ? ["▼", "bad", "остановил"] : ["–", "", "пропущено"];

export function TraceSteps({ steps }: { steps: TraceStep[] }) {
  return (
    <ol className="trace" aria-label="Шаги решения">
      {steps.map((s) => {
        const [glyph, cls, said] = mark(s.ok);
        return (
          <li key={s.node} className={cls}>
            <i aria-hidden="true">{glyph}</i>
            <b>{NODE_TITLE[s.node] ?? s.node}</b>
            <span>
              <span className="sr">{said}: </span>
              {s.text}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
