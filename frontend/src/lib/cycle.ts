// Итог цикла одной строкой для ленты «Что делает бот».

import type { BotNode, CycleTrace } from "./types";

const ORDER: BotNode[] = ["scan", "strat", "ai", "sac", "risk", "order", "track"];
const TITLE: Record<BotNode, string> = {
  scan: "Сканер",
  strat: "Стратегии",
  ai: "AI",
  sac: "SAC",
  risk: "Риск",
  order: "Ордер",
  track: "Выход",
};

/** Где цикл остановился (первый ▼), иначе — последний пройденный шаг. */
export function cycleLine(c: CycleTrace): string {
  const steps = ORDER.map((n) => [n, c.nodes[n]] as const).filter(([, s]) => s);
  const stop = steps.find(([, s]) => s!.ok === false);
  if (stop) return `Цикл: ${TITLE[stop[0]]} — ${stop[1]!.text}`;
  const done = [...steps].reverse().find(([, s]) => s!.ok === true);
  return done ? `Цикл: ${TITLE[done[0]]} — ${done[1]!.text}` : "Цикл прошёл";
}
