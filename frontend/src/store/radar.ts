// Радар монет в сторе: стадии цикла по событиям бота (scan → prescore → signal → guards → rejected | open).

import { base, plural } from "../lib/format";
import type { Side } from "../lib/types";

export type Data = Record<string, any>;

export interface Radar {
  cycle: number;
  scan: string[];
  prescore: string[];
  signal: Record<string, { side: Side; confidence: number; strategy: string; reasoning?: string }>;
  guards: string[];
  /** lane — на каком этапе отсеяна: 0 скан, 1 отбор, 2 сигнал, 3 проверки. */
  rejected: Record<string, { reason: string; at: number; lane: number }>;
  opened: string[];
  note: string;
  /** Последний этап цикла и когда он пришёл — схема подсвечивает блок, где бот сейчас. */
  stage?: string;
  stageAt?: number;
}

export const emptyRadar = (cycle = 0): Radar => ({
  cycle,
  scan: [],
  prescore: [],
  signal: {},
  guards: [],
  rejected: {},
  opened: [],
  note: "",
});

const n = (list: unknown): number => (Array.isArray(list) ? list.length : 0);

/** Конвейер радара: scan → prescore → signal → guards → rejected | open. */
export function reduceRadar(r0: Radar, d: Data): Radar {
  const r = reduceStage(r0, d);
  return d.stage ? { ...r, stage: d.stage, stageAt: Date.now() } : r;
}

function reduceStage(r: Radar, d: Data): Radar {
  const sym: string = d.symbol;
  switch (d.stage) {
    case "scan":
      return {
        ...emptyRadar(r.cycle + 1),
        scan: d.symbols ?? [],
        note: `Сканирую ${n(d.symbols)} ${plural(n(d.symbols), ["монету", "монеты", "монет"])}`,
      };
    case "prescore":
      return { ...r, prescore: d.symbols ?? [], note: `В отбор прошли ${n(d.symbols)}: объём, ATR и тренд` };
    case "signal":
      return {
        ...r,
        signal: {
          ...r.signal,
          [sym]: { side: d.side, confidence: d.confidence, strategy: d.strategy, reasoning: d.reasoning },
        },
        note: `Сигнал по ${base(sym)}: ${d.side === "long" ? "LONG" : "SHORT"}`,
      };
    case "guards":
      return {
        ...r,
        guards: r.guards.includes(sym) ? r.guards : [...r.guards, sym],
        note: `Проверяю риски по ${base(sym)}…`,
      };
    case "rejected": {
      const lane = r.guards.includes(sym) ? 3 : r.signal[sym] ? 2 : r.prescore.includes(sym) ? 1 : 0;
      return {
        ...r,
        guards: r.guards.filter((s) => s !== sym),
        rejected: { ...r.rejected, [sym]: { reason: d.reason, at: Date.now(), lane } },
        note: `Отсеял ${base(sym)}: ${d.reason}`,
      };
    }
    case "open":
      return {
        ...r,
        guards: r.guards.filter((s) => s !== sym),
        opened: [...r.opened, sym],
        note: `Открыл ${base(sym)}. SL и TP на бирже.`,
      };
    default:
      return r;
  }
}
