// «Почему бот молчит?» — одна причина простыми словами, почему последний цикл не открыл сделку.

import type { BotNode, CycleTrace } from "./types";

export interface IdleInput {
  alive: boolean;
  paused: boolean;
  cycle: CycleTrace | null;
  slots: { used: number; max: number } | null;
  /** TRADING_HOURS: "8-20" в часах UTC, пусто — круглосуточно. */
  hours: string;
  /** Текущий час UTC. */
  hourUtc: number;
}

export interface IdleReason {
  tone: "ok" | "warn" | "bad";
  title: string;
  detail: string;
  /** Шаг цепочки, где это настраивается. */
  node?: BotNode;
}

/** Входит ли час UTC в диапазон «a-b»; диапазон через полночь («22-6») тоже работает. */
export function inHours(hours: string, h: number): boolean {
  if (!hours) return true;
  const [a, b] = hours.split("-").map(Number);
  return a < b ? h >= a && h < b : h >= a || h < b;
}

const ORDER: BotNode[] = ["scan", "strat", "ai", "sac", "risk", "order", "track"];

/** null — молчать не о чем: сделка только что открыта. Порядок проверок — от причины, которая перекрывает остальные. */
export function idleReason(x: IdleInput): IdleReason | null {
  if (!x.alive) return { tone: "bad", title: "Бот не отвечает", detail: "Процесс бота не присылает отчёты. Перезапустите его на компьютере или сервере." };
  if (x.paused) return { tone: "warn", title: "Пауза", detail: "Новые входы остановлены, открытые позиции сопровождаются. Снимите паузу переключателем вверху." };
  if (!inHours(x.hours, x.hourUtc))
    return {
      tone: "warn",
      title: "Вне часов торговли",
      detail: `Сейчас ${String(x.hourUtc).padStart(2, "0")}:00 UTC, бот открывает сделки с ${x.hours.replace("-", " до ")} UTC.`,
      node: "scan",
    };
  if (x.slots && x.slots.max > 0 && x.slots.used >= x.slots.max)
    return {
      tone: "ok",
      title: "Все слоты заняты",
      detail: `Открыто ${x.slots.used} из ${x.slots.max}. Новая сделка — после закрытия одной из позиций.`,
      node: "risk",
    };
  const c = x.cycle;
  if (!c || !Object.keys(c.nodes).length) return { tone: "ok", title: "Бот запускается", detail: "Первый цикл ещё не прошёл — причина появится через минуту-две." };
  if (c.nodes.order?.ok) return null;
  const stop = ORDER.map((n) => c.nodes[n]).find((s) => s?.ok === false);
  if (stop) return { tone: "warn", title: "Сделку остановил шаг цепочки", detail: cap(stop.text), node: stop.node };
  const skip = c.nodes.strat?.text.match(/^пропущено: (.+)/);
  if (skip) return { tone: "warn", title: "Защита пропустила цикл", detail: cap(skip[1]), node: "risk" };
  return {
    tone: "ok",
    title: "Подходящих сигналов нет",
    detail: `${cap(c.nodes.scan?.text ?? "рынок просканирован")} — условий для входа не нашлось. Это нормально: бот ждёт.`,
    node: "strat",
  };
}

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
