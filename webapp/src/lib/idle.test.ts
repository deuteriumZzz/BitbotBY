import { describe, expect, it } from "vitest";

import { bestWindow, idleReason, inHours, type IdleInput } from "./idle";
import type { CycleTrace } from "./types";

const base: IdleInput = { alive: true, paused: false, cycle: null, slots: { used: 0, max: 3 }, hours: "", hourUtc: 12 };
const cycle = (nodes: Record<string, [boolean | null, string]>) =>
  ({
    at: 0,
    mode: "ai",
    state: "scan",
    nodes: Object.fromEntries(Object.entries(nodes).map(([node, [ok, text]]) => [node, { node, ok, text }])),
  }) as CycleTrace;

describe("«Почему бот молчит?»", () => {
  it("часы торговли, в том числе через полночь", () => {
    expect(inHours("8-20", 7)).toBe(false);
    expect(inHours("8-20", 8)).toBe(true);
    expect(inHours("22-6", 23)).toBe(true);
    expect(inHours("22-6", 12)).toBe(false);
    expect(inHours("", 3)).toBe(true);
  });

  it("прибыльное окно часов: подряд, через полночь, без окна в плюсе", () => {
    const net = Array<number>(24).fill(-1);
    [net[9], net[10], net[11], net[12], net[13]] = [3, 3, 3, -1, 5];
    expect(bestWindow(net)).toBe("9-14");
    const night = Array<number>(24).fill(-1);
    [night[22], night[23], night[0], night[1]] = [2, 2, 2, 2];
    expect(bestWindow(night)).toBe("22-2");
    expect(inHours("22-2", 1)).toBe(true);
    expect(bestWindow(Array<number>(24).fill(-1))).toBeNull();
  });

  it("называет одну причину по старшинству", () => {
    expect(idleReason({ ...base, alive: false })?.title).toBe("Бот не отвечает");
    expect(idleReason({ ...base, paused: true, hours: "8-20", hourUtc: 3 })?.title).toBe("Пауза");
    expect(idleReason({ ...base, hours: "8-20", hourUtc: 3 })).toMatchObject({ title: "Вне часов торговли", node: "scan" });
    expect(idleReason({ ...base, slots: { used: 3, max: 3 } })?.title).toBe("Все слоты заняты");
    expect(idleReason(base)?.title).toBe("Бот запускается");
  });

  it("читает последний цикл: стоп на шаге, защита, нет сигналов, сделка открыта", () => {
    const risk = cycle({ scan: [true, "40 монет"], risk: [false, "отсеяно 2: дневной лимит"], order: [null, "сделок нет"] });
    expect(idleReason({ ...base, cycle: risk })).toMatchObject({ node: "risk", detail: "Отсеяно 2: дневной лимит" });
    const guard = cycle({ scan: [true, "40 монет"], strat: [null, "пропущено: макро-пауза FOMC"] });
    expect(idleReason({ ...base, cycle: guard })).toMatchObject({ title: "Защита пропустила цикл", detail: "Макро-пауза FOMC" });
    const quiet = cycle({ scan: [true, "40 монет"], strat: [null, "сигналов нет"], order: [null, "сделок нет"] });
    expect(idleReason({ ...base, cycle: quiet })?.title).toBe("Подходящих сигналов нет");
    expect(idleReason({ ...base, cycle: cycle({ order: [true, "открыта: BTC"] }) })).toBeNull();
  });
});
