import { beforeEach, describe, expect, it, vi } from "vitest";

import { positionPnl, reduceRadar, useStore, type Radar } from "./store";

const ev = (type: string, data: Record<string, unknown>) => ({ type, ts: 0, data });
const empty: Radar = { cycle: 0, scan: [], prescore: [], signal: {}, guards: [], rejected: {}, opened: [], note: "" };

beforeEach(() => {
  vi.useFakeTimers(); // refreshSoon не должен ходить в сеть во время теста
  useStore.setState({
    positions: {},
    feed: [],
    pending: [],
    radar: empty,
    bot: { alive: true, paused: false, paper: true, mode: "ai", profile: "altcoin" },
  });
});

describe("radar", () => {
  it("ведёт монету по конвейеру и запоминает этап отказа", () => {
    let r = reduceRadar(empty, { stage: "scan", symbols: ["SOL/USDT", "ETH/USDT"] });
    expect(r.cycle).toBe(1);
    r = reduceRadar(r, { stage: "prescore", symbols: ["SOL/USDT", "ETH/USDT"] });
    r = reduceRadar(r, { stage: "signal", symbol: "SOL/USDT", side: "long", confidence: 0.7, strategy: "breakout" });
    r = reduceRadar(r, { stage: "guards", symbol: "SOL/USDT" });
    r = reduceRadar(r, { stage: "rejected", symbol: "SOL/USDT", reason: "все слоты заняты" });
    r = reduceRadar(r, { stage: "rejected", symbol: "ETH/USDT", reason: "мало ликвидности" });
    expect(r.guards).toEqual([]);
    expect(r.rejected["SOL/USDT"].lane).toBe(3);
    expect(r.rejected["ETH/USDT"].lane).toBe(1);
  });

  it("новый скан сбрасывает прошлый цикл", () => {
    const r = reduceRadar({ ...empty, guards: ["SOL/USDT"] }, { stage: "scan", symbols: [] });
    expect(r.guards).toEqual([]);
  });
});

describe("события", () => {
  it("тик создаёт карточку позиции и обновляет цену и SL", () => {
    const { apply } = useStore.getState();
    apply(ev("tick", { symbol: "SOL/USDT", trade_id: 7, side: "long", entry: 100, qty: 2, sl: 95, tp: 110, price: 101 }));
    apply(ev("tick", { symbol: "SOL/USDT", price: 104, sl: 99 }));
    const p = useStore.getState().positions["SOL/USDT"];
    expect(p.price).toBe(104);
    expect(p.sl).toBe(99);
    expect(positionPnl(p).pnl).toBeCloseTo(8);
  });

  it("закрытие убирает позицию и пишет итог в ленту", () => {
    const { apply } = useStore.getState();
    apply(ev("tick", { symbol: "ETH/USDT", trade_id: 1, side: "short", entry: 100, qty: 1, sl: 105, tp: 90, price: 97 }));
    apply(ev("position.closed", { symbol: "ETH/USDT" }));
    const s = useStore.getState();
    expect(s.positions["ETH/USDT"]).toBeUndefined();
    expect(s.feed[0]).toMatchObject({ kind: "win" });
    expect(s.feed[0].text).toContain("+$3.00");
  });

  it("пауза из Telegram видна по событию цикла", () => {
    useStore.getState().apply(ev("cycle", { state: "paused" }));
    expect(useStore.getState().bot?.paused).toBe(true);
    useStore.getState().apply(ev("cycle", { state: "scan" }));
    expect(useStore.getState().bot?.paused).toBe(false);
  });

  it("предложение сделки появляется и исчезает после ответа", () => {
    const { apply } = useStore.getState();
    apply(ev("signal.pending", { id: "sig5", symbol: "SOL/USDT", side: "long" }));
    expect(useStore.getState().pending).toHaveLength(1);
    apply(ev("signal.resolved", { id: "sig5", symbol: "SOL/USDT", confirmed: false, by: "telegram" }));
    expect(useStore.getState().pending).toHaveLength(0);
    expect(useStore.getState().feed[0].text).toContain("пропущена (в Telegram)");
  });
});
