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

  it("позиция помнит, почему открыта, а закрытие — по какой причине и цене", () => {
    const { apply } = useStore.getState();
    apply(
      ev("position.opened", {
        symbol: "SOL/USDT",
        trade_id: 3,
        side: "buy",
        entry: 100,
        qty: 1,
        stop_loss: 95,
        take_profit: 110,
        strategy: "rsi_mean_reversion",
        confidence: 0.71,
        reasoning: "RSI перепродан (28).",
      }),
    );
    expect(useStore.getState().positions["SOL/USDT"]).toMatchObject({ strategy: "rsi_mean_reversion", reasoning: "RSI перепродан (28)." });
    apply(ev("position.closed", { symbol: "SOL/USDT", reason: "tp", price: 110 }));
    expect(useStore.getState().feed[0].text).toBe("Закрыл SOL по тейк-профиту: ▲ +$10.00");
  });

  it("итог цикла и шаги решения приходят живыми событиями", () => {
    const { apply } = useStore.getState();
    const ai = { node: "ai", ok: true, text: "ответил Gemini · 2 сигнала" };
    apply(ev("trace.cycle", { at: 1, mode: "ai", state: "scan", nodes: { ai } }));
    expect(useStore.getState().lastCycle?.nodes.ai).toEqual(ai);
    // состояние цикла (скан/мониторинг/пауза) этим не затирается
    expect(useStore.getState().cycle.state).not.toBeUndefined();
    const trace = [{ node: "risk", ok: true, text: "проверки пройдены" }];
    apply(ev("position.opened", { symbol: "SOL/USDT", trade_id: 4, side: "buy", entry: 100, qty: 1, stop_loss: 95, take_profit: 110, trace }));
    expect(useStore.getState().positions["SOL/USDT"].trace).toEqual(trace);
  });

  it("пауза из Telegram видна по событию цикла", () => {
    useStore.getState().apply(ev("cycle", { state: "paused" }));
    expect(useStore.getState().bot?.paused).toBe(true);
    useStore.getState().apply(ev("cycle", { state: "scan" }));
    expect(useStore.getState().bot?.paused).toBe(false);
  });

  it("статус AI-провайдера обновляется живым событием", () => {
    const provider = (id: string) => ({ id, name: id.toUpperCase(), status: "unknown", cooldown_until: null, calls_today: 0 });
    useStore.setState({
      ai: { mode: "free", pinned: "auto", last: "", chain: [], providers: [provider("groq"), provider("gemini")] } as never,
    });
    const { apply } = useStore.getState();
    apply(ev("ai.status", { provider: "groq", model: "m", status: "rate_limit", until: 1_900_000_000 }));
    apply(ev("ai.status", { provider: "gemini", model: "m", status: "ok", until: null }));
    const s = useStore.getState();
    expect(s.ai?.providers[0]).toMatchObject({ status: "rate_limit", cooldown_until: 1_900_000_000, calls_today: 1 });
    expect(s.ai?.last).toBe("gemini");
    expect(s.bot?.ai).toBe("gemini");
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

describe("notifications", () => {
  it("открытие окна помечает все события просмотренными, закрытие номер не трогает", () => {
    useStore.setState({
      seenFeed: 0,
      feed: [
        { id: 7, at: 1, kind: "win", text: "a" },
        { id: 5, at: 0, kind: "open", text: "b" },
      ],
    });
    useStore.getState().openNotif(true);
    expect(useStore.getState().seenFeed).toBe(7);
    useStore.setState({ feed: [{ id: 9, at: 2, kind: "guard", text: "c" }, ...useStore.getState().feed] });
    useStore.getState().openNotif(false);
    expect(useStore.getState().seenFeed).toBe(7);
    expect(useStore.getState().notifOpen).toBe(false);
  });
});

describe("ссылка лаунчера", () => {
  it("берёт токен только из #local= и только достаточно длинный", async () => {
    const { parseLocalHash } = await import("./lib/platform");
    const t = "a".repeat(43);
    expect(parseLocalHash(`#local=${t}`)).toBe(t);
    expect(parseLocalHash("#local=short")).toBeNull();
    expect(parseLocalHash(`#other=${t}`)).toBeNull();
    expect(parseLocalHash(`#local=${t}&x=1`)).toBeNull();
    expect(parseLocalHash("")).toBeNull();
  });
});

describe("режим «Простой / Профи»", () => {
  it("«Простой» не трогает обычные вкладки, а узлы, обучение и параметры открывает в «Профи»", () => {
    const st = useStore.getState();
    st.setUi("simple");
    st.setTab("journal");
    expect(useStore.getState().ui).toBe("simple");
    st.openNode("risk");
    expect(useStore.getState()).toMatchObject({ ui: "pro", tab: "bots", node: "risk" });
    st.setUi("simple");
    st.setTab("learn");
    expect(useStore.getState().ui).toBe("pro");
    st.setUi("simple");
    st.searchParams("риск");
    expect(useStore.getState()).toMatchObject({ ui: "pro", tab: "settings" });
  });
});
