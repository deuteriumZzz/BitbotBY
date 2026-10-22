import { describe, expect, it } from "vitest";

import { OWN_HOME } from "../lib/where";
import type { SettingDef } from "../lib/types";
import { formatSetting, takesPart } from "./pipeline";
import { RULES, templateOf, templateValues } from "./rules";

const def = (over: Partial<SettingDef>): SettingDef => ({
  key: "X",
  group: "",
  label: "",
  type: "float",
  default: null,
  min: null,
  max: null,
  step: null,
  unit: "",
  options: [],
  hint: "",
  readonly: false,
  hot: false,
  basic: true,
  ...over,
});

describe("конвейер решений", () => {
  it("AI и SAC гаснут в режимах, где SignalCombiner их не спрашивает", () => {
    const on = (mode: string) => (["ai", "sac"] as const).filter((n) => takesPart(mode, n));
    expect(on("local")).toEqual([]);
    expect(on("ai")).toEqual(["ai"]);
    expect(on("dqn")).toEqual(["sac"]);
    expect(on("hybrid")).toEqual(["ai", "sac"]);
    expect(takesPart("local", "strat") && takesPart("dqn", "risk") && takesPart("local", "track")).toBe(true);
  });

  it("параметры вкладки «Бот» ведут в свой узел", () => {
    expect(OWN_HOME.RISK_PER_TRADE.node).toBe("risk");
    expect(OWN_HOME.PAPER_GUARDS_OFF.node).toBe("risk");
    expect(OWN_HOME.AUTO_EXECUTE.node).toBe("order");
    expect(OWN_HOME.MARKET_TYPE.node).toBe("scan");
    expect(OWN_HOME.LOCAL_FALLBACK.node).toBe("strat");
  });

  it("сводка показывает значения так же, как «Все параметры»", () => {
    expect(formatSetting(def({ type: "pct" }), 0.6)).toBe("60%");
    expect(formatSetting(def({ type: "pct" }), 0.025)).toBe("2.5%");
    expect(formatSetting(def({ type: "bool" }), true)).toBe("вкл");
    expect(formatSetting(def({ type: "choice", options: [{ value: "volatility", label: "По волатильности" }] }), "volatility")).toBe("По волатильности");
    expect(formatSetting(def({ unit: "× ATR" }), 1)).toBe("1 × ATR");
    expect(formatSetting(undefined, "")).toBe("—");
  });
});

describe("шаблоны порогов защиты", () => {
  const schema = [
    def({ key: "DAILY_LOSS_LIMIT", default: 0.05 }),
    def({ key: "CIRCUIT_BREAKER_LOSSES", default: 3 }),
    def({ key: "MAX_DRAWDOWN_PERCENT", default: 0.2 }),
    def({ key: "SYMBOL_COOLDOWN_HOURS", default: 3 }),
    def({ key: "DRAWDOWN_SCALE_THRESHOLD", default: 0.1 }),
  ];
  const conf = (v: Record<string, unknown>) => Object.fromEntries(Object.entries(v).map(([k, x]) => [k, { value: x, source: "live" }]));

  it("«Как задумано» — значения по умолчанию из схемы", () => {
    const base = templateValues("base", schema);
    expect(base.DAILY_LOSS_LIMIT).toBe(0.05);
    expect(templateOf(conf(base), schema)).toBe("base");
  });

  it("узнаёт шаблон по текущим порогам и «свои» при расхождении", () => {
    const strict = templateValues("strict", schema);
    expect(templateOf(conf(strict), schema)).toBe("strict");
    expect(templateOf(conf({ ...strict, DAILY_LOSS_LIMIT: 0.04 }), schema)).toBeNull();
  });

  it("каждое правило с порогом или тумблером ссылается на ключ", () => {
    for (const r of RULES) expect(Boolean(r.key || r.on || r.alert)).toBe(true);
  });
});
