import { describe, expect, it } from "vitest";

import { OWN_HOME } from "../lib/where";
import type { SettingDef } from "../lib/types";
import { formatSetting, takesPart } from "./pipeline";

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
