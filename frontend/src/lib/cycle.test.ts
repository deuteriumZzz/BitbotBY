import { describe, expect, it } from "vitest";
import { cycleLine } from "./cycle";

describe("cycleLine", () => {
  it("names the step that stopped the cycle", () => {
    const c = { at: 0, mode: "ai", state: "scan", nodes: { scan: { node: "scan", ok: true, text: "20 монет" }, risk: { node: "risk", ok: false, text: "отсеяно 2" } } };
    expect(cycleLine(c as never)).toBe("Цикл: Риск — отсеяно 2");
  });
  it("falls back to the last passed step", () => {
    const c = { at: 0, mode: "ai", state: "scan", nodes: { scan: { node: "scan", ok: true, text: "20 монет" }, track: { node: "track", ok: null, text: "—" } } };
    expect(cycleLine(c as never)).toBe("Цикл: Сканер — 20 монет");
  });
});

import { tfText } from "../features/pipeline";

describe("tfText", () => {
  it("speaks timeframes in Russian", () => {
    expect(tfText("15m")).toBe("15 мин");
    expect(tfText("1h")).toBe("1 ч");
    expect(tfText("1d")).toBe("1 д");
    expect(tfText("odd")).toBe("odd");
  });
});
