import { describe, expect, it } from "vitest";
import { opsSummary } from "./format";
import type { OpsStatus } from "./types";

const base: OpsStatus = {
  mode: "paper", prompt_version: "v3", flags: { WS_SHADOW_ENABLED: true, AI_SANITIZE_PROMPT: false, UNKNOWN: true },
  forecaster: "none", ws: null, backup: null, emergency: 0, ts: "2026-11-24T10:00:00Z",
};

describe("opsSummary", () => {
  it("показывает только известные флаги", () => {
    const v = opsSummary(base);
    expect(v.flags.map((f) => [f.key, f.on])).toEqual([["WS_SHADOW_ENABLED", true], ["AI_SANITIZE_PROMPT", false]]);
    expect(v.ws).toBeNull();
  });
  it("тревога при молчащих потоках и ошибке копии", () => {
    const v = opsSummary({ ...base, ws: { streams: 4, silent: ["a"], reconnects: 0, lag_p99_ms: null, gaps: 0 }, backup: { ok: false, at: "", path: null, error: "диск" } });
    expect(v.ws).toBe("молчат 1 из 4 потоков");
    expect(v.wsAlarm && v.backupAlarm).toBe(true);
    expect(v.backup).toBe("копия не создана: диск");
  });
  it("норма WS и копия", () => {
    const v = opsSummary({ ...base, ws: { streams: 4, silent: [], reconnects: 1, lag_p99_ms: 120.4, gaps: 0 }, backup: { ok: true, at: "2026-11-24T03:00:00+00:00", path: "x", error: null } });
    expect(v.ws).toContain("p99 задержки 120 мс");
    expect(v.backup).toBe("последняя копия 2026-11-24 03:00 UTC");
  });
});
