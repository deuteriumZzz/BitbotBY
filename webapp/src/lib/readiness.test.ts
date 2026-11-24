import { describe, expect, it } from "vitest";
import { readinessReason } from "./format";

describe("readinessReason", () => {
  it("переводит известные коды причин режима", () => {
    expect(readinessReason("sac_model_missing")).toContain("модель SAC не загружена");
    expect(readinessReason("sac_not_approved")).toContain("не допущена реестром");
    expect(readinessReason("ai_unavailable")).toContain("AI недоступен");
  });

  it("неизвестный код показывает как есть, пустой — пустой строкой", () => {
    expect(readinessReason("brand_new_reason")).toBe("brand_new_reason");
    expect(readinessReason("")).toBe("");
    expect(readinessReason(null)).toBe("");
    expect(readinessReason(undefined)).toBe("");
  });
});
