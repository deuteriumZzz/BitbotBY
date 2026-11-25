// Табло честности одной строкой: вердикт словами, без цифр, пока данных мало.

import { describe, expect, it } from "vitest";

import { scoreLine } from "./ai";

describe("scoreLine", () => {
  it("мало данных — без вердикта", () => {
    expect(scoreLine({ n: 141, hit_rate: 0.52, brier: 0.24, brier_base: 0.25, skill: 0.02, verdict: "мало данных" }, 200)).toBe(
      "проверено 141 из 200 — пока рано судить",
    );
  });
  it("вердикт и точность против 50/50", () => {
    expect(scoreLine({ n: 312, hit_rate: 0.49, brier: 0.252, brier_base: 0.2499, skill: -0.008, verdict: "не лучше монетки" }, 200)).toBe(
      "не лучше монетки · 312 проверок · исходов вверх 49% · точность против 50/50: -1%",
    );
  });
});
