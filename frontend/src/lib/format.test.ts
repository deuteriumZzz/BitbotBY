import { describe, expect, it } from "vitest";

import { plainReason } from "./format";

describe("plainReason", () => {
  it("убирает служебные теги, оставляет обычные скобки", () => {
    expect(plainReason("EMA 9 пересекла EMA 21. [SAC 64%, regime=trending_up]")).toBe("EMA 9 пересекла EMA 21.");
    expect(plainReason("Пробой [4h] на объёме")).toBe("Пробой [4h] на объёме");
  });
});
