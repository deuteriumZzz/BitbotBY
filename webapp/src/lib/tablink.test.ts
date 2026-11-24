import { describe, expect, it } from "vitest";

import { tabFromUrl } from "./platform";

describe("tabFromUrl — вкладка из кнопки Telegram-уведомления", () => {
  it("открывает известную вкладку", () => {
    expect(tabFromUrl("?tab=learn")).toBe("learn");
    expect(tabFromUrl("?mock&tab=bots")).toBe("bots");
  });

  it("игнорирует пустое и чужое", () => {
    expect(tabFromUrl("")).toBeNull();
    expect(tabFromUrl("?tab=admin")).toBeNull();
    expect(tabFromUrl("?tab=")).toBeNull();
  });
});
