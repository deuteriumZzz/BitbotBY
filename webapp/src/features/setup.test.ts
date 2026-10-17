import { describe, expect, it } from "vitest";

import type { AiState } from "../lib/types";
import { readiness } from "./setup";

const ai = (n: number) => ({ chain: Array.from({ length: n }, () => ({ provider: "groq", model: "m" })) }) as unknown as AiState;
const byStep = (c: ReturnType<typeof readiness>) => Object.fromEntries(c.map((x) => [x.step, x]));

describe("готовность", () => {
  it("в Paper ключи биржи не обязательны, AI — нужен", () => {
    const c = byStep(readiness({ keys_in_app: true, bybit: false, telegram: false }, ai(0), "Умеренный", true));
    expect(c.bybit).toMatchObject({ ok: false, need: false });
    expect(c.ai).toMatchObject({ ok: false, need: true });
    expect(c.risk.text).toBe("Умеренный");
  });

  it("в Live без ключей биржи — обязательный пункт", () => {
    const c = byStep(readiness({ keys_in_app: false, bybit: false, telegram: true }, ai(2), "свои настройки", false));
    expect(c.bybit).toMatchObject({ ok: false, need: true });
    expect(c.telegram.ok).toBe(true);
    expect(c.ai).toMatchObject({ ok: true, text: "подключено провайдеров: 2" });
  });
});
