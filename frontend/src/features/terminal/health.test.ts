// Главный экран честно говорит, когда AI не отвечает и как прошла сверка с биржей.

import { describe, expect, it } from "vitest";

import type { AiProvider, AiState } from "../../lib/types";
import { aiDown } from "./index";
import { reconText } from "./positions";

const prov = (id: string, status: string, cooldown_until: number | null = null) =>
  ({ id, name: id.toUpperCase(), status, cooldown_until }) as AiProvider;
const state = (...providers: AiProvider[]): AiState =>
  ({ chain: providers.map((p) => ({ provider: p.id, model: "" })), providers }) as unknown as AiState;

describe("aiDown", () => {
  const now = 1_000_000_000;
  it("пусто, пока хоть одно звено может ответить", () => {
    expect(aiDown(state(prov("a", "no_balance"), prov("b", "ok")), now)).toBe("");
    expect(aiDown(state(prov("a", "unknown")), now)).toBe("");
  });
  it("все звенья недоступны — перечисляет причины", () => {
    const s = state(prov("deepseek", "no_balance"), prov("gemini", "ok", now / 1000 + 60));
    expect(aiDown(s, now)).toBe("DEEPSEEK — нет денег, GEMINI — пауза");
  });
  it("без цепочки — не тревога (это «нет ключа AI», другое сообщение)", () => {
    expect(aiDown(state(), now)).toBe("");
  });
});

describe("reconText", () => {
  const now = 10_000;
  it("сверено — зелёный и давность", () => {
    expect(reconText({ at: now - 120, problems: [] }, now)).toEqual(["ok", "Сверено с биржей 2 мин назад"]);
  });
  it("расхождение важнее ожидающих ордеров", () => {
    const [cls, text] = reconText({ at: now, problems: ["untracked:DOGE/USDT:USDT"], pending_orders: 1 }, now);
    expect(cls).toBe("bad");
    expect(text).toContain("DOGE — есть на бирже, бот её не ведёт");
  });
  it("ордер без ответа биржи", () => {
    expect(reconText({ at: now, problems: [], pending_orders: 2 }, now)).toEqual(["warn", "2 ордера отправлены, ждём ответа биржи"]);
  });
});
