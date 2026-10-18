import { describe, expect, it } from "vitest";

import { detectProvider } from "./keys";

describe("detectProvider", () => {
  it("узнаёт провайдера по префиксу", () => {
    expect(detectProvider("gsk_abcdefghijklmnop")).toEqual(["groq"]);
    expect(detectProvider("  AIzaSyAbcdefghijkl ")).toEqual(["gemini"]);
    expect(detectProvider("sk-ant-api03-abcdefgh")).toEqual(["anthropic"]);
    expect(detectProvider("sk-or-v1-abcdefgh")).toEqual(["openrouter"]);
  });
  it("sk- без уточнения — два кандидата, а не угадывание", () => {
    expect(detectProvider("sk-abcdefghijklmnop")).toEqual(["openai", "deepseek"]);
  });
  it("короткое и незнакомое не определяется", () => {
    expect(detectProvider("gsk_")).toEqual([]);
    expect(detectProvider("0123456789abcdef0123456789abcdef")).toEqual([]);
  });
});
