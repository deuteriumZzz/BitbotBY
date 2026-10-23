/// <reference types="vite/client" />
// У каждой настройки одно место, где её меняют (дома — lib/where.ts). «Все параметры»
// правят всё по определению и не считаются; пресет риска ставит набор значений целиком.

import { describe, expect, it } from "vitest";

import { OWN_HOME } from "../lib/where";
import { RULES } from "./rules";

const SOURCES = import.meta.glob("./*.tsx", { query: "?raw", import: "default", eager: true }) as Record<string, string>;

// Ярлыки «Кто решает»: мастер первого запуска и «Включить AI» после ввода ключа.
const SHORTCUTS: Record<string, string[]> = { MODE: ["./ai.tsx", "./setup.tsx"] };

function writers(key: string): string[] {
  const files = new Set<string>();
  for (const [file, text] of Object.entries(SOURCES)) {
    if (new RegExp(`patchConfig\\(\\{[^}]*\\b${key}\\b`).test(text) || new RegExp(`useDebouncedPatch\\([^;]*"${key}"`).test(text)) files.add(file);
  }
  if (RULES.some((r) => (r.key === key && !r.elsewhere) || r.on === key)) files.add("./rules.tsx");
  return [...files].filter((f) => !(SHORTCUTS[key] ?? []).includes(f)).sort();
}

describe("у каждой настройки один дом", () => {
  const keys = new Set([...Object.keys(OWN_HOME), ...RULES.flatMap((r) => [r.key, r.on].filter((k): k is string => Boolean(k)))]);
  for (const key of keys) {
    it(key, () => {
      expect(writers(key).length, `${key} меняется в: ${writers(key).join(", ")}`).toBeLessThanOrEqual(1);
    });
  }

  it("сделок одновременно, риск и плечо — в «Как рискуем», просадка и защиты — в «Правилах»", () => {
    for (const k of ["MAX_POSITIONS", "RISK_PER_TRADE", "LEVERAGE_MODE"]) expect(writers(k)).toEqual(["./bots.tsx"]);
    for (const k of ["DRAWDOWN_SCALE_ENABLED", "PAPER_GUARDS_OFF"]) expect(writers(k)).toEqual(["./rules.tsx"]);
  });

  it("дома ведут в существующие места", () => {
    for (const h of Object.values(OWN_HOME)) expect(h.pane === undefined || ["main", "conn", "all"].includes(h.pane)).toBe(true);
  });
});
