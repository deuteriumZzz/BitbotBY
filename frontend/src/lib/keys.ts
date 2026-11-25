// Определение провайдера по формату ключа: только префиксы, которые у провайдеров стабильны.
// Ключ никуда не отправляется — это чистая функция над строкой.

const PREFIXES: readonly (readonly [string, string])[] = [
  ["gsk_", "groq"],
  ["AIza", "gemini"],
  ["sk-ant-", "anthropic"],
];

/** Провайдеры, у которых ключ с таким началом: один — уверенно, несколько — выбор, пусто — не узнали. */
export function detectProvider(raw: string): string[] {
  const key = raw.trim();
  if (key.length < 8) return [];
  const hit = PREFIXES.find(([p]) => key.startsWith(p));
  if (hit) return [hit[1]];
  // Ключ OpenRouter («sk-or-…») похож на OpenAI, но такого провайдера у нас нет — не угадываем.
  if (key.startsWith("sk-or-")) return [];
  // «sk-…» без уточнения у OpenAI и DeepSeek одинаковое начало.
  if (key.startsWith("sk-")) return ["openai", "deepseek"];
  return [];
}
