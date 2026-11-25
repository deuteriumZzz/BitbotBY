// Определение провайдера по формату ключа: только префиксы, которые у провайдеров стабильны.
// Ключ никуда не отправляется — это чистая функция над строкой.

const PREFIXES: readonly (readonly [string, string])[] = [
  ["gsk_", "groq"],
  ["AIza", "gemini"],
  ["sk-ant-", "anthropic"],
  ["sk-or-", "openrouter"],
  ["nvapi-", "nvidia"],
  ["csk-", "cerebras"],
  ["hf_", "huggingface"],
  ["github_pat_", "github"],
  ["ghp_", "github"],
];

/** Провайдеры, у которых ключ с таким началом: один — уверенно, несколько — выбор, пусто — не узнали. */
export function detectProvider(raw: string): string[] {
  const key = raw.trim();
  if (key.length < 8) return [];
  const hit = PREFIXES.find(([p]) => key.startsWith(p));
  if (hit) return [hit[1]];
  // «sk-…» без уточнения у OpenAI и DeepSeek одинаковое начало.
  if (key.startsWith("sk-")) return ["openai", "deepseek"];
  return [];
}
