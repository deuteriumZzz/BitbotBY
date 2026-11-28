// Фирменные значки сервисов: Bybit — из @web3icons (как монеты), Telegram, Reddit и X — логотипы Phosphor в цветах бренда.

import type { ReactNode } from "react";
import { RedditLogo, TelegramLogo, XLogo } from "@phosphor-icons/react";

const SVG = import.meta.glob<string>("/node_modules/@web3icons/core/dist/svgs/exchanges/branded/bybit.svg.js", { import: "default", eager: true });
const BYBIT = Object.values(SVG)[0] ?? "";

const LOGO: Record<string, ReactNode> = {
  // SVG из собственной сборки приложения, не из сети.
  bybit: <span className="brand-svg" dangerouslySetInnerHTML={{ __html: BYBIT }} />,
  telegram: <TelegramLogo weight="fill" color="#2AABEE" />,
  telethon: <TelegramLogo weight="fill" color="#2AABEE" />,
  reddit: <RedditLogo weight="fill" color="#FF4500" />,
  twitter: <XLogo weight="bold" />,
};

/** Логотип сервиса или null, если своего нет (тогда берут обычную иконку). */
export const brandLogo = (id: string): ReactNode | null => LOGO[id] ?? null;

// Логотипы AI-провайдеров из @lobehub/icons-static-svg (MIT): цветные, где они есть; одноцветные — в цвете бренда.
const AI_SVG = import.meta.glob<string>(
  [
    "/node_modules/@lobehub/icons-static-svg/icons/groq.svg",
    "/node_modules/@lobehub/icons-static-svg/icons/gemini-color.svg",
    "/node_modules/@lobehub/icons-static-svg/icons/deepseek-color.svg",
    "/node_modules/@lobehub/icons-static-svg/icons/openai.svg",
    "/node_modules/@lobehub/icons-static-svg/icons/claude-color.svg",
    "/node_modules/@lobehub/icons-static-svg/icons/openrouter.svg",
  ],
  { query: "?raw", import: "default", eager: true },
);
const aiFile = (f: string) => AI_SVG[`/node_modules/@lobehub/icons-static-svg/icons/${f}.svg`] ?? "";
const AI: Record<string, [string, string?]> = {
  groq: [aiFile("groq"), "#F55036"],
  gemini: [aiFile("gemini-color")],
  deepseek: [aiFile("deepseek-color")],
  openai: [aiFile("openai")],
  anthropic: [aiFile("claude-color")],
  openrouter: [aiFile("openrouter")],
};

/** Логотип AI-провайдера; null — своего нет (тогда буквы). SVG из собственной сборки, не из сети. */
export function aiLogo(id: string, size = 20): ReactNode | null {
  const [svg, color] = AI[id] ?? [""];
  if (!svg) return null;
  return <span className="brand-svg ai" style={{ width: size, height: size, color }} dangerouslySetInnerHTML={{ __html: svg }} />;
}
