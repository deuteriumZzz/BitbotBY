// Значок монеты: официальная иконка из @web3icons (MIT), а если её нет — тикер в кружке.
// Иконки лежат в сборке (без запросов к чужим сайтам) и грузятся по одной, только когда монета на экране.

import { useEffect, useState, type CSSProperties } from "react";

import { base } from "../lib/format";

const DIR = "/node_modules/@web3icons/core/dist/svgs/tokens/branded";
const ICONS = import.meta.glob<string>("/node_modules/@web3icons/core/dist/svgs/tokens/branded/*.svg.js", { import: "default" });

/** null — иконки нет, рисуем тикер; строка — SVG. */
const cache = new Map<string, string | null>();

export function Coin({ symbol, className = "", style }: { symbol: string; className?: string; style?: CSSProperties }) {
  const ticker = base(symbol).toUpperCase();
  const [svg, setSvg] = useState<string | null | undefined>(() => cache.get(ticker));
  useEffect(() => {
    if (cache.has(ticker)) return setSvg(cache.get(ticker));
    const load = ICONS[`${DIR}/${ticker}.svg.js`];
    if (!load) {
      cache.set(ticker, null);
      return setSvg(null);
    }
    let alive = true;
    load()
      .then((s) => cache.set(ticker, s))
      .catch(() => cache.set(ticker, null))
      .finally(() => alive && setSvg(cache.get(ticker) ?? null));
    return () => {
      alive = false;
    };
  }, [ticker]);
  return svg ? (
    // SVG из собственной сборки приложения, не от пользователя и не из сети.
    <span className={`coin img ${className}`} style={style} role="img" aria-label={ticker} dangerouslySetInnerHTML={{ __html: svg }} />
  ) : (
    // Длинный тикер в кружок не помещается — первые три буквы, полное имя в подсказке.
    <span className={`coin ${className}`} style={style} title={ticker}>
      {ticker.length > 4 ? ticker.slice(0, 3) : ticker}
    </span>
  );
}
