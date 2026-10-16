// Числа в стиле терминала: узкий неразрывный пробел между разрядами, точка в дробной части.

export const fmtN = (n: number, d = 2): string =>
  Math.abs(n)
    .toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d })
    .replace(/,/g, " ");

export const sign = (n: number, withPlus = true): string =>
  n < 0 ? "−" : withPlus && n > 0 ? "+" : "";

export const usd = (n: number, withPlus = true): string => `${sign(n, withPlus)}$${fmtN(n)}`;

/** Знаков после точки по величине цены: 64210.00, 182.40, 3.842, 0.00001020. */
export const priceDecimals = (p: number): number =>
  p > 0 ? Math.min(10, Math.max(2, 3 - Math.floor(Math.log10(p)))) : 2;

export const price = (p: number): string => fmtN(p, priceDecimals(p));

export const pct = (n: number, d = 2): string => `${sign(n)}${fmtN(n, d)}%`;

export const arrow = (n: number): string => (n > 0 ? "▲" : n < 0 ? "▼" : "•");

export const tone = (n: number): "up" | "down" | "" => (n > 0 ? "up" : n < 0 ? "down" : "");

/** Русское множественное число: plural(3, ["сделка", "сделки", "сделок"]). */
export function plural(n: number, forms: [string, string, string]): string {
  const a = Math.abs(n) % 100;
  const b = a % 10;
  if (a > 10 && a < 20) return forms[2];
  if (b > 1 && b < 5) return forms[1];
  if (b === 1) return forms[0];
  return forms[2];
}

export const hhmm = (d: Date = new Date()): string => d.toTimeString().slice(0, 5);

export const base = (symbol: string): string => symbol.split("/")[0];
