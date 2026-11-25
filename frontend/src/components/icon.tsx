// Иконки интерфейса (SVG).

import { type ReactNode } from "react";

// ── Иконки ────────────────────────────────────────────────────────────────────

export const svg = (d: ReactNode, size = 18) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {d}
  </svg>
);

export const Icon = {
  // Иконки управления списками: одна обводка 1.8, скруглённые концы, как у остальных.
  grip: (
    <svg width="14" height="18" viewBox="0 0 14 18" aria-hidden="true">
      {[3, 9, 15].map((y) => [4, 10].map((x) => <circle key={`${x}-${y}`} cx={x} cy={y} r="1.5" fill="currentColor" />))}
    </svg>
  ),
  dots: (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
      {[3, 8, 13].map((x) => (
        <circle key={x} cx={x} cy="8" r="1.5" fill="currentColor" />
      ))}
    </svg>
  ),
  up: (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M4 10l4-4 4 4" />
    </svg>
  ),
  down: (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M4 6l4 4 4-4" />
    </svg>
  ),
  star: (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" aria-hidden="true">
      <path d="M8 1.8l1.9 3.9 4.3.6-3.1 3 .7 4.3L8 11.5l-3.8 2.1.7-4.3-3.1-3 4.3-.6z" />
    </svg>
  ),
  reserve: (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M2.5 6.5h13v8a1 1 0 0 1-1 1h-11a1 1 0 0 1-1-1zM1.8 3.5h14.4v3H1.8zM7 10.5h4" />
    </svg>
  ),
  logo: (
    <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 1.5 21.5 7v10L12 22.5 2.5 17V7z" style={{ fill: "var(--brand)" }} />
      <path d="M9 7h4.2a2.6 2.6 0 0 1 1.4 4.8A2.8 2.8 0 0 1 13.4 17H9zm2 2v2.8h2a1.4 1.4 0 0 0 0-2.8zm0 4.6V15h2.3a1.45 1.45 0 0 0 0-2.9H11z" style={{ fill: "var(--on-brand)" }} />
    </svg>
  ),
  terminal: svg(<polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />, 20),
  bots: svg(
    <>
      <rect x="4" y="4" width="16" height="16" rx="2" />
      <rect x="9" y="9" width="6" height="6" />
      <path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3" />
    </>,
    20,
  ),
  settings: svg(<path d="M21 4h-7M10 4H3M21 12h-9M8 12H3M21 20h-5M12 20H3M14 2v4M8 10v4M16 18v4" />, 20),
  learn: svg(
    <>
      <path d="M22 10 12 5 2 10l10 5 10-5z" />
      <path d="M6 12v5c3 2 9 2 12 0v-5" />
    </>,
    20,
  ),
  journal: svg(<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01" />, 20),
  search: svg(
    <>
      <circle cx="11" cy="11" r="7" />
      <path d="m20 20-3.5-3.5" />
    </>,
    16,
  ),
  panic: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2" />
      <path d="M12 8v4M12 16h.01" />
    </svg>
  ),
  shield: svg(
    <>
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      <path d="M12 8v4M12 16h.01" />
    </>,
  ),
  offline: svg(
    <path d="M2 2l20 20M8.5 16.5a5 5 0 0 1 7 0M5 13a10 10 0 0 1 5.17-2.8M19 13a10 10 0 0 0-2.3-1.6M2 8.8a15 15 0 0 1 4.2-2.7M22 8.8A15 15 0 0 0 10.7 5M12 20h.01" />,
  ),
  warn: svg(<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0zM12 9v4M12 17h.01" />, 16),
  lock: svg(
    <>
      <rect x="3" y="11" width="18" height="11" rx="2" />
      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </>,
    16,
  ),
  eye: svg(
    <>
      <path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z" />
      <circle cx="12" cy="12" r="3" />
    </>,
    16,
  ),
  check: (
    <svg className="ck" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M20 6 9 17l-5-5" />
    </svg>
  ),
};
