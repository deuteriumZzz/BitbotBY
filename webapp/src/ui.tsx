// Общие элементы интерфейса: переключатель, сегменты, подсказки, анимированные числа.

import { useEffect, useRef, useState, type ReactNode } from "react";

import { reducedMotion } from "./lib/platform";

// ── Подсказки «?» и всплывающие пояснения ─────────────────────────────────────

interface TipState {
  content: ReactNode;
  rect: { left: number; width: number; top: number; bottom: number };
}
let setTipGlobal: ((t: TipState | null) => void) | null = null;
let tipOwner: unknown = null;

export function showTip(content: ReactNode, rect: TipState["rect"], owner: unknown): void {
  tipOwner = owner;
  setTipGlobal?.({ content, rect });
}
export function hideTip(): void {
  tipOwner = null;
  setTipGlobal?.(null);
}

export function TipLayer() {
  const [tip, setTip] = useState<TipState | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState({ left: -9999, top: 0 });
  useEffect(() => {
    setTipGlobal = setTip;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && hideTip();
    const onDoc = (e: MouseEvent) => {
      const q = (e.target as Element).closest?.(".q") as HTMLElement | null;
      if (q?.dataset.tip) {
        if (tipOwner === q) hideTip();
        else showTip(q.dataset.tip, q.getBoundingClientRect(), q);
      } else if (tipOwner instanceof HTMLElement && tipOwner.classList.contains("q")) hideTip();
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("click", onDoc);
    return () => {
      setTipGlobal = null;
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("click", onDoc);
    };
  }, []);
  useEffect(() => {
    const el = ref.current;
    if (!tip || !el) return;
    const w = el.offsetWidth;
    const h = el.offsetHeight;
    const r = tip.rect;
    const left = Math.max(8, Math.min(r.left + r.width / 2 - w / 2, innerWidth - w - 8));
    const top = r.bottom + 8 + h > innerHeight - 8 ? r.top - h - 8 : r.bottom + 8;
    setPos({ left, top });
  }, [tip]);
  if (!tip) return null;
  return (
    <div className="tip" ref={ref} style={pos} role="tooltip">
      {tip.content}
    </div>
  );
}

/** Кнопка «?» с пояснением простыми словами. */
export const Q = ({ tip }: { tip: string }) => (
  <button className="q" type="button" data-tip={tip} aria-label="Пояснение">
    ?
  </button>
);

// ── Переключатель и сегменты ──────────────────────────────────────────────────

export function Toggle(props: {
  on: boolean;
  onChange: (on: boolean) => void;
  label: string;
  brand?: boolean;
  disabled?: boolean;
  children?: ReactNode;
  className?: string;
}) {
  return (
    <button
      className={`tgl${props.brand ? " brand" : ""} ${props.className ?? ""}`}
      type="button"
      role="switch"
      aria-checked={props.on}
      aria-label={props.label}
      disabled={props.disabled}
      onClick={() => props.onChange(!props.on)}
    >
      <span className="tr">
        <span className="kn" />
      </span>
      {props.children}
    </button>
  );
}

export function Seg<T extends string | number>(props: {
  value: T | null | undefined;
  options: readonly (readonly [T, string])[];
  onChange: (v: T) => void;
  className?: string;
  label?: string;
  disabled?: boolean;
}) {
  return (
    <div className={`seg ${props.className ?? ""}`} role="group" aria-label={props.label}>
      {props.options.map(([v, text]) => (
        <button
          key={String(v)}
          type="button"
          aria-pressed={v === props.value}
          disabled={props.disabled}
          onClick={() => v !== props.value && props.onChange(v)}
        >
          {text}
        </button>
      ))}
    </div>
  );
}

// ── Анимированные числа ───────────────────────────────────────────────────────

/** Число плавно «докручивается» до нового значения (ease-out cubic, 0.8 с). */
export function Tween({ value, fmt, className }: { value: number; fmt: (v: number) => string; className?: string }) {
  const [shown, setShown] = useState(value);
  const cur = useRef(value);
  useEffect(() => {
    const start = cur.current;
    if (reducedMotion() || Math.abs(start - value) < 1e-9) {
      cur.current = value;
      setShown(value);
      return;
    }
    const t0 = performance.now();
    let raf = 0;
    const step = (t: number) => {
      const k = Math.min(1, (t - t0) / 800);
      cur.current = start + (value - start) * (1 - (1 - k) ** 3);
      setShown(cur.current);
      if (k < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [value]);
  return <span className={className}>{fmt(shown)}</span>;
}

/** Значение вспыхивает зелёным/красным при изменении: смена key перезапускает CSS-анимацию. */
export function Flash({ value, text, className = "" }: { value: number; text: string; className?: string }) {
  const prev = useRef(value);
  const dir = value > prev.current ? "fl-up" : value < prev.current ? "fl-down" : "";
  useEffect(() => {
    prev.current = value;
  }, [value]);
  return (
    <b key={value} className={`${className} ${dir}`}>
      {text}
    </b>
  );
}

// ── Кнопка с подтверждением вторым нажатием ───────────────────────────────────

/** Первое нажатие «взводит» кнопку на 2.6 с, второе — выполняет. */
export function useArm(): [boolean, () => boolean] {
  const [armed, setArmed] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout>>();
  useEffect(() => () => clearTimeout(timer.current), []);
  const press = (): boolean => {
    clearTimeout(timer.current);
    if (armed) {
      setArmed(false);
      return true;
    }
    setArmed(true);
    timer.current = setTimeout(() => setArmed(false), 2600);
    return false;
  };
  return [armed, press];
}

// ── Иконки ────────────────────────────────────────────────────────────────────

const svg = (d: ReactNode, size = 18) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {d}
  </svg>
);

export const Icon = {
  logo: (
    <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 1.5 21.5 7v10L12 22.5 2.5 17V7z" fill="#7000FF" />
      <path d="M9 7h4.2a2.6 2.6 0 0 1 1.4 4.8A2.8 2.8 0 0 1 13.4 17H9zm2 2v2.8h2a1.4 1.4 0 0 0 0-2.8zm0 4.6V15h2.3a1.45 1.45 0 0 0 0-2.9H11z" fill="#fff" />
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
  journal: svg(<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01" />, 20),
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
