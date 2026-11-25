// Переключатели: Toggle, сегменты Seg и двухшаговое подтверждение опасных действий (useArm).

import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";

import { haptic } from "../lib/platform";

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
      onClick={() => {
        haptic("tap");
        props.onChange(!props.on);
      }}
    >
      <span className="tr">
        <span className="kn" />
      </span>
      {props.children}
    </button>
  );
}

/** Сегменты с бегунком: подсветка выбранного едет к новому значению, как каретка шкалы. */
export function Seg<T extends string | number>(props: {
  value: T | null | undefined;
  options: readonly (readonly [T, string])[];
  onChange: (v: T) => void;
  className?: string;
  label?: string;
  disabled?: boolean;
}) {
  const host = useRef<HTMLDivElement>(null);
  const [ind, setInd] = useState<CSSProperties | null>(null);
  const labels = props.options.map(([, t]) => t).join("|");
  useLayoutEffect(() => {
    const el = host.current;
    if (!el) return;
    const move = () => {
      const b = el.querySelector<HTMLElement>('button[aria-pressed="true"]');
      setInd(b ? { width: b.offsetWidth, height: b.offsetHeight, transform: `translate(${b.offsetLeft}px,${b.offsetTop}px)` } : null);
    };
    move();
    const ro = new ResizeObserver(move);
    ro.observe(el);
    return () => ro.disconnect();
  }, [props.value, labels]);
  return (
    <div ref={host} className={`seg ${ind ? "has-ind" : ""} ${props.className ?? ""}`} role="group" aria-label={props.label}>
      {ind && <span className="seg-ind" style={ind} aria-hidden="true" />}
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
