// Анимация чисел и шкал, вспышка изменения, скелетоны загрузки.

import { useEffect, useRef, useState } from "react";

// ── Анимированные числа ───────────────────────────────────────────────────────

const DIGITS = [..."0123456789"];

/** Механический счётчик: каждая цифра — барабан, который проворачивается к новому значению.
 *  Разряды считаются справа, чтобы при смене длины числа барабаны не перескакивали. */
export function Tween({ value, fmt, className }: { value: number; fmt: (v: number) => string; className?: string }) {
  const text = fmt(value);
  const chars = [...text];
  return (
    <span className={`odo ${className ?? ""}`}>
      <span className="sr">{text}</span>
      {chars.map((ch, i) => {
        const k = chars.length - i;
        return /\d/.test(ch) ? (
          <span key={`d${k}`} className="odo-d" aria-hidden="true">
            <span className="odo-s" style={{ transform: `translateY(${-Number(ch) * 10}%)` }}>
              {DIGITS.map((d) => (
                <i key={d}>{d}</i>
              ))}
            </span>
          </span>
        ) : (
          <span key={`c${k}`} className="odo-c" aria-hidden="true">
            {ch}
          </span>
        );
      })}
    </span>
  );
}

/** Стрелочный индикатор 0…1: шкала с делениями, стрелка доводится без отскока.
 *  При появлении стрелка выходит из нуля — как прибор при включении. */
export function Gauge({ value, label }: { value: number; label: string }) {
  const v = Math.max(0, Math.min(1, value));
  const [shown, setShown] = useState(0);
  useEffect(() => {
    const raf = requestAnimationFrame(() => setShown(v));
    return () => cancelAnimationFrame(raf);
  }, [v]);
  const tick = (i: number) => {
    const a = ((-180 + 18 * i) * Math.PI) / 180;
    const r0 = i % 5 ? 13.5 : 11.5;
    return (
      <line key={i} x1={20 + r0 * Math.cos(a)} y1={21 + r0 * Math.sin(a)} x2={20 + 16 * Math.cos(a)} y2={21 + 16 * Math.sin(a)} className={i >= 7 ? "g-t hi" : "g-t"} />
    );
  };
  return (
    <svg className="gauge" viewBox="0 0 40 24" width="40" height="24" role="img" aria-label={`${label}: ${v.toFixed(2)}`}>
      {Array.from({ length: 11 }, (_, i) => tick(i))}
      <line className="g-n" x1="20" y1="21" x2="20" y2="7" style={{ transform: `rotate(${-90 + 180 * shown}deg)` }} />
      <circle cx="20" cy="21" r="2" className="g-c" />
    </svg>
  );
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

/** Скелетон загрузки: прибор «прогревается» — строки-заготовки и бегущая строка развёртки. */
export const Skel = ({ rows = 3, label }: { rows?: number; label: string }) => (
  <div className="skel" role="status" aria-label={label}>
    {Array.from({ length: rows }, (_, i) => (
      <i key={i} />
    ))}
  </div>
);
