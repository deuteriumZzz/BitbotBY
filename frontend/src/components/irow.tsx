// Строка настройки с иконкой: значок · название (и короткая подпись) · элемент управления справа.
// Пояснение — в «?», а не текстом под строкой.

import type { ReactNode } from "react";

import { Q } from "./tooltip";

export function IRow(props: { icon: ReactNode; title: ReactNode; sub?: ReactNode; tip?: string; off?: boolean; tone?: "ok" | "warn" | "bad"; children?: ReactNode }) {
  return (
    <div className={`irow${props.off ? " off" : ""}${props.tone ? ` ${props.tone}` : ""}`}>
      <span className="irow-ico" aria-hidden="true">
        {props.icon}
      </span>
      <div className="irow-t">
        <b>
          {props.title}
          {props.tip && <Q tip={props.tip} />}
        </b>
        {props.sub && <small>{props.sub}</small>}
      </div>
      {props.children && <div className="irow-c">{props.children}</div>}
    </div>
  );
}

/** Выбор плитками: иконка, название, короткая подпись. Одно нажатие вместо списка. */
export function Tiles<T extends string>(props: {
  value: T | null;
  options: readonly (readonly [T, string, ReactNode, string?])[];
  label: string;
  onChange: (v: T) => void;
  /** Плитка, которую нельзя выбрать прямо сейчас: трясётся и краснеет; n — чтобы анимация проигрывалась на каждый клик. */
  shake?: { v: T; n: number } | null;
}) {
  return (
    <div className="tiles" role="group" aria-label={props.label} style={{ gridTemplateColumns: `repeat(${props.options.length},minmax(0,1fr))` }}>
      {props.options.map(([v, name, icon, sub]) => (
        <button
          key={props.shake?.v === v ? `${v}${props.shake.n}` : v}
          type="button"
          className={`tile${props.shake?.v === v ? " no" : ""}`}
          aria-pressed={props.value === v}
          onClick={() => props.value !== v && props.onChange(v)}
        >
          <span className="tile-ico" aria-hidden="true">
            {icon}
          </span>
          <b>{name}</b>
          {sub && <small>{sub}</small>}
        </button>
      ))}
    </div>
  );
}
