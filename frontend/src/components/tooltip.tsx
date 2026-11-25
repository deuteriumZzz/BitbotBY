// Подсказки: общий слой всплывающих подсказок и значок «?» с пояснением.

import { useEffect, useRef, useState, type ReactNode } from "react";

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
