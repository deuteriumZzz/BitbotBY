// Полосы под шагами «Цепочки» — сводка того, что сейчас работает в шаге. Включается и выключается
// всё в панели шага: нажатие на полосу её открывает. Перетаскивание осталось только у AI — порядок
// провайдеров в полосе и есть порядок очереди (то же, что в схеме AI), с «Отменить».

import { useMemo, useState } from "react";

import { actions, useStore } from "../store";
import { cfg, stratName } from "./bots";
import { STATUS } from "./chainops";
import { RULES } from "./rules";

export type Step = "strat" | "ai" | "risk" | "track";
const STEPS: readonly Step[] = ["strat", "ai", "risk", "track"];
export const isStep = (id: string): id is Step => (STEPS as readonly string[]).includes(id);

export interface Item {
  id: string;
  title: string;
  /** Подсказка при наведении. */
  sub: string;
  /** Метка справа: «эксп.», «всегда». */
  tag?: string;
  /** Точка статуса (провайдер AI): ok | warn | bad | "" — ещё не отвечал. */
  dot?: string;
}

const pct = (v: number) => `${Math.round(v * 100)}%`;
const EMPTY: Record<Step, string> = { strat: "ни одной стратегии", ai: "нет провайдеров", risk: "защиты выключены", track: "" };

/** Что сейчас включено в каждом шаге — из тех же данных, что и панели шагов. */
export function useLanes(): Record<Step, Item[]> {
  const strategies = useStore((s) => s.strategies);
  const ai = useStore((s) => s.ai);
  const config = useStore((s) => s.config);
  return useMemo(() => {
    const strat = strategies
      .filter((s) => s.enabled)
      .map((s) => ({
        id: `strat:${s.name}`,
        title: stratName(s.name),
        sub: s.live ? `живые ${pct(s.live.win_rate)} · ${s.live.trades}` : s.backtest ? `бэктест ${pct(s.backtest.win_rate)}` : "без статистики",
        tag: s.experimental ? "эксп." : undefined,
      }));
    const order = ai?.chain.map((c) => c.provider) ?? [];
    const aiItems = order.map((id, i) => {
      const p = ai?.providers.find((x) => x.id === id);
      const [dot, said] = STATUS[p?.status ?? "unknown"] ?? STATUS.unknown;
      return { id, title: ai?.single ? p?.name ?? id : `${i + 1} · ${p?.name ?? id}`, sub: `${i + 1} в очереди · ${said}`, dot };
    });
    // Защиты: те же правила, что выполняет бот; выключаемые — тумблером или нулём в пороге.
    const risk = RULES.filter((r) => {
      if (!r.key && !r.on) return false;
      if (r.on) return cfg<boolean>(config, r.on, true) !== false;
      return r.zeroOff && r.key ? Number(cfg(config, r.key, 0)) > 0 : true;
    }).map((r) => ({ id: `rule:${r.id}`, title: r.ifm, sub: r.cat, tag: r.on || r.zeroOff ? undefined : "всегда" }));
    const track: Item[] = [
      ...(cfg<boolean>(config, "PARTIAL_TP_ENABLED", true) !== false
        ? [{ id: "exit:ptp", title: "Частичная фиксация", sub: `${pct(cfg(config, "PARTIAL_TP_FRACTION", 0.5))} на ${pct(cfg(config, "PARTIAL_TP_TRIGGER", 0.6))} пути` }]
        : []),
      { id: "exit:trail", title: "Трейлинг-стоп", sub: `${cfg(config, "TRAILING_STOP_ATR_MULT", 1)} × ATR`, tag: "всегда" },
    ];
    return { strat, ai: aiItems, risk, track };
  }, [strategies, ai, config]);
}

/** Полоса под шагом: что в нём работает. Нажатие — панель шага; у AI чипы перетаскиваются — это порядок очереди. */
export function Lane({ step, items, onOpen }: { step: Step; items: Item[]; onOpen: () => void }) {
  const ai = useStore((s) => s.ai);
  const [drag, setDrag] = useState<string | null>(null);
  const [over, setOver] = useState<string | null>(null);
  const cap = step === "strat" ? 5 : 6;
  const reorder = step === "ai" && Boolean(ai) && !ai?.single && items.length > 1;
  // Брошенный провайдер встаёт на место того, на кого его бросили.
  const drop = (target: string) => {
    if (!ai || !drag || drag === target) return;
    const order = ai.chain.map((c) => c.provider);
    const disabled = ai.providers.filter((p) => p.disabled).map((p) => p.id);
    const next = order.filter((id) => id !== drag);
    next.splice(order.indexOf(target), 0, drag);
    void actions.putAiChain({ disabled, order: next }).then(() => useStore.getState().sayUndo("Порядок AI изменён", () => void actions.putAiChain({ disabled, order })));
  };
  const end = () => (setDrag(null), setOver(null));
  return (
    <div className={`lane2 sum${drag ? " sorting" : ""}`} onClick={onOpen} title="Открыть настройки шага">
      {items.slice(0, cap).map((b) => (
        <span
          key={b.id}
          className={`chip2${reorder ? " grab" : ""}${drag === b.id ? " lifted" : ""}${drag && over === b.id && drag !== b.id ? " over" : ""}`}
          title={reorder ? `${b.sub} · перетащите, чтобы поменять очередь` : b.sub}
          draggable={reorder || undefined}
          onDragStart={(e) => {
            e.dataTransfer.setData("text/plain", b.id);
            e.dataTransfer.effectAllowed = "move";
            setDrag(b.id);
          }}
          onDragEnd={end}
          onDragOver={(e) => {
            if (!drag) return;
            e.preventDefault();
            setOver(b.id);
          }}
          onDrop={(e) => {
            e.preventDefault();
            drop(b.id);
            end();
          }}
        >
          {b.dot !== undefined && <span className={`dot ${b.dot}`} aria-hidden="true" />}
          <span className="chip2-t">{b.title}</span>
          {b.tag && <i className="chip2-lock">{b.tag}</i>}
        </span>
      ))}
      {items.length > cap && <span className="chip2 more">ещё {items.length - cap}</span>}
      {!items.length && EMPTY[step] && <span className="lane2-empty">{EMPTY[step]}</span>}
    </div>
  );
}
