// Полосы под шагами «Цепочки» — сводка того, что сейчас работает в шаге. Включается и выключается
// всё в панели шага: нажатие на полосу её открывает. Порядок AI-провайдеров выбирает бот.

import { useMemo } from "react";

import { useStore } from "../store";
import { cfg, stratName } from "./bots";
import { STATUS } from "./chainops";
import { RULES } from "./rules";

export type Step = "strat" | "ai" | "risk" | "track";
// У AI плашек нет: два ключа видны строкой на самом блоке.
const STEPS: readonly Step[] = ["strat", "risk", "track"];
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
      return { id, title: `${i + 1} · ${p?.name ?? id}`, sub: `${i + 1} в очереди · ${said}`, dot };
    });
    // Защиты: те же правила, что выполняет бот; выключаемые — тумблером или нулём в пороге.
    const risk = RULES.filter((r) => {
      if (r.cat === "Бот" || (!r.key && !r.on)) return false;
      if (r.on) return cfg<boolean>(config, r.on, true) !== false;
      return r.zeroOff && r.key ? Number(cfg(config, r.key, 0)) > 0 : true;
    }).map((r) => ({ id: `rule:${r.id}`, title: r.chip ?? r.ifm, sub: r.cat, tag: r.on || r.zeroOff ? undefined : "всегда" }));
    const track: Item[] = [
      ...(cfg<boolean>(config, "PARTIAL_TP_ENABLED", true) !== false
        ? [{ id: "exit:ptp", title: "Частичный выход", sub: `${pct(cfg(config, "PARTIAL_TP_FRACTION", 0.5))} на ${pct(cfg(config, "PARTIAL_TP_TRIGGER", 0.6))} пути` }]
        : []),
      { id: "exit:trail", title: "Трейлинг-стоп", sub: `${cfg(config, "TRAILING_STOP_ATR_MULT", 1)} × ATR`, tag: "всегда" },
    ];
    return { strat, ai: aiItems, risk, track };
  }, [strategies, ai, config]);
}

/** Полоса под шагом: что в нём работает. Нажатие — панель шага. */
export function Lane({ step, items, onOpen }: { step: Step; items: Item[]; onOpen: () => void }) {
  const cap = step === "strat" ? 5 : 6;
  return (
    <div className="lane2 sum" onClick={onOpen} title="Открыть настройки шага">
      {items.slice(0, cap).map((b) => (
        // «всегда» стоит почти на каждой защите и съедает место у названия — уходит в подсказку.
        <span key={b.id} className="chip2" title={b.tag === "всегда" ? `${b.sub} · работает всегда` : b.sub}>
          {b.dot !== undefined && <span className={`dot ${b.dot}`} aria-hidden="true" />}
          <span className="chip2-t">{b.title}</span>
          {b.tag && b.tag !== "всегда" && <i className="chip2-lock">{b.tag}</i>}
        </span>
      ))}
      {items.length > cap && <span className="chip2 more">ещё {items.length - cap}</span>}
      {!items.length && EMPTY[step] && <span className="lane2-empty">{EMPTY[step]}</span>}
    </div>
  );
}
