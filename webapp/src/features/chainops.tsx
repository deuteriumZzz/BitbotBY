// Общие части цепочки AI: действия, статусы, меню. На них опираются и список, и схема.

import { useEffect, useRef, useState } from "react";

import type { AiProvider, AiState } from "../lib/types";
import { actions, useStore } from "../store";

/** Провайдер, который решает в режиме «Один провайдер». */
export const chosenId = (ai: AiState) => (ai.pinned !== "auto" ? ai.pinned : (ai.chain[0]?.provider ?? ""));

export const STATUS: Record<string, [string, string]> = {
  ok: ["ok", "работает"],
  rate_limit: ["warn", "лимит"],
  no_balance: ["bad", "нет денег"],
  bad_key: ["bad", "ключ не принят"],
  bad_model: ["bad", "нет такой модели"],
  bad_format: ["warn", "ломает формат ответа"],
  bad_request: ["warn", "отклонил запрос"],
  error: ["warn", "нет связи"],
  unknown: ["", "ещё не отвечал"],
};

export const hhmm = (epoch: number) =>
  new Date(epoch * 1000).toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" });

/** Закрепить провайдера (AI_PROVIDER, как /provider в Telegram) и сразу обновить раздел. */
export async function pin(id: string): Promise<void> {
  if (await actions.patchConfig({ AI_PROVIDER: id })) await useStore.getState().loadAi().catch(() => undefined);
}

export const applyOrder = (all: string[], subset: string[], next: string[]): string[] => {
  const inSet = new Set(subset);
  let k = 0;
  return all.map((id) => (inSet.has(id) ? next[k++] : id));
};

export interface Probe {
  busy?: boolean;
  ok?: boolean;
  ms?: number;
  message?: string;
}
export interface CtxItem {
  label: string;
  hint?: string;
  danger?: boolean;
  disabled?: boolean;
  run: () => void;
}

/** Контекстное меню: правый щелчок, долгое нажатие или кнопка «⋯»; стрелки, Enter, Esc. */
export function ChainMenu({ items, x, y, onClose }: { items: CtxItem[]; x: number; y: number; onClose: () => void }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    ref.current?.querySelector<HTMLElement>("button:not(:disabled)")?.focus();
    const away = (e: PointerEvent) => {
      if (!ref.current?.contains(e.target as Node)) onClose();
    };
    const key = (e: KeyboardEvent) => {
      const bs = Array.from(ref.current?.querySelectorAll<HTMLElement>("button:not(:disabled)") ?? []);
      const i = bs.indexOf(document.activeElement as HTMLElement);
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        const step = e.key === "ArrowDown" ? 1 : -1;
        bs[(i + step + bs.length) % bs.length]?.focus();
      }
    };
    window.addEventListener("pointerdown", away, true);
    window.addEventListener("keydown", key);
    return () => {
      window.removeEventListener("pointerdown", away, true);
      window.removeEventListener("keydown", key);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const style = { left: Math.max(8, Math.min(x, window.innerWidth - 248)), top: Math.max(8, Math.min(y, window.innerHeight - items.length * 52 - 16)) };
  return (
    <div ref={ref} className="ctx" role="menu" style={style}>
      {items.map((it) => (
        <button
          key={it.label}
          type="button"
          role="menuitem"
          className={`ctx-i${it.danger ? " danger" : ""}`}
          disabled={it.disabled}
          onClick={() => {
            onClose();
            it.run();
          }}
        >
          <span>{it.label}</span>
          {it.hint && <small>{it.hint}</small>}
        </button>
      ))}
    </div>
  );
}


/** Все действия над цепочкой — одни и те же в списке и на схеме. */
export function useChainOps(ai: AiState, list: AiProvider[]) {
  const say = useStore((s) => s.say);
  const sayUndo = useStore((s) => s.sayUndo);
  const active = list.filter((p) => !p.disabled);
  const reserve = list.filter((p) => p.disabled);
  const ids = active.map((p) => p.id);
  const disabledIds = reserve.map((p) => p.id);
  const [probes, setProbes] = useState<Record<string, Probe>>({});
  const [heir, setHeir] = useState<string | null>(null);
  const nameOf = (id: string) => ai.providers.find((p) => p.id === id)?.name ?? id;

  const commitOrder = (next: string[]) => actions.putAiChain({ order: applyOrder(ai.providers.map((x) => x.id), ids, next) });
  const setDisabled = (id: string, off: boolean) =>
    actions.putAiChain({ disabled: off ? [...new Set([...disabledIds, id])] : disabledIds.filter((x) => x !== id) });
  const toReserve = async (p: AiProvider) => {
    if (await setDisabled(p.id, true)) sayUndo(`${p.name} в резерве: бот его не спрашивает, ключ сохранён`, () => void setDisabled(p.id, false));
  };
  const probe = async (p: AiProvider) => {
    setProbes((s) => ({ ...s, [p.id]: { busy: true } }));
    const r = await actions.aiProbe(p.id);
    setProbes((s) => ({ ...s, [p.id]: { ok: r.ok, ms: r.ms, message: r.message } }));
    setTimeout(() => setProbes((s) => Object.fromEntries(Object.entries(s).filter(([k]) => k !== p.id))), 12000);
  };
  const showHeir = (p: AiProvider) => {
    if (ai.single) return say("В режиме «Один провайдер» запасных нет: если он упадёт, цикл пойдёт на локальных стратегиях.");
    const chain = ai.chain.map((c) => c.provider);
    const i = chain.indexOf(p.id);
    if (i < 0) return say(`${p.name} и так не участвует в цепочке.`);
    const next = chain[i + 1];
    if (!next) return say(`После ${p.name} запасных нет: цикл пойдёт на локальных стратегиях.`);
    setHeir(next);
    say(`Если ${p.name} упадёт, ответит ${nameOf(next)}.`);
    setTimeout(() => setHeir(null), 2600);
  };
  const removeKey = (p: AiProvider) => {
    const t = setTimeout(() => void actions.deleteSecret(p.id), 8000);
    sayUndo(`Ключ ${p.name} будет удалён`, () => clearTimeout(t));
  };
  const items = (p: AiProvider): CtxItem[] => {
    const first = p.id === (ai.single ? chosenId(ai) : ai.pinned);
    const out: CtxItem[] = [];
    if (!first) out.push({ label: ai.single ? "Выбрать" : "Сделать первым", hint: "бот спросит его раньше остальных", run: () => void pin(p.id) });
    out.push({ label: "Проверить связь", hint: "лимит не тратится", run: () => void probe(p) });
    if (!ai.single) out.push({ label: "Если он упадёт…", hint: "покажет, кто его заменит", run: () => showHeir(p) });
    out.push(
      p.disabled
        ? { label: "Вернуть в цепочку", run: () => void setDisabled(p.id, false) }
        : { label: "В резерв", hint: "ключ останется, бот не спросит", run: () => void toReserve(p) },
    );
    if (p.key_set) out.push({ label: "Удалить ключ…", hint: "через 8 секунд, можно отменить", danger: true, run: () => removeKey(p) });
    return out;
  };


  return { active, reserve, ids, disabledIds, probes, heir, nameOf, commitOrder, setDisabled, toReserve, probe, showHeir, removeKey, items };
}

/** Значок провайдера: монограмма в собственном приглушённом цвете (чужие логотипы не используем). */
export function Avatar({ id, name }: { id: string; name: string }) {
  const h = [...id].reduce((acc, c) => (acc * 31 + c.charCodeAt(0)) % 360, 7);
  return (
    <span className="pav" style={{ background: `hsl(${h} 30% 22%)`, color: `hsl(${h} 65% 78%)`, borderColor: `hsl(${h} 32% 36%)` }} aria-hidden="true">
      {name.replace(/[^A-Za-zА-Яа-я0-9]/g, "").slice(0, 2)}
    </span>
  );
}

