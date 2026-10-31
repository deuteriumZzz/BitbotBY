// Блоки цепочки — настройка перетаскиванием. Каждый блок — настоящая настройка бота:
// бросили на шаг — включилась, унесли обратно в библиотеку — выключилась; нажатие делает то же без перетаскивания.
// Всё отменяется в течение 8 секунд («Отменить» в тосте).

import { useMemo, useState } from "react";

import { actions, useStore } from "../store";
import { Seg } from "../ui";
import { cfg, stratName } from "./bots";
import { STATUS } from "./chainops";
import { RULES } from "./rules";

export type Step = "scan" | "strat" | "ai" | "risk" | "track";

const STEP_TABS: readonly (readonly [Step, string])[] = [
  ["scan", "Рынок"],
  ["strat", "Стратегии"],
  ["ai", "AI"],
  ["risk", "Защиты"],
  ["track", "Выход"],
];
export const isStep = (id: string): id is Step =>
  STEP_TABS.some(([s]) => s === id);

export interface Block {
  id: string;
  step: Step;
  title: string;
  sub: string;
  tag?: string;
  on: boolean;
  /** Работает всегда — убрать нельзя, только настроить в панели шага. */
  locked?: boolean;
  /** Точка статуса на чипе (провайдер AI): ok | warn | bad | "" — ещё не отвечал. */
  dot?: string;
  add: () => void;
  remove: () => void;
}

export interface Drag {
  id: string;
  from: "lib" | "lane";
}

const pct = (v: number) => `${Math.round(v * 100)}%`;

/** Все блоки с их текущим состоянием — из тех же данных, что и карточки шагов. */
export function useBlocks(): Block[] {
  const strategies = useStore((s) => s.strategies);
  const profiles = useStore((s) => s.profiles);
  const ai = useStore((s) => s.ai);
  const config = useStore((s) => s.config);
  const schema = useStore((s) => s.schema);
  return useMemo(() => {
    const st = useStore.getState();
    const out: Block[] = [];

    // Рынок: профиль один — новый блок заменяет прежний.
    const active = profiles.find((p) => p.active);
    for (const p of profiles) {
      out.push({
        id: `profile:${p.id}`,
        step: "scan",
        title: p.label,
        sub: `профиль рынка · ${p.timeframe}`,
        on: p.active,
        locked: true,
        add: async () => {
          await actions.setProfile(p.id);
          if (active && active.id !== p.id)
            st.sayUndo(
              `Профиль рынка: ${p.label}`,
              () => void actions.setProfile(active.id),
            );
        },
        remove: () =>
          st.say("Профиль убрать нельзя — перетащите на «Сканер» другой"),
      });
    }

    // Стратегии: включить — бросить на шаг, выключить — унести в библиотеку.
    for (const s of strategies) {
      const stat = s.live
        ? `живые ${pct(s.live.win_rate)} · ${s.live.trades}`
        : s.backtest
          ? `бэктест ${pct(s.backtest.win_rate)}`
          : "без статистики";
      out.push({
        id: `strat:${s.name}`,
        step: "strat",
        title: stratName(s.name),
        sub: stat,
        tag: s.experimental ? "эксп." : undefined,
        on: s.enabled,
        add: async () => {
          await actions.setStrategy(s.name, true);
          st.sayUndo(
            `Стратегия «${stratName(s.name)}» включена`,
            () => void actions.setStrategy(s.name, false),
          );
        },
        remove: async () => {
          await actions.setStrategy(s.name, false);
          st.sayUndo(
            `Стратегия «${stratName(s.name)}» выключена`,
            () => void actions.setStrategy(s.name, true),
          );
        },
      });
    }

    // AI: блок в цепочке — провайдер спрашивается по очереди; в библиотеке — в резерве (ключ остаётся).
    if (ai) {
      const order = ai.chain.map((c) => c.provider);
      const disabled = ai.providers.filter((p) => p.disabled).map((p) => p.id);
      const back = () => void actions.putAiChain({ disabled, order });
      // Полоса шага — в порядке очереди, с номером и статусом (это и есть цепочка AI на холсте).
      const rank = (id: string) => (order.includes(id) ? order.indexOf(id) : order.length);
      for (const p of [...ai.providers].sort((a, b) => rank(a.id) - rank(b.id))) {
        const on = order.includes(p.id);
        const [dot, said] = STATUS[p.status] ?? STATUS.unknown;
        out.push({
          id: `ai:${p.id}`,
          step: "ai",
          title: on && !ai.single ? `${order.indexOf(p.id) + 1} · ${p.name}` : p.name,
          dot: on ? dot : undefined,
          sub: on
            ? `${order.indexOf(p.id) + 1} в очереди · ${said}`
            : p.usable
              ? p.free_tier
                ? "бесплатный"
                : "платный"
              : "нужен ключ",
          tag: p.usable ? undefined : "ключ",
          on,
          add: async () => {
            if (!p.usable) {
              st.openNode("ai");
              st.say(
                `Вставьте ключ ${p.name} в поле «Вставьте ключ» — и он встанет в цепочку`,
              );
              return;
            }
            await actions.putAiChain({
              disabled: disabled.filter((x) => x !== p.id),
              order: [...order.filter((x) => x !== p.id), p.id],
            });
            st.sayUndo(`«${p.name}» в цепочке`, back);
          },
          remove: async () => {
            if (order.length <= 1)
              return st.say("В цепочке должен остаться хотя бы один провайдер");
            await actions.putAiChain({ disabled: [...disabled, p.id] });
            st.sayUndo(`«${p.name}» в резерве — ключ сохранён`, back);
          },
        });
      }
    }

    // Защиты: те же правила, что выполняет бот. Выключаемые — тумблером или нулём в пороге; остальные работают всегда.
    for (const r of RULES) {
      if (!r.key && !r.on) continue;
      const toggle = Boolean(r.on || r.zeroOff);
      const on = r.on
        ? cfg<boolean>(config, r.on, true) !== false
        : r.zeroOff && r.key
          ? Number(cfg(config, r.key, 0)) > 0
          : true;
      const def = r.key
        ? Number(schema.find((d) => d.key === r.key)?.default)
        : 0;
      out.push({
        id: `rule:${r.id}`,
        step: "risk",
        title: r.ifm,
        sub: r.cat,
        tag: toggle ? undefined : "всегда",
        on,
        locked: !toggle,
        // patchConfig сам предлагает «Отменить».
        add: () =>
          void actions.patchConfig(
            r.on ? { [r.on]: true } : { [r.key as string]: def > 0 ? def : 1 },
          ),
        remove: () =>
          toggle
            ? void actions.patchConfig(
                r.on ? { [r.on]: false } : { [r.key as string]: 0 },
              )
            : st.say(
                "Эта защита работает всегда — порог меняется в панели шага",
              ),
      });
    }

    // Выход: частичная фиксация включается блоком; трейлинг работает всегда.
    out.push({
      id: "exit:ptp",
      step: "track",
      title: "Частичная фиксация",
      sub: `${pct(cfg(config, "PARTIAL_TP_FRACTION", 0.5))} на ${pct(cfg(config, "PARTIAL_TP_TRIGGER", 0.6))} пути`,
      on: cfg<boolean>(config, "PARTIAL_TP_ENABLED", true) !== false,
      add: () => void actions.patchConfig({ PARTIAL_TP_ENABLED: true }),
      remove: () => void actions.patchConfig({ PARTIAL_TP_ENABLED: false }),
    });
    out.push({
      id: "exit:trail",
      step: "track",
      title: "Трейлинг-стоп",
      sub: `${cfg(config, "TRAILING_STOP_ATR_MULT", 1)} × ATR`,
      tag: "всегда",
      on: true,
      locked: true,
      add: () => undefined,
      remove: () =>
        st.say("Трейлинг работает всегда — отступ меняется в панели шага"),
    });
    return out;
  }, [strategies, profiles, ai, config, schema]);
}

const dragStart = (e: React.DragEvent, id: string) => {
  e.dataTransfer.setData("text/plain", id);
  e.dataTransfer.effectAllowed = "move";
};

/** Библиотека блоков над конвейером: вкладки по шагам, блоки, которых ещё нет в цепочке. */
export function BlockLibrary(props: {
  blocks: Block[];
  drag: Drag | null;
  setDrag: (d: Drag | null) => void;
}) {
  const { blocks, drag, setDrag } = props;
  const [step, setStep] = useState<Step>("strat");
  const [q, setQ] = useState("");
  const needle = q.trim().toLowerCase();
  const list = blocks.filter(
    (b) =>
      b.step === step &&
      !b.on &&
      (!needle || b.title.toLowerCase().includes(needle)),
  );
  const back = drag?.from === "lane";
  const title = STEP_TABS.find(([s]) => s === step)?.[1] ?? "";
  return (
    <div
      className={`blk-lib${back ? " back" : ""}`}
      onDragOver={(e) => back && e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        const b = drag && blocks.find((x) => x.id === drag.id);
        setDrag(null);
        if (b && back) b.remove();
      }}
    >
      <div className="blk-lib-h">
        <b>Блоки</b>
        <span className="label">
          {back
            ? "Отпустите здесь — блок выключится"
            : "перетащите на шаг ниже или нажмите"}
        </span>
        <Seg
          className="sm sp"
          value={step}
          options={STEP_TABS}
          onChange={(v) => setStep(v)}
          label="Блоки какого шага"
        />
      </div>
      {step === "strat" && (
        <input
          className="blk-find"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Найти стратегию"
          aria-label="Найти стратегию"
        />
      )}
      <div className="blk-list">
        {list.map((b) => (
          <button
            key={b.id}
            type="button"
            className="blk"
            draggable
            onDragStart={(e) => {
              dragStart(e, b.id);
              setDrag({ id: b.id, from: "lib" });
            }}
            onDragEnd={() => setDrag(null)}
            onClick={() => b.add()}
            aria-label={`Добавить «${b.title}» в шаг «${title}»`}
          >
            <span className="grip" aria-hidden="true">
              ⋮⋮
            </span>
            <span className="blk-t">
              <b>{b.title}</b>
              <small>{b.sub}</small>
            </span>
            {b.tag && <i className="blk-tag">{b.tag}</i>}
            <span className="blk-plus" aria-hidden="true">
              +
            </span>
          </button>
        ))}
        {!list.length && (
          <p className="blk-empty">
            {needle
              ? "Ничего не нашлось"
              : "Все блоки этого шага уже в цепочке."}
          </p>
        )}
      </div>
    </div>
  );
}

/** Полоса под шагом: его блоки. Сюда бросают из библиотеки; отсюда уносят обратно. */
export function Lane(props: {
  step: Step;
  blocks: Block[];
  drag: Drag | null;
  setDrag: (d: Drag | null) => void;
  onMore: () => void;
}) {
  const { step, blocks, drag, setDrag, onMore } = props;
  const [over, setOver] = useState(false);
  const mine = blocks.filter((b) => b.step === step && b.on);
  const dragged = drag ? blocks.find((x) => x.id === drag.id) : undefined;
  const fromLib = drag?.from === "lib";
  const valid = fromLib && dragged?.step === step;
  const cap = step === "strat" ? 5 : 6;
  return (
    <div
      className={`lane2${valid ? " can" : ""}${valid && over ? " over" : ""}${fromLib && !valid ? " dim" : ""}`}
      onDragOver={(e) => {
        if (!valid) return;
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        setDrag(null);
        if (valid && dragged) dragged.add();
      }}
    >
      {mine.slice(0, cap).map((b) => (
        <span
          key={b.id}
          className={`chip2${b.locked ? " locked" : ""}`}
          draggable={!b.locked}
          onDragStart={(e) => {
            dragStart(e, b.id);
            setDrag({ id: b.id, from: "lane" });
          }}
          onDragEnd={() => setDrag(null)}
          title={b.sub}
        >
          {b.dot !== undefined && <span className={`dot ${b.dot}`} aria-hidden="true" />}
          <span className="chip2-t">{b.title}</span>
          {b.locked ? (
            <i className="chip2-lock">{b.tag ?? "основа"}</i>
          ) : (
            <button
              type="button"
              aria-label={`Убрать «${b.title}»`}
              onClick={() => b.remove()}
            >
              ×
            </button>
          )}
        </span>
      ))}
      {mine.length > cap && (
        <button type="button" className="chip2 more" onClick={onMore}>
          ещё {mine.length - cap}
        </button>
      )}
      {valid && (
        <span className="lane2-hint">
          {over ? "Отпустите — блок встанет сюда" : "Сюда"}
        </span>
      )}
      {!mine.length && !valid && (
        <span className="lane2-empty">пусто — перетащите блок</span>
      )}
    </div>
  );
}
