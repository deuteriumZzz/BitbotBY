// Раздел «AI» в настройках: режим и порядок цепочки, модели, статусы, новые провайдеры.

import { useEffect, useLayoutEffect, useRef, useState } from "react";

import { fmtN, plural } from "../lib/format";
import { detectProvider } from "../lib/keys";
import { reducedMotion } from "../lib/platform";
import type { AiMode, AiModel, AiProvider, AiState, SecretItem } from "../lib/types";
import { actions, useStore } from "../store";
import { Icon, Q, Seg, Skel, Toggle } from "../ui";

const EASE = "cubic-bezier(.2,.8,.2,1)";

const MODES: readonly (readonly [AiMode, string])[] = [
  ["free", "Бесплатные"],
  ["auto", "Смешанный"],
  ["paid", "Платные"],
];
const MODE_NOTE: Record<AiMode, string> = {
  free: "Только бесплатные модели. Бот никогда сам не начнёт тратить деньги: платные провайдеры не используются.",
  auto: "Сначала бесплатные, платные — только когда бесплатные кончились или на паузе.",
  paid: "Только платные провайдеры: бесплатные не используются.",
};

const HOW: readonly (readonly ["chain" | "single", string])[] = [
  ["chain", "Цепочка"],
  ["single", "Один провайдер"],
];
const HOW_NOTE = {
  chain: "Если провайдер не ответил или упёрся в лимит, бот сразу спрашивает следующего — AI почти всегда на связи.",
  single:
    "Решения всегда принимает одна и та же модель — стабильнее и предсказуемее. Если она недоступна (лимит, нет связи), цикл идёт без AI, на локальных стратегиях.",
};

/** Провайдер, который решает в режиме «Один провайдер». */
const chosenId = (ai: AiState) => (ai.pinned !== "auto" ? ai.pinned : (ai.chain[0]?.provider ?? ""));

const STATUS: Record<string, [string, string]> = {
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

const hhmm = (epoch: number) =>
  new Date(epoch * 1000).toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" });

const modelLabel = (m: AiModel) =>
  `${m.id}${m.recommended ? " ★" : ""} · ${m.free ? "бесплатно" : "платно"}${m.strong ? "" : " · слабая"}`;

/** Модели, выбранные в приложении: PUT заменяет карту целиком. */
const appModels = (ai: AiState) =>
  Object.fromEntries(ai.providers.filter((p) => p.model_source === "app").map((p) => [p.id, p.model]));

/** Закрепить провайдера (AI_PROVIDER, как /provider в Telegram) и сразу обновить раздел. */
async function pin(id: string): Promise<void> {
  if (await actions.patchConfig({ AI_PROVIDER: id })) await useStore.getState().loadAi().catch(() => undefined);
}

/** Цепочка ↔ один провайдер. Переходя к одному, явно закрепляем того, кто сейчас первый. */
async function setHow(ai: AiState, how: "chain" | "single"): Promise<void> {
  if (how === "single" && ai.pinned === "auto" && ai.chain[0]) {
    await pin(ai.chain[0].provider);
  }
  await actions.putAiChain({ single: how === "single" }, how === "single" ? "Работает один провайдер, без запасных" : "Работает цепочка провайдеров");
}

export function AiPane() {
  const ai = useStore((s) => s.ai);
  const secrets = useStore((s) => s.secrets);
  const [needRestart, setNeedRestart] = useState(false);
  const local = useStore((s) => s.bot?.mode === "local");
  if (!ai) return <Skel rows={4} label="Загружаю список провайдеров" />;
  const inChain = ai.providers.filter((p) => p.usable);
  const rest = ai.providers.filter((p) => !p.usable);
  const secretOf = (id: string) => secrets?.find((s) => s.service === id);
  return (
    <div className="set-pane view enter">
      {local && (
        <div className="honest">
          {Icon.warn}
          <div>
            <b>Сейчас AI не используется.</b> Включён режим «Локальные стратегии (без AI)» — провайдеры ниже подключены, но бот их не
            спрашивает.{" "}
            <button className="btn" type="button" onClick={() => void actions.patchConfig({ MODE: "ai" })}>
              Вернуть AI
            </button>
          </div>
        </div>
      )}
      <ChainCard ai={ai} />
      <article className="card">
        <div className="card-h">
          <h2>{ai.single ? "Подключённые провайдеры" : "В цепочке"}</h2>
          <Q
            tip={
              ai.single
                ? "В режиме «Один провайдер» решает только выбранный; остальные подключены, но не используются."
                : "Бот спрашивает провайдеров сверху вниз. Ошибка или лимит — сразу следующий; после лимита провайдер отдыхает и потом возвращается."
            }
          />
          <span className="pill sp">{ai.chain.length}</span>
        </div>
        {inChain.length ? <ChainList ai={ai} list={inChain} /> : <div className="empty">
            <p>Пока ни одного провайдера. Добавьте бесплатный — хватит одного ключа.</p>
            <button className="btn primary" type="button" onClick={() => document.querySelector<HTMLInputElement>(".quick input")?.focus()}>
              Вставить ключ
            </button>
          </div>}
      </article>
      <QuickAdd ai={ai} secretOf={secretOf} onSaved={() => setNeedRestart(true)} />
      <AddProviders ai={ai} list={rest} secretOf={secretOf} onSaved={() => setNeedRestart(true)} />
      {needRestart && (
        <div className="applybar">
          <span>Новые ключи заработают после перезапуска бота</span>
          <span className="sp">
            <button
              className="btn primary"
              type="button"
              onClick={() => void actions.restart().then((ok) => ok && setNeedRestart(false))}
            >
              Перезапустить бота
            </button>
          </span>
        </div>
      )}
    </div>
  );
}

// ── Режим и что сейчас отвечает ───────────────────────────────────────────────

function ChainCard({ ai }: { ai: AiState }) {
  const byId = Object.fromEntries(ai.providers.map((p) => [p.id, p]));
  const last = byId[ai.last];
  const pinned = ai.pinned !== "auto" ? byId[ai.pinned] : undefined;
  return (
    <article className="card">
      <div className="card-h">
        <h2>Цепочка AI</h2>
        {last && (
          <span className="pill ok sp">
            <span className="dot ok live" />
            отвечает {last.name}
          </span>
        )}
      </div>
      <div className="field">
        <div className="field-h">
          <span>Как работать</span>
        </div>
        <Seg className="full" value={ai.single ? "single" : "chain"} options={HOW} label="Как работать" onChange={(how) => void setHow(ai, how)} />
        <p>{HOW_NOTE[ai.single ? "single" : "chain"]}</p>
      </div>
      {ai.single ? (
        <div className="field">
          <div className="field-h">
            <span>Провайдер</span>
          </div>
          <select
            className="sel"
            value={chosenId(ai)}
            aria-label="Провайдер, который принимает решения"
            onChange={(e) => void pin(e.target.value)}
          >
            {!chosenId(ai) && <option value="">— выберите —</option>}
            {ai.providers
              .filter((p) => p.usable)
              .map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} · {p.model}
                </option>
              ))}
          </select>
        </div>
      ) : (
        <div className="field">
          <div className="field-h">
            <span>Какие модели</span>
          </div>
          <Seg className="full" value={ai.mode} options={MODES} label="Режим цепочки" onChange={(mode) => void actions.putAiChain({ mode }, MODES.find(([m]) => m === mode)?.[1])} />
          <p>{MODE_NOTE[ai.mode]}</p>
        </div>
      )}
      <p className="ai-route">
        {ai.chain.length ? (
          ai.chain.map((c, i) => (
            <span key={c.provider}>
              {i > 0 && " → "}
              <b>{byId[c.provider]?.name ?? c.provider}</b>
            </span>
          ))
        ) : (
          <span className="down">
            {ai.single ? "Выбранный провайдер недоступен" : "Нет доступных провайдеров"} — бот работает на локальных стратегиях.
          </span>
        )}
        {ai.single && ai.chain[0] && <span className="muted"> · {ai.chain[0].model} · без запасных</span>}
      </p>
      {pinned && !ai.single && (
        <div className="row-tgl" style={{ marginTop: 8 }}>
          <span className="note-soon">
            Первым всегда идёт <b>{pinned.name}</b> — выбран в /provider.
          </span>
          <button className="btn" type="button" onClick={() => void pin("auto")}>
            Открепить
          </button>
        </div>
      )}
    </article>
  );
}

// ── Подключённые провайдеры ───────────────────────────────────────────────────

/** Карточки плавно переезжают при перестановке (FLIP по offsetTop). */
function useListFlip(ref: React.RefObject<HTMLElement>, skip?: React.MutableRefObject<boolean>): void {
  const last = useRef(new Map<string, number>());
  useLayoutEffect(() => {
    const host = ref.current;
    if (!host) return;
    const next = new Map<string, number>();
    const quiet = skip?.current ?? false;
    host.querySelectorAll<HTMLElement>("[data-pid]").forEach((el) => {
      const k = el.dataset.pid!;
      next.set(k, el.offsetTop);
      const prev = last.current.get(k);
      if (!quiet && prev !== undefined && prev !== el.offsetTop && !reducedMotion())
        el.animate([{ transform: `translateY(${prev - el.offsetTop}px)` }, { transform: "none" }], { duration: 380, easing: EASE });
    });
    last.current = next;
    if (skip && quiet) skip.current = false;
  });
}

/** Эстафета: метка «отвечает» переезжает к провайдеру, который ответил последним. */
function useRelay(host: React.RefObject<HTMLElement>, last: string): React.CSSProperties | null {
  const [at, setAt] = useState<[number, number] | null>(null);
  // Строки меняют высоту (статус, модель) — сверяемся после каждой отрисовки,
  // но состояние трогаем только при реальном сдвиге, иначе цикл рендеров.
  useLayoutEffect(() => {
    const row = last ? host.current?.querySelector<HTMLElement>(`[data-pid="${CSS.escape(last)}"]`) : null;
    const next: [number, number] | null = row ? [row.offsetTop, row.offsetHeight] : null;
    setAt((cur) => (cur?.[0] === next?.[0] && cur?.[1] === next?.[1] ? cur : next));
  });
  return at && { transform: `translateY(${at[0]}px)`, height: at[1] };
}

const applyOrder = (all: string[], subset: string[], next: string[]): string[] => {
  const inSet = new Set(subset);
  let k = 0;
  return all.map((id) => (inSet.has(id) ? next[k++] : id));
};

interface Probe {
  busy?: boolean;
  ok?: boolean;
  ms?: number;
  message?: string;
}
interface Drag {
  id: string;
  dy: number;
  orig: Record<string, number>;
  tops: Record<string, number>;
  bin: boolean;
}
interface CtxItem {
  label: string;
  hint?: string;
  danger?: boolean;
  disabled?: boolean;
  run: () => void;
}

/** Контекстное меню: правый щелчок, долгое нажатие или кнопка «⋯»; стрелки, Enter, Esc. */
function ChainMenu({ items, x, y, onClose }: { items: CtxItem[]; x: number; y: number; onClose: () => void }) {
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

function ChainList({ ai, list }: { ai: AiState; list: AiProvider[] }) {
  const host = useRef<HTMLOListElement>(null);
  const noFlip = useRef(false);
  useListFlip(host, noFlip);
  const relay = useRelay(host, ai.last);
  const say = useStore((s) => s.say);
  const sayUndo = useStore((s) => s.sayUndo);
  const position = new Map(ai.chain.map((c, i) => [c.provider, i + 1]));
  const active = list.filter((p) => !p.disabled);
  const reserve = list.filter((p) => p.disabled);
  const ids = active.map((p) => p.id);
  const disabledIds = reserve.map((p) => p.id);
  const [order, setOrder] = useState<string[] | null>(null); // клавиатура и ожидание ответа сервера
  const [grab, setGrab] = useState<string | null>(null);
  const [drag, setDrag] = useState<Drag | null>(null);
  const [menu, setMenu] = useState<{ id: string; x: number; y: number } | null>(null);
  const [probes, setProbes] = useState<Record<string, Probe>>({});
  const [picked, setPicked] = useState<string[]>([]);
  const [heir, setHeir] = useState<string | null>(null);
  useEffect(() => setOrder(null), [ids.join()]); // eslint-disable-line react-hooks/exhaustive-deps
  const shown = (order ?? ids).map((id) => active.find((p) => p.id === id)).filter((p): p is AiProvider => Boolean(p));
  const nameOf = (id: string) => ai.providers.find((p) => p.id === id)?.name ?? id;

  const commit = (next: string[]) => {
    if (next.join() === ids.join()) return setOrder(null);
    setOrder(next);
    void actions.putAiChain({ order: applyOrder(ai.providers.map((x) => x.id), ids, next) }).then((ok) => {
      if (!ok) setOrder(null);
    });
  };
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

  // ── Перетаскивание мышью и пальцем ──
  const startDrag = (e: React.PointerEvent<HTMLButtonElement>, id: string) => {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    const rows = Array.from(host.current?.querySelectorAll<HTMLElement>("[data-pid]") ?? []);
    const rects = rows.map((r) => ({ id: r.dataset.pid ?? "", top: r.offsetTop, h: r.offsetHeight }));
    const me = rects.find((r) => r.id === id);
    if (!me) return;
    e.preventDefault();
    const gap = rects.length > 1 ? rects[1].top - rects[0].top - rects[0].h : 0;
    const orig = Object.fromEntries(rects.map((r) => [r.id, r.top]));
    const layout = (seq: string[]) => {
      const tops: Record<string, number> = {};
      let t = rects[0].top;
      for (const k of seq) {
        tops[k] = t;
        t += (rects.find((r) => r.id === k)?.h ?? 0) + gap;
      }
      return tops;
    };
    const grip = e.currentTarget;
    grip.setPointerCapture(e.pointerId);
    const y0 = e.clientY;
    let plan = rects.map((r) => r.id);
    let overBin = false;
    const onMove = (ev: PointerEvent) => {
      const dy = ev.clientY - y0;
      const binTop = document.querySelector(".chain-bin")?.getBoundingClientRect().top ?? Infinity;
      overBin = ev.clientY > binTop - 8;
      const others = rects.filter((r) => r.id !== id);
      const center = me.top + me.h / 2 + dy;
      const idx = others.filter((r) => r.top + r.h / 2 < center).length;
      const seq = others.map((r) => r.id);
      seq.splice(idx, 0, id);
      plan = overBin ? rects.map((r) => r.id) : seq;
      setDrag({ id, dy, orig, tops: layout(plan), bin: overBin });
    };
    const end = (cancel: boolean) => {
      grip.removeEventListener("pointermove", onMove);
      grip.removeEventListener("pointerup", onUp);
      grip.removeEventListener("pointercancel", onCancel);
      if (cancel) return setDrag(null);
      const p = active.find((x) => x.id === id);
      if (overBin && p) {
        setDrag(null);
        return void toReserve(p);
      }
      // Строка «садится» в свою ячейку, и только потом порядок уходит на сервер.
      setDrag({ id, dy: layout(plan)[id] - me.top, orig, tops: layout(plan), bin: false });
      setTimeout(() => {
        noFlip.current = true;
        commit(plan);
        setDrag(null);
      }, reducedMotion() ? 0 : 170);
    };
    const onUp = () => end(false);
    const onCancel = () => end(true);
    grip.addEventListener("pointermove", onMove);
    grip.addEventListener("pointerup", onUp);
    grip.addEventListener("pointercancel", onCancel);
    setDrag({ id, dy: 0, orig, tops: layout(plan), bin: false });
  };

  // ── Клавиатура: пробел «взять», стрелки — перенести, пробел — положить, Esc — отмена ──
  const onGripKey = (e: React.KeyboardEvent, id: string) => {
    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      if (grab === id) {
        setGrab(null);
        commit(shown.map((p) => p.id));
      } else setGrab(id);
    } else if (grab === id && e.key === "Escape") {
      e.preventDefault();
      setGrab(null);
      setOrder(null);
    } else if (grab === id && (e.key === "ArrowUp" || e.key === "ArrowDown")) {
      e.preventDefault();
      const cur = shown.map((p) => p.id);
      const i = cur.indexOf(id);
      const j = i + (e.key === "ArrowUp" ? -1 : 1);
      if (j < 0 || j >= cur.length) return;
      [cur[i], cur[j]] = [cur[j], cur[i]];
      setOrder(cur);
    }
  };
  useEffect(() => {
    if (grab) host.current?.querySelector<HTMLElement>(`[data-pid="${CSS.escape(grab)}"] .grip`)?.focus();
  }, [order, grab]);

  const menuProvider = menu ? ai.providers.find((p) => p.id === menu.id) : undefined;
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

  return (
    <div className="aichain-wrap">
      {picked.length > 0 && (
        <div className="chain-sel" role="toolbar" aria-label="Выбранные провайдеры">
          <b>Выбрано {picked.length}</b>
          <button className="btn" type="button" onClick={() => picked.forEach((id) => { const p = active.find((x) => x.id === id); if (p) void probe(p); })}>
            Проверить
          </button>
          <button
            className="btn"
            type="button"
            onClick={async () => {
              const keep = picked;
              if (await actions.putAiChain({ disabled: [...new Set([...disabledIds, ...keep])] }))
                sayUndo(`В резерве: ${keep.length}`, () => void actions.putAiChain({ disabled: disabledIds }));
              setPicked([]);
            }}
          >
            В резерв
          </button>
          <button className="btn" type="button" onClick={() => setPicked([])}>
            Снять выделение
          </button>
        </div>
      )}
      {relay && <span className="relay" style={relay} title="Последним ответил этот провайдер" aria-hidden="true" />}
      <ol className={`aichain${drag ? " dragging" : ""}`} ref={host}>
        {shown.map((p, i) => (
          <ProviderRow
            key={p.id}
            ai={ai}
            p={p}
            n={position.get(p.id)}
            first={i === 0}
            lastRow={i === shown.length - 1}
            onMove={(d) => {
              const cur = shown.map((x) => x.id);
              const j = i + d;
              if (j < 0 || j >= cur.length) return;
              [cur[i], cur[j]] = [cur[j], cur[i]];
              commit(cur);
            }}
            shift={drag && drag.id !== p.id ? drag.tops[p.id] - drag.orig[p.id] : undefined}
            lifted={drag?.id === p.id ? drag.dy : undefined}
            grabbed={grab === p.id}
            selected={picked.includes(p.id)}
            heir={heir === p.id}
            probe={probes[p.id]}
            onGripDown={(e) => startDrag(e, p.id)}
            onGripKey={(e) => onGripKey(e, p.id)}
            onMenu={(x, y) => setMenu({ id: p.id, x, y })}
            onPick={() => setPicked((s) => (s.includes(p.id) ? s.filter((x) => x !== p.id) : [...s, p.id]))}
          />
        ))}
      </ol>
      <div className={`chain-bin${drag?.bin ? " over" : ""}`} aria-label="Резерв: выключенные провайдеры">
        <span className="bin-h">Резерв</span>
        {reserve.length ? (
          reserve.map((p) => (
            <button key={p.id} className="bin-chip" type="button" title="Вернуть в цепочку" onClick={() => void setDisabled(p.id, false)}>
              {p.name}
              <span>вернуть</span>
            </button>
          ))
        ) : (
          <span className="bin-hint">Перетащите сюда провайдера за ⋮⋮ или выберите «В резерв» в меню: бот его не спросит, ключ останется.</span>
        )}
      </div>
      <p className="sr-only" aria-live="polite">
        {grab ? `${nameOf(grab)}: позиция ${shown.findIndex((p) => p.id === grab) + 1} из ${shown.length}. Стрелки переносят, пробел сохраняет, Esc отменяет.` : ""}
      </p>
      {menu && menuProvider && <ChainMenu items={items(menuProvider)} x={menu.x} y={menu.y} onClose={() => setMenu(null)} />}
    </div>
  );
}

function why(ai: AiState, p: AiProvider): string {
  if (ai.single) return p.id === chosenId(ai) ? "" : "запасной — не используется в режиме «Один провайдер»";
  if (p.id === ai.pinned) return "";
  if (p.disabled) return "выключен вами";
  if (ai.mode === "free" && !p.free) return "платная модель — не используется в режиме «Бесплатные»";
  if (ai.mode === "paid" && p.free) return "бесплатная модель — не используется в режиме «Платные»";
  return "";
}

interface RowProps {
  ai: AiState;
  p: AiProvider;
  n?: number;
  first: boolean;
  lastRow: boolean;
  onMove: (dir: -1 | 1) => void;
  shift?: number;
  lifted?: number;
  grabbed: boolean;
  selected: boolean;
  heir: boolean;
  probe?: Probe;
  onGripDown: (e: React.PointerEvent<HTMLButtonElement>) => void;
  onGripKey: (e: React.KeyboardEvent<HTMLButtonElement>) => void;
  onMenu: (x: number, y: number) => void;
  onPick: () => void;
}

function ProviderRow(props: RowProps) {
  const { ai, p } = props;
  const [cls, text] = STATUS[p.status] ?? ["warn", p.status];
  const paused = p.cooldown_until && p.cooldown_until * 1000 > Date.now();
  const left = paused ? Math.round(p.cooldown_until! - Date.now() / 1000) : 0;
  const skip = why(ai, p);
  const disabledIds = ai.providers.filter((x) => x.disabled).map((x) => x.id);
  const keyless = ai.providers.filter((x) => x.keyless_on).map((x) => x.id);
  // Провайдер только что отказал (лимит, ошибка): строка коротко «вздрагивает».
  const prev = useRef(p.status);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    const was = prev.current;
    prev.current = p.status;
    if (was !== p.status && p.status !== "ok" && p.status !== "unknown") {
      setFailed(true);
      const t = setTimeout(() => setFailed(false), 1100);
      return () => clearTimeout(t);
    }
  }, [p.status]);
  // Долгое нажатие пальцем открывает то же меню, что правый щелчок.
  const press = useRef<ReturnType<typeof setTimeout>>();
  const origin = useRef<[number, number]>([0, 0]);
  const cancelPress = () => clearTimeout(press.current);
  const moving = props.lifted !== undefined || props.shift !== undefined;
  const style: React.CSSProperties | undefined =
    props.lifted !== undefined
      ? { transform: `translateY(${props.lifted}px)`, zIndex: 4 }
      : props.shift !== undefined
        ? { transform: `translateY(${props.shift}px)` }
        : undefined;
  const role = ai.single ? "" : props.n === 1 ? "основной" : props.n ? "запасной" : "";
  return (
    <li
      className={`aip${props.n ? "" : " off"}${moving ? " shifting" : ""}${props.lifted !== undefined ? " lifted" : ""}${props.selected ? " sel" : ""}${props.heir ? " heir" : ""}${failed ? " fail" : ""}${props.grabbed ? " grabbed" : ""}`}
      data-pid={p.id}
      style={style}
      onContextMenu={(e) => {
        e.preventDefault();
        props.onMenu(e.clientX, e.clientY);
      }}
      onPointerDown={(e) => {
        if (e.pointerType !== "touch") return;
        origin.current = [e.clientX, e.clientY];
        press.current = setTimeout(() => props.onMenu(origin.current[0], origin.current[1]), 520);
      }}
      onPointerMove={(e) => Math.hypot(e.clientX - origin.current[0], e.clientY - origin.current[1]) > 8 && cancelPress()}
      onPointerUp={cancelPress}
      onPointerCancel={cancelPress}
    >
      <div
        className="aip-h"
        onClick={(e) => {
          if (e.metaKey || e.ctrlKey || e.shiftKey) props.onPick();
        }}
      >
        <button
          className="grip"
          type="button"
          aria-label={`Перенести: ${p.name}. Пробел — взять, стрелки — переместить`}
          aria-pressed={props.grabbed}
          onPointerDown={props.onGripDown}
          onKeyDown={props.onGripKey}
        >
          ⋮⋮
        </button>
        <span className="n">
          {props.n ?? "—"}
          {/* Пауза после лимита: кольцо убывает до конца паузы. */}
          {paused && (
            <svg key={p.cooldown_until} className="pause-ring" viewBox="0 0 20 20" style={{ animationDuration: `${left}s` }} aria-hidden="true">
              <circle cx="10" cy="10" r="8.5" pathLength={1} style={{ animationDuration: `${left}s` }} />
            </svg>
          )}
        </span>
        <b>{p.name}</b>
        <span className={`tag ${p.free ? "hot" : "rst"}`}>{p.free ? "бесплатно" : "платно"}</span>
        {role && <span className="tag rst">{role}</span>}
        {!p.strong && <span className="tag rst" title="Мелкая модель чаще ломает формат ответа и хуже решает">слабая</span>}
        {p.id === ai.pinned && <span className="tag hot">первый</span>}
        <span className="sp">
          {props.probe && (
            <span className={`probe ${props.probe.busy ? "" : props.probe.ok ? "ok" : "bad"}`} role="status">
              {props.probe.busy ? <><span className="spin" /> проверяю…</> : props.probe.ok ? `✓ ${props.probe.ms} мс` : `✗ ${props.probe.message}`}
            </span>
          )}
          <span className={`pill ${cls}`}>
            <span className={`dot ${cls}`} />
            {paused ? `пауза до ${hhmm(p.cooldown_until!)}` : text}
          </span>
          <span className="mv">
            <button className="btn" type="button" aria-label={`Выше: ${p.name}`} disabled={props.first} onClick={() => props.onMove(-1)}>
              ↑
            </button>
            <button className="btn" type="button" aria-label={`Ниже: ${p.name}`} disabled={props.lastRow} onClick={() => props.onMove(1)}>
              ↓
            </button>
            <button
              className="btn"
              type="button"
              aria-haspopup="menu"
              aria-label={`Действия: ${p.name}`}
              onClick={(e) => {
                const r = e.currentTarget.getBoundingClientRect();
                props.onMenu(r.left, r.bottom + 4);
              }}
            >
              ⋯
            </button>
          </span>
          <Toggle
            on={!p.disabled}
            label={`${p.name} в цепочке`}
            onChange={(on) => void actions.putAiChain({ disabled: on ? disabledIds.filter((id) => id !== p.id) : [...disabledIds, p.id] })}
          />
        </span>
      </div>
      <ModelPicker ai={ai} p={p} />
      <small>
        {skip && <span className="down">{skip}. </span>}
        {p.model_source === "env" && "Модель из .env — выберите здесь, чтобы заменить. "}
        {!p.verified && "ID моделей не сверены — нажмите «Загрузить список». "}
        сегодня {p.calls_today} {plural(p.calls_today, ["запрос", "запроса", "запросов"])}
        {p.tokens_today > 0 && ` · ${fmtN(p.tokens_today, 0)} токенов`}
      </small>
      <div className="btn-row">
        {p.id !== (ai.single ? chosenId(ai) : ai.pinned) && (
          <button className="btn" type="button" onClick={() => void pin(p.id)}>
            {ai.single ? "Выбрать" : "Сделать первым"}
          </button>
        )}
        {p.keyless_on && (
          <button className="btn" type="button" onClick={() => void actions.putAiChain({ keyless: keyless.filter((id) => id !== p.id) }, `${p.name} отключён`)}>
            Отключить (без ключа)
          </button>
        )}
      </div>
    </li>
  );
}

function ModelPicker({ ai, p }: { ai: AiState; p: AiProvider }) {
  const [live, setLive] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const known = new Set(p.models.map((m) => m.id));
  const extra = [...new Set([...(known.has(p.model) ? [] : [p.model]), ...live.filter((id) => !known.has(id))])];
  const pick = (model: string) => void actions.putAiChain({ models: { ...appModels(ai), [p.id]: model } }, `${p.name}: ${model}`);
  return (
    <div className="aip-row">
      <select className="sel" value={p.model} aria-label={`Модель ${p.name}`} onChange={(e) => pick(e.target.value)}>
        <optgroup label="Каталог">
          {p.models.map((m) => (
            <option key={m.id} value={m.id}>
              {modelLabel(m)}
            </option>
          ))}
        </optgroup>
        {extra.length > 0 && (
          <optgroup label={live.length ? `У провайдера: ${live.length}` : "Своя"}>
            {extra.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </optgroup>
        )}
      </select>
      <button
        className="btn"
        type="button"
        disabled={busy}
        title="Спросить у провайдера актуальный список моделей (лимит не тратит)"
        onClick={async () => {
          setBusy(true);
          setLive(await actions.aiModels(p.id));
          setBusy(false);
        }}
      >
        {busy ? <span className="spin" /> : "Загрузить список"}
      </button>
    </div>
  );
}

// ── Быстрое добавление: вставили ключ — провайдер определился сам ─────────────

function QuickAdd(props: { ai: AiState; secretOf: (id: string) => SecretItem | undefined; onSaved: () => void }) {
  const { ai } = props;
  const [val, setVal] = useState("");
  const [pick, setPick] = useState("");
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);
  const [over, setOver] = useState(false);
  const guess = detectProvider(val);
  const chosen = guess.length === 1 ? guess[0] : pick;
  const provider = ai.providers.find((p) => p.id === chosen);
  const field = props.secretOf(chosen)?.fields.find((f) => f.secret) ?? props.secretOf(chosen)?.fields[0];
  const take = async (e: React.DragEvent) => {
    e.preventDefault();
    setOver(false);
    const file = e.dataTransfer.files[0];
    const text = file ? await file.text() : e.dataTransfer.getData("text");
    if (text.trim()) setVal(text.trim().split(/\s+/)[0]);
  };
  const save = async () => {
    if (!provider || !field) return;
    setBusy(true);
    const ok = await actions.putSecret(provider.id, { [field.env]: val.trim() });
    setBusy(false);
    if (ok) {
      setVal("");
      setPick("");
      setDone(true);
      setTimeout(() => setDone(false), 1800);
      props.onSaved();
    }
  };
  return (
    <article className={`card quick${over ? " over" : ""}${done ? " done" : ""}`} onDragOver={(e) => (e.preventDefault(), setOver(true))} onDragLeave={() => setOver(false)} onDrop={(e) => void take(e)}>
      <div className="card-h">
        <h2>Вставьте ключ</h2>
        <Q tip="Определю провайдера по началу ключа (например, gsk_ — Groq). Ключ уходит на сервер, проверяется запросом к сервису и хранится зашифрованным. Можно перетащить сюда текст или файл с ключом." />
      </div>
      <div className="quick-row">
        <svg className="lock" viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
          <rect x="5" y="11" width="14" height="9" rx="2" />
          <path className="shackle" d="M8 11V8a4 4 0 0 1 8 0v3" />
        </svg>
        <input
          className="inp"
          type="password"
          autoComplete="off"
          spellCheck={false}
          placeholder="Вставьте или перетащите ключ"
          aria-label="Ключ нового провайдера"
          value={val}
          onChange={(e) => setVal(e.target.value)}
        />
      </div>
      {val.trim().length >= 8 && (
        <div className="quick-guess" aria-live="polite">
          {guess.length === 1 && provider && (
            <span>
              Похоже на <b>{provider.name}</b>
              {provider.key_set && " — ключ уже есть, он будет заменён"}.
            </span>
          )}
          {guess.length > 1 && (
            <>
              <span>Начало подходит нескольким — выберите:</span>
              {guess.map((id) => (
                <button key={id} type="button" className="chip-btn" aria-pressed={pick === id} onClick={() => setPick(id)}>
                  {ai.providers.find((p) => p.id === id)?.name ?? id}
                </button>
              ))}
            </>
          )}
          {guess.length === 0 && (
            <>
              <span>Формат не узнал — выберите провайдера:</span>
              <select className="sel" value={pick} aria-label="Провайдер" onChange={(e) => setPick(e.target.value)}>
                <option value="">—</option>
                {ai.providers.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
            </>
          )}
          <button className="btn primary" type="button" disabled={busy || !provider || !field} onClick={() => void save()}>
            {busy ? (
              <>
                <span className="spin" />
                Проверяю…
              </>
            ) : (
              "Проверить и сохранить"
            )}
          </button>
        </div>
      )}
    </article>
  );
}

// ── Новые провайдеры ──────────────────────────────────────────────────────────

function AddProviders(props: { ai: AiState; list: AiProvider[]; secretOf: (id: string) => SecretItem | undefined; onSaved: () => void }) {
  const free = props.list.filter((p) => p.free_tier);
  const paid = props.list.filter((p) => !p.free_tier);
  if (!props.list.length) return null;
  return (
    <article className="card">
      <div className="card-h">
        <h2>Добавить провайдера</h2>
        <Q tip="Чем больше бесплатных провайдеров с ключами, тем реже бот остаётся без AI: у каждого свой дневной лимит." />
      </div>
      {[
        ["Бесплатные", free],
        ["Платные", paid],
      ].map(([title, list]) =>
        (list as AiProvider[]).length ? (
          <section className="kgroup" key={title as string}>
            <h3>{title as string}</h3>
            <div className="addp">
              {(list as AiProvider[]).map((p) => (
                <AddCard key={p.id} ai={props.ai} p={p} secret={props.secretOf(p.id)} onSaved={props.onSaved} />
              ))}
            </div>
          </section>
        ) : null,
      )}
    </article>
  );
}

export function AddCard({ ai, p, secret, onSaved }: { ai: AiState; p: AiProvider; secret?: SecretItem; onSaved: () => void }) {
  const [vals, setVals] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const say = useStore((s) => s.say);
  const fields = secret?.fields ?? [];
  const save = async () => {
    const filled = Object.fromEntries(
      Object.entries(vals)
        .filter(([, v]) => v.trim())
        .map(([k, v]) => [k, v.trim()]),
    );
    if (!Object.keys(filled).length) return say("Вставьте ключ.");
    setBusy(true);
    const ok = await actions.putSecret(p.id, filled);
    setBusy(false);
    if (ok) {
      setVals({});
      onSaved();
    }
  };
  const keyless = ai.providers.filter((x) => x.keyless_on).map((x) => x.id);
  return (
    <div className="kcard">
      <div className="kc-h">
        <b>{p.name}</b>
        <span className={`tag ${p.free_tier ? "hot" : "rst"}`}>{p.free_tier ? "бесплатно" : "платно"}</span>
      </div>
      <p className="kc-d">
        {p.note}{" "}
        {p.signup && (
          <a href={p.signup} target="_blank" rel="noopener noreferrer">
            Где взять ключ ↗
          </a>
        )}
      </p>
      {fields.map((f) => (
        <div className="kf" key={f.env}>
          <label htmlFor={`ai-${f.env}`}>
            <span>{f.label}</span>
            <span className="saved">{f.set ? (f.secret ? `сохранён ••••${f.last4 ?? ""}` : f.value) : f.optional ? "необязательно" : "не задан"}</span>
          </label>
          <input
            className="inp"
            id={`ai-${f.env}`}
            type={f.secret ? "password" : "text"}
            autoComplete="off"
            spellCheck={false}
            placeholder={f.secret ? "Вставьте ключ" : ""}
            value={vals[f.env] ?? ""}
            onChange={(e) => setVals({ ...vals, [f.env]: e.target.value })}
          />
        </div>
      ))}
      <div className="kc-a">
        {fields.length > 0 && (
          <button className="btn primary" type="button" disabled={busy} onClick={() => void save()}>
            {busy ? (
              <>
                <span className="spin" />
                Проверяю…
              </>
            ) : (
              "Проверить и сохранить"
            )}
          </button>
        )}
        {p.key_optional && (
          <button className="btn" type="button" onClick={() => void actions.putAiChain({ keyless: [...keyless, p.id] }, `${p.name} включён`)}>
            {Icon.check}
            Включить без ключа
          </button>
        )}
      </div>
      {secret?.message && <p className={`kc-msg ${secret.status === "ok" ? "ok" : secret.status === "bad" ? "bad" : ""}`}>{secret.message}</p>}
    </div>
  );
}
