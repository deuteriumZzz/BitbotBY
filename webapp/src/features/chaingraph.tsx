// Схема цепочки AI в стиле редактора потоков: узлы, линии с бегущим потоком, панорама и масштаб.
// Порядок цепочки — это порядок узлов слева направо. Перетащили узел — порядок пересобрался;
// перетащили вниз, в «Резерв» — провайдер выключен (ключ остаётся).

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import { reducedMotion } from "../lib/platform";
import type { AiProvider, AiState } from "../lib/types";
import { useStore } from "../store";
import { Icon } from "../ui";
import { Avatar, ChainMenu, STATUS, chosenId, hhmm, useChainOps } from "./chainops";

const NW = 214; // ширина узла
const NH = 98; // высота узла
const GX = 280; // шаг между узлами
const X0 = 28;
const ROW_Y = 70;
const RES_Y = 330;

interface View {
  x: number;
  y: number;
  z: number;
}
interface Move {
  id: string;
  dx: number;
  dy: number;
}

const edgePath = (x1: number, y1: number, x2: number, y2: number) => {
  const dx = Math.max(60, (x2 - x1) / 2);
  return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
};

export function ChainGraph({ ai, list, onAdd }: { ai: AiState; list: AiProvider[]; onAdd: () => void }) {
  const ops = useChainOps(ai, list);
  const { active, reserve, ids } = ops;
  const fallbackOn = useStore((s) => Boolean(s.config.LOCAL_FALLBACK?.value));
  const box = useRef<HTMLDivElement>(null);
  const [view, setView] = useState<View>({ x: 12, y: 8, z: 1 });
  const [move, setMove] = useState<Move | null>(null);
  const [order, setOrder] = useState<string[] | null>(null); // ожидание ответа сервера
  const [menu, setMenu] = useState<{ id: string; x: number; y: number } | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [pulse, setPulse] = useState<{ k: number; d: string } | null>(null);
  const position = new Map(ai.chain.map((c, i) => [c.provider, i + 1]));
  useEffect(() => setOrder(null), [ids.join()]); // eslint-disable-line react-hooks/exhaustive-deps

  const seq = (order ?? ids).filter((id) => active.some((p) => p.id === id));
  const byId = (id: string) => list.find((p) => p.id === id) as AiProvider;

  // План расстановки: во время переноса узел вставляется в ближайшую ячейку, остальные расступаются.
  const plan = useMemo(() => {
    if (!move) return { chain: seq, reserve: reserve.map((p) => p.id), toReserve: false, slot: -1 };
    const fromReserve = reserve.some((p) => p.id === move.id);
    const base = fromReserve ? [...seq] : seq.filter((id) => id !== move.id);
    const startX = fromReserve ? X0 + GX * reserve.findIndex((p) => p.id === move.id) : X0 + GX * (seq.indexOf(move.id) + 1);
    const startY = fromReserve ? RES_Y : ROW_Y;
    const cx = startX + move.dx + NW / 2;
    const cy = startY + move.dy + NH / 2;
    const toReserve = cy > RES_Y - 40;
    const slot = Math.max(0, Math.min(base.length, Math.round((cx - NW / 2 - X0 - GX) / GX)));
    if (toReserve) return { chain: base, reserve: [...reserve.map((p) => p.id).filter((id) => id !== move.id), move.id], toReserve: !fromReserve || true, slot: -1 };
    const chain = [...base.slice(0, slot), move.id, ...base.slice(slot)];
    return { chain, reserve: reserve.map((p) => p.id).filter((id) => id !== move.id), toReserve: false, slot };
  }, [move, seq.join(), reserve.map((p) => p.id).join()]); // eslint-disable-line react-hooks/exhaustive-deps

  const chainPos = (i: number) => ({ x: X0 + GX * (i + 1), y: ROW_Y });
  const posOf = (id: string): { x: number; y: number } => {
    const i = plan.chain.indexOf(id);
    if (i >= 0) return chainPos(i);
    const j = plan.reserve.indexOf(id);
    return { x: X0 + GX * j, y: RES_Y };
  };
  const endX = X0 + GX * (plan.chain.length + 1);

  // Подогнать вид под ширину, когда набор узлов меняется.
  const fit = () => {
    const w = box.current?.clientWidth ?? 800;
    const need = endX + NW + 40;
    setView({ x: 12, y: 8, z: Math.max(0.45, Math.min(1, w / need)) });
  };
  useLayoutEffect(fit, [plan.chain.length]); // eslint-disable-line react-hooks/exhaustive-deps

  // Поток запроса: когда отвечает новый провайдер, по линиям пробегает точка.
  const lastSeen = useRef(ai.last);
  useEffect(() => {
    if (!ai.last || ai.last === lastSeen.current) return;
    lastSeen.current = ai.last;
    const upto = seq.indexOf(ai.last);
    if (upto < 0 || reducedMotion()) return;
    let d = "";
    let px = X0 + NW;
    const py = ROW_Y + NH / 2;
    for (let i = 0; i <= upto; i++) {
      const nx = X0 + GX * (i + 1);
      d += `${i ? " L" : "M"} ${px} ${py} C ${px + 60} ${py}, ${nx - 60} ${py}, ${nx} ${py}`;
      px = nx + NW;
    }
    const k = Date.now();
    setPulse({ k, d });
    const t = setTimeout(() => setPulse((p) => (p?.k === k ? null : p)), 1300);
    return () => clearTimeout(t);
  }, [ai.last]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Панорама и масштаб ──
  const onBackground = (e: React.PointerEvent) => {
    if ((e.target as HTMLElement).closest("[data-node],button")) return;
    setSelected(null);
    const sx = e.clientX;
    const sy = e.clientY;
    const v0 = view;
    const mv = (ev: PointerEvent) => setView({ ...v0, x: v0.x + ev.clientX - sx, y: v0.y + ev.clientY - sy });
    const up = () => {
      window.removeEventListener("pointermove", mv);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", mv);
    window.addEventListener("pointerup", up);
  };
  const zoomAt = (factor: number, cx: number, cy: number) =>
    setView((v) => {
      const z = Math.max(0.4, Math.min(1.6, v.z * factor));
      const k = z / v.z;
      return { z, x: cx - (cx - v.x) * k, y: cy - (cy - v.y) * k };
    });
  const onWheel = (e: React.WheelEvent) => {
    const r = box.current?.getBoundingClientRect();
    if (!r) return;
    zoomAt(e.deltaY < 0 ? 1.1 : 1 / 1.1, e.clientX - r.left, e.clientY - r.top);
  };

  // ── Перенос узла ──
  const startMove = (e: React.PointerEvent, id: string) => {
    if ((e.target as HTMLElement).closest("button")) return;
    if (e.pointerType === "mouse" && e.button !== 0) return;
    e.stopPropagation();
    setSelected(id);
    const sx = e.clientX;
    const sy = e.clientY;
    const z = view.z;
    let last: Move = { id, dx: 0, dy: 0 };
    let moved = false;
    const mv = (ev: PointerEvent) => {
      last = { id, dx: (ev.clientX - sx) / z, dy: (ev.clientY - sy) / z };
      if (!moved && Math.hypot(last.dx, last.dy) > 4) moved = true;
      if (moved) setMove(last);
    };
    const up = () => {
      window.removeEventListener("pointermove", mv);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
      if (!moved) return setMove(null);
      finish(last);
    };
    window.addEventListener("pointermove", mv);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
  };
  const finish = async (m: Move) => {
    const wasReserve = reserve.some((p) => p.id === m.id);
    const startX = wasReserve ? X0 + GX * reserve.findIndex((p) => p.id === m.id) : X0 + GX * (seq.indexOf(m.id) + 1);
    const startY = wasReserve ? RES_Y : ROW_Y;
    const cx = startX + m.dx + NW / 2;
    const cy = startY + m.dy + NH / 2;
    const p = byId(m.id);
    const toRes = cy > RES_Y - 40;
    const base = (wasReserve ? [...seq] : seq.filter((id) => id !== m.id));
    const slot = Math.max(0, Math.min(base.length, Math.round((cx - NW / 2 - X0 - GX) / GX)));
    const next = [...base.slice(0, slot), m.id, ...base.slice(slot)];
    setMove(null);
    if (toRes && !wasReserve) return void ops.toReserve(p);
    if (toRes) return;
    if (wasReserve) {
      if (await ops.setDisabled(m.id, false)) void ops.commitOrder([...base.slice(0, slot), m.id, ...base.slice(slot)]);
      return;
    }
    if (next.join() === seq.join()) return;
    setOrder(next);
    void ops.commitOrder(next).then((ok) => !ok && setOrder(null));
  };

  // Клавиатура: ←/→ переносят выбранный узел по цепочке, Enter открывает меню.
  const onKey = (e: React.KeyboardEvent, id: string) => {
    const i = seq.indexOf(id);
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
      setMenu({ id, x: r.left + 24, y: r.bottom - 8 });
    } else if ((e.key === "ArrowLeft" || e.key === "ArrowRight") && i >= 0) {
      e.preventDefault();
      const j = i + (e.key === "ArrowLeft" ? -1 : 1);
      if (j < 0 || j >= seq.length) return;
      const next = [...seq];
      [next[i], next[j]] = [next[j], next[i]];
      setOrder(next);
      void ops.commitOrder(next).then((ok) => !ok && setOrder(null));
    }
  };

  const menuProvider = menu ? list.find((p) => p.id === menu.id) : undefined;
  const single = ai.single;
  const first = single ? chosenId(ai) : ai.pinned;
  const lastIdx = plan.chain.length - 1;
  const lastP = lastIdx >= 0 ? byId(plan.chain[lastIdx]) : null;
  const edges: { d: string; warn: boolean; label?: [number, number]; off: boolean }[] = [];
  const outOf = (id: string | "in") => (id === "in" ? { x: X0 + NW, y: ROW_Y + NH / 2 } : { x: posOf(id).x + NW, y: posOf(id).y + NH / 2 });
  const chainIds: (string | "in")[] = ["in", ...plan.chain];
  chainIds.forEach((id, i) => {
    const to = i + 1 < chainIds.length ? chainPos(i).x : endX;
    const a = outOf(id);
    const src = id === "in" ? null : byId(id);
    const warn = !!src && !["ok", "unknown"].includes(src.status);
    edges.push({ d: edgePath(a.x, a.y, to, ROW_Y + NH / 2), warn, off: !!src && !position.has(src.id), label: id !== "in" && i + 1 < chainIds.length ? [(a.x + to) / 2, ROW_Y + NH / 2] : undefined });
  });

  return (
    <div className="cg-wrap">
      <div className="cg-tools" role="toolbar" aria-label="Масштаб схемы">
        <button className="btn" type="button" aria-label="Приблизить" onClick={() => zoomAt(1.2, (box.current?.clientWidth ?? 600) / 2, 120)}>
          +
        </button>
        <button className="btn" type="button" aria-label="Отдалить" onClick={() => zoomAt(1 / 1.2, (box.current?.clientWidth ?? 600) / 2, 120)}>
          −
        </button>
        <button className="btn" type="button" onClick={fit}>
          По размеру
        </button>
        <span className="cg-z">{Math.round(view.z * 100)}%</span>
      </div>
      <div className="cg" ref={box} onPointerDown={onBackground} onWheel={onWheel} role="group" aria-label="Схема цепочки AI">
        <div className="cg-world" style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.z})` }}>
          <svg className="cg-edges" width={endX + NW + 80} height={RES_Y + NH + 40} aria-hidden="true">
            {edges.map((e, i) => (
              <path key={i} className={`cg-edge${e.warn ? " warn" : ""}${e.off ? " off" : ""}`} d={e.d} />
            ))}
            {lastP && (
              <path
                className="cg-edge fall"
                d={edgePath(outOf(lastP.id).x, outOf(lastP.id).y + 10, endX + NW / 2, ROW_Y + NH + 62)}
              />
            )}
          </svg>
          {pulse && <i className="cg-dot" key={pulse.k} style={{ offsetPath: `path("${pulse.d}")` }} />}

          <div className="cg-node io" style={{ left: X0, top: ROW_Y, width: NW, height: NH }} data-node="in">
            <span className="cg-port out" />
            <div className="cg-h">
              <span className="cg-ico">{Icon.up}</span>
              <b>Запрос решения</b>
            </div>
            <div className="cg-m">Бот просит AI решить, открывать ли сделку</div>
          </div>

          {[...plan.chain, ...plan.reserve].map((id) => {
            const p = byId(id);
            const at = posOf(id);
            const dragging = move?.id === id;
            const n = position.get(id);
            const [cls, text] = STATUS[p.status] ?? ["warn", p.status];
            const paused = p.cooldown_until && p.cooldown_until * 1000 > Date.now();
            const probe = ops.probes[id];
            const off = p.disabled;
            const role = single ? (id === first ? "решает" : "в запасе") : n === 1 ? "основной" : n ? "запасной" : "";
            return (
              <div
                key={id}
                data-node={id}
                tabIndex={0}
                role="button"
                aria-label={`${p.name}: ${off ? "в резерве" : `позиция ${plan.chain.indexOf(id) + 1}`}. Стрелки влево и вправо переносят, Enter — меню`}
                className={`cg-node${dragging ? " lifted" : ""}${selected === id ? " sel" : ""}${off ? " res" : ""}${ops.heir === id ? " heir" : ""}${!off && !n ? " dim" : ""}`}
                style={{ left: 0, top: 0, width: NW, height: NH, transform: `translate(${at.x + (dragging ? move!.dx : 0)}px, ${at.y + (dragging ? move!.dy : 0)}px)`, transition: dragging ? "none" : undefined }}
                onPointerDown={(e) => startMove(e, id)}
                onKeyDown={(e) => onKey(e, id)}
                onContextMenu={(e) => {
                  e.preventDefault();
                  setMenu({ id, x: e.clientX, y: e.clientY });
                }}
              >
                {!off && <span className="cg-port in" />}
                {!off && <span className="cg-port out" />}
                <div className="cg-h">
                  <Avatar id={p.id} name={p.name} />
                  <b>{p.name}</b>
                  <span className={`dot ${cls}`} title={paused ? `пауза до ${hhmm(p.cooldown_until!)}` : text} />
                  <button
                    className="cg-more"
                    type="button"
                    aria-haspopup="menu"
                    aria-label={`Действия: ${p.name}`}
                    onClick={(e) => {
                      const r = e.currentTarget.getBoundingClientRect();
                      setMenu({ id, x: r.left, y: r.bottom + 4 });
                    }}
                  >
                    {Icon.dots}
                  </button>
                </div>
                <div className="cg-m">{p.model}</div>
                <div className="cg-f">
                  {role && <span className="tag rst">{role}</span>}
                  {id === ai.pinned && <span className="tag hot">первый</span>}
                  <span className={`cg-s ${probe ? (probe.busy ? "" : probe.ok ? "ok" : "bad") : ""}`}>
                    {probe ? (probe.busy ? "проверяю…" : probe.ok ? `✓ ${probe.ms} мс` : `✗ ${probe.message}`) : paused ? `пауза до ${hhmm(p.cooldown_until!)}` : text}
                  </span>
                </div>
              </div>
            );
          })}

          <button className="cg-plus" type="button" aria-label="Добавить провайдера" title="Добавить провайдера" style={{ left: endX - 62, top: ROW_Y + NH / 2 - 15 }} onClick={onAdd}>
            +
          </button>

          <div className="cg-node io ok" style={{ left: endX, top: ROW_Y, width: NW, height: NH }} data-node="out">
            <span className="cg-port in" />
            <div className="cg-h">
              <span className="cg-ico">{Icon.check}</span>
              <b>Решение</b>
            </div>
            <div className="cg-m">Ответ первого, кто справился: открыть сделку или пропустить</div>
          </div>
          <div className={`cg-node io fall${fallbackOn ? "" : " none"}`} style={{ left: endX, top: ROW_Y + NH + 36, width: NW, height: 88 }} data-node="fall">
            <span className="cg-port in" />
            <div className="cg-h">
              <b>{fallbackOn ? "Локальные стратегии" : "Сделки нет"}</b>
            </div>
            <div className="cg-m">{fallbackOn ? "Если все провайдеры отказали, торгуют стратегии" : "Если все отказали, цикл пропускается"}</div>
          </div>

          <div className="cg-res" style={{ left: 0, top: RES_Y - 34, width: Math.max(endX + NW, X0 + GX * 3) }}>
            <span>{Icon.reserve} Резерв: перетащите сюда узел, чтобы выключить провайдера (ключ останется)</span>
          </div>
        </div>
      </div>
      <p className="cg-hint">Перетаскивайте узлы, чтобы менять порядок. Колесо — масштаб, фон — панорама. Правый щелчок или «⋯» — действия.</p>
      {menu && menuProvider && (
        <ChainMenu items={ops.items(menuProvider)} x={menu.x} y={menu.y} onClose={() => setMenu(null)} />
      )}
    </div>
  );
}

