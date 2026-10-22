// Командная палитра (Ctrl/⌘ K): переходы и безопасные действия с клавиатуры.
// PANIC и перезапуск сюда не входят: им нужно удержание или подтверждение.

import { useEffect, useMemo, useRef, useState } from "react";

import { actions, useStore } from "../store";

interface Cmd {
  label: string;
  hint?: string;
  keys: string;
  run: () => void;
}

export function Palette() {
  const open = useStore((s) => s.paletteOpen);
  const set = useStore((s) => s.openPalette);
  const paused = useStore((s) => s.bot?.paused ?? false);
  const [q, setQ] = useState("");
  const [at, setAt] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const cmds = useMemo<Cmd[]>(() => {
    const st = useStore.getState();
    const go = (label: string, tab: Parameters<typeof st.setTab>[0], key: string): Cmd => ({
      label: `Перейти: ${label}`,
      hint: key,
      keys: `${label} вкладка перейти открыть`,
      run: () => st.setTab(tab),
    });
    const node = (label: string, id: Parameters<typeof st.openNode>[0], keys: string): Cmd => ({
      label: `Бот → ${label}`,
      keys: `бот конвейер ${label} ${keys}`,
      run: () => st.openNode(id),
    });
    return [
      go("Терминал", "terminal", "1"),
      go("Бот", "bots", "2"),
      go("Обучение", "learn", "3"),
      go("Журнал", "journal", "4"),
      go("Настройки", "settings", "5"),
      node("Сканер", "scan", "что торговать профиль рынок спот фьючерсы сезон chronos"),
      node("Стратегии", "strat", "запасной путь закрепить"),
      node("AI-цепочка", "ai", "ai провайдеры цепочка ключ"),
      node("SAC", "sac", "нейросеть модель откат"),
      node("Риск", "risk", "риск пресет плечо слоты просадка защиты портфеля"),
      node("Ордер", "order", "подтверждение открытие сделок таймаут"),
      node("Сопровождение", "track", "выход частичный tp фиксация прибыли безубыток трейлинг"),
      { label: "Справка", hint: "?", keys: "справка помощь подсказки", run: () => st.openHelp(true) },
      { label: "Уведомления", hint: "N", keys: "уведомления события колокольчик", run: () => st.openNotif(true) },
      paused
        ? { label: "Возобновить бота", keys: "возобновить продолжить запустить старт бот", run: () => void actions.resume() }
        : { label: "Поставить бота на паузу", keys: "пауза остановить стоп бот", run: () => void actions.pause() },
    ];
  }, [paused]);
  const needle = q.trim().toLowerCase();
  const shown = cmds.filter((c) => !needle || `${c.label} ${c.keys}`.toLowerCase().includes(needle));
  // Любой текст можно отправить в поиск по параметрам — как раньше делал Ctrl/⌘ K.
  const rows: Cmd[] = needle
    ? [...shown, { label: `Искать параметр «${q.trim()}»`, keys: "", run: () => useStore.getState().searchParams(q.trim()) }]
    : shown;
  useEffect(() => {
    if (!open) return;
    setQ("");
    setAt(0);
    const back = document.activeElement as HTMLElement | null;
    setTimeout(() => input.current?.focus(), 0);
    return () => back?.focus?.();
  }, [open]);
  useEffect(() => setAt(0), [q]);
  if (!open) return null;
  const run = (c: Cmd | undefined) => {
    if (!c) return;
    set(false);
    c.run();
  };
  return (
    <div className="pal-back" onClick={() => set(false)}>
      <div className="pal" role="dialog" aria-modal="true" aria-label="Командная палитра" onClick={(e) => e.stopPropagation()}>
        <input
          ref={input}
          className="pal-in"
          placeholder="Что сделать? Например: журнал, пауза, плечо…"
          aria-label="Команда или параметр"
          autoComplete="off"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Escape") set(false);
            else if (e.key === "ArrowDown") (e.preventDefault(), setAt((a) => Math.min(a + 1, rows.length - 1)));
            else if (e.key === "ArrowUp") (e.preventDefault(), setAt((a) => Math.max(a - 1, 0)));
            else if (e.key === "Enter") (e.preventDefault(), run(rows[at]));
          }}
        />
        <ul className="pal-list" role="listbox">
          {rows.map((c, i) => (
            <li key={c.label} role="option" aria-selected={i === at} className={i === at ? "on" : ""} onMouseEnter={() => setAt(i)} onClick={() => run(c)}>
              <span>{c.label}</span>
              {c.hint && <kbd>{c.hint}</kbd>}
            </li>
          ))}
          {!rows.length && <li className="pal-empty">Ничего не нашлось</li>}
        </ul>
      </div>
    </div>
  );
}
