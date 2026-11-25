// Центр уведомлений: колокольчик с числом новых событий и окно со списком.
// Данные — лента «Что делает бот» из стора; ничего нового с сервера не запрашивается.

import { useEffect, useRef } from "react";

import { hhmm } from "../lib/format";
import { useStore } from "../store";

/** Какие события считаем «важными» для счётчика: сделки и защиты, а не каждый пропуск. */
const LOUD = new Set(["open", "win", "loss", "guard"]);

export function NotifButton() {
  const feed = useStore((s) => s.feed);
  const seen = useStore((s) => s.seenFeed);
  const open = useStore((s) => s.notifOpen);
  const set = useStore((s) => s.openNotif);
  const fresh = feed.filter((f) => f.id > seen && LOUD.has(f.kind)).length;
  return (
    <button
      className="help-btn bell"
      type="button"
      aria-label={fresh ? `Уведомления: ${fresh} новых` : "Уведомления"}
      aria-haspopup="dialog"
      aria-expanded={open}
      title="Уведомления"
      onClick={() => set(true)}
    >
      <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M6 9a6 6 0 0 1 12 0c0 6 2 7 2 7H4s2-1 2-7" />
        <path d="M10 20a2 2 0 0 0 4 0" />
      </svg>
      {fresh > 0 && <i className="bell-n">{fresh > 9 ? "9+" : fresh}</i>}
    </button>
  );
}

export function NotifSheet() {
  const open = useStore((s) => s.notifOpen);
  const set = useStore((s) => s.openNotif);
  const feed = useStore((s) => s.feed);
  const close = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    const back = document.activeElement as HTMLElement | null;
    close.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && set(false);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      back?.focus?.();
    };
  }, [open, set]);
  if (!open) return null;
  return (
    <div className="help-back" onClick={() => set(false)}>
      <aside className="help-sheet" role="dialog" aria-modal="true" aria-label="Уведомления" onClick={(e) => e.stopPropagation()}>
        <div className="help-h">
          <h2>Уведомления</h2>
          <button ref={close} className="btn" type="button" aria-label="Закрыть уведомления" onClick={() => set(false)}>
            ✕
          </button>
        </div>
        <ul className="notif-list">
          {feed.map((f) => (
            <li key={f.id} className={`ev ev-${f.kind}`}>
              <time>{hhmm(new Date(f.at))}</time>
              <span className="dot" />
              <p>{f.text}</p>
            </li>
          ))}
          {!feed.length && <li className="note-soon">Пока тихо: события появятся, когда бот начнёт работать.</li>}
        </ul>
      </aside>
    </div>
  );
}
