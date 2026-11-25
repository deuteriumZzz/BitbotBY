// Терминал: лента событий бота.

import { hhmm } from "../../lib/format";

import { useStore } from "../../store";

// ── Лента и календарь ─────────────────────────────────────────────────────────

export function Feed() {
  const feed = useStore((s) => s.feed);
  const wsUp = useStore((s) => s.wsUp);
  return (
    <article className="card feed">
      <div className="card-h">
        <h2>Что делает бот</h2>
        <span className="pill sp">
          <span className={`dot ${wsUp ? "ok live" : "bad"}`} />
          {wsUp ? "в реальном времени" : "переподключаюсь…"}
        </span>
      </div>
      <ul aria-live="polite">
        {feed.map((f) => (
          // Реакция на исход — только у свежих записей, не при каждом открытии вкладки.
          <li
            key={f.id}
            className={`ev ev-${f.kind}${Date.now() - f.at < 2000 ? " fresh" : ""}`}
          >
            <time>{hhmm(new Date(f.at))}</time>
            <span className="dot" />
            <p>{f.text}</p>
          </li>
        ))}
        {!feed.length && (
          <li className="ev ev-skip">
            <time>{hhmm()}</time>
            <span className="dot" />
            <p>Жду событий от бота…</p>
          </li>
        )}
      </ul>
    </article>
  );
}
