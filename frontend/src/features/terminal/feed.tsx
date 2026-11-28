// Терминал: «Что делает бот» — что происходит сейчас, воронка последнего цикла и лента событий.

import { hhmm, plural } from "../../lib/format";
import { useStore, type FeedItem } from "../../store";
import { profileName } from "../bots";
import { NODE_TITLE } from "../trace";
import { cycleKind } from "./radar";

const NOW: Record<ReturnType<typeof cycleKind>, [string, string]> = {
  scan: ["Ищет сделки", "brand live"],
  monitor: ["Ведёт позиции", "ok live"],
  paused: ["На паузе", "warn"],
  guard: ["Сработала защита", "warn"],
  off: ["Нет связи", "bad"],
};

const STEP_KIND = (ok: boolean | null): FeedItem["kind"] => (ok ? "cycle" : "skip");

export function Feed() {
  const feed = useStore((s) => s.feed);
  const kind = useStore(cycleKind);
  const cycle = useStore((s) => s.lastCycle);
  const profile = useStore((s) => s.bot?.profile);
  const open = useStore((s) => Object.keys(s.positions).length);
  const [now, dot] = NOW[kind];
  const sub =
    kind === "scan"
      ? profile
        ? profileName(profile)
        : ""
      : kind === "monitor"
        ? `${open} ${plural(open, ["позиция", "позиции", "позиций"])}`
        : kind === "paused"
          ? "новые сделки не открываются"
          : kind === "off"
            ? "показаны последние данные"
            : "входы остановлены";
  const f = cycle?.funnel;
  // После перезагрузки ленты ещё нет — показываем шаги последнего цикла, чтобы карточка не была пустой.
  const rows: FeedItem[] = feed.length
    ? feed
    : cycle
      ? Object.values(cycle.nodes)
          .filter((s) => s!.ok !== null)
          .map((s, i) => ({ id: -i - 1, at: cycle.at * 1000, kind: STEP_KIND(s!.ok), text: `${NODE_TITLE[s!.node]}: ${s!.text}` }))
      : [];
  return (
    <article className="card feed">
      <div className="card-h">
        <h2>Что делает бот</h2>
        {cycle && <span className="feed-at">цикл {hhmm(new Date(cycle.at * 1000))}</span>}
      </div>
      <div className="feed-now">
        <span className={`dot ${dot}`} />
        <b>{now}</b>
        {sub && <span>{sub}</span>}
      </div>
      {f && (
        <ol className="funnel" aria-label="Последний цикл">
          <li>
            <b>{f.scanned}</b>монет
          </li>
          <li>
            <b>{f.candidates}</b>в отборе
          </li>
          <li className={f.signals ? "hit" : ""}>
            <b>{f.signals}</b>
            {plural(f.signals, ["сигнал", "сигнала", "сигналов"])}
          </li>
          <li className={f.opened ? "hit" : ""}>
            <b>{f.opened}</b>
            {plural(f.opened, ["сделка", "сделки", "сделок"])}
          </li>
        </ol>
      )}
      <ul aria-live="polite">
        {rows.map((r) => (
          // Реакция на исход — только у свежих записей, не при каждом открытии вкладки.
          <li key={r.id} className={`ev ev-${r.kind}${Date.now() - r.at < 2000 ? " fresh" : ""}`}>
            <time>{hhmm(new Date(r.at))}</time>
            <span className="dot" />
            <p>{r.text}</p>
          </li>
        ))}
        {!rows.length && <li className="feed-empty">Жду первого цикла…</li>}
      </ul>
    </article>
  );
}
