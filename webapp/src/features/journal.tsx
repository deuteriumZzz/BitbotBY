// Журнал сделок: закрытые сделки с причинами входа и выхода, итоги по стратегиям.

import { useEffect, useMemo, useRef, useState } from "react";

import { arrow, base, exitReason, fmtN, pct, price, tone, usd } from "../lib/format";
import type { JournalTrade, Period, Side } from "../lib/types";
import { useStore } from "../store";
import { Seg, Skel } from "../ui";
import { stratName } from "./bots";
import { Calendar, ExecutionCard } from "./terminal";
import { TraceSteps } from "./trace";

/** Выгрузка показанных сделок: открывается в Excel и Numbers. */
function exportCsv(trades: JournalTrade[]): void {
  const cols: (keyof JournalTrade)[] = ["id", "symbol", "side", "strategy", "confidence", "entry", "exit", "qty", "pnl", "pnl_pct", "fees", "opened", "closed", "reason"];
  const esc = (v: unknown) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const text = [cols.join(","), ...trades.map((t) => cols.map((c) => esc(t[c])).join(","))].join("\n");
  const url = URL.createObjectURL(new Blob(["\ufeff" + text], { type: "text/csv;charset=utf-8" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `bitbotby-journal-${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const PERIODS = [
  ["24h", "24ч"],
  ["7d", "7д"],
  ["30d", "30д"],
  ["all", "Всё"],
] as const;
const SIDES = [
  ["all", "Все"],
  ["long", "LONG"],
  ["short", "SHORT"],
] as const;
const RESULTS = [
  ["all", "Все"],
  ["win", "В плюс"],
  ["loss", "В минус"],
] as const;
const PAGE = 50;

type SideFilter = "all" | Side;
type ResultFilter = "all" | "win" | "loss";

const when = (iso: string): string =>
  new Date(iso).toLocaleString("ru-RU", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

function lasted(t: JournalTrade): string {
  const min = Math.max(0, Math.round((Date.parse(t.closed) - Date.parse(t.opened)) / 60_000));
  if (min < 60) return `${min} мин`;
  if (min < 1440) return `${Math.floor(min / 60)} ч ${min % 60} мин`;
  return `${Math.floor(min / 1440)} д ${Math.floor((min % 1440) / 60)} ч`;
}

const winRate = (t: JournalTrade[]): number => (t.length ? t.filter((x) => x.pnl > 0).length / t.length : 0);
const net = (t: JournalTrade[]): number => t.reduce((s, x) => s + x.pnl, 0);

export function Journal() {
  const journal = useStore((s) => s.journal);
  const load = useStore((s) => s.loadJournal);
  const [period, setPeriod] = useState<Period>(journal?.period ?? "30d");
  const [strategy, setStrategy] = useState("");
  const [side, setSide] = useState<SideFilter>("all");
  const [result, setResult] = useState<ResultFilter>("all");
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    setFailed(false);
    load(period).catch(() => setFailed(true));
  }, [period, load]);

  // Пока грузится новый период, старые сделки не показываем — цифры бы врали.
  const all = journal?.period === period ? journal.trades : null;
  const trades = useMemo(
    () =>
      (all ?? []).filter(
        (t) =>
          (!strategy || t.strategy === strategy) &&
          (side === "all" || t.side === side) &&
          (result === "all" || (result === "win" ? t.pnl > 0 : t.pnl <= 0)),
      ),
    [all, strategy, side, result],
  );
  const names = useMemo(() => [...new Set((all ?? []).map((t) => t.strategy))].sort(), [all]);

  return (
    <section className="view enter" aria-label="Журнал">
      <div className="cols jr">
        <div className="col">
          <article className="card jr-sum">
            <div className="card-h">
              <h2>Итоги</h2>
              <Seg className="sm sp" value={period} options={PERIODS} onChange={setPeriod} label="Период" />
            </div>
            {all && <Totals trades={trades} capped={all.length >= (journal?.limit ?? Infinity)} />}
            <div className="jr-filters">
              <select className="sel" value={strategy} onChange={(e) => setStrategy(e.target.value)} aria-label="Стратегия">
                <option value="">Все стратегии</option>
                {names.map((n) => (
                  <option key={n} value={n}>
                    {stratName(n)}
                  </option>
                ))}
              </select>
              <Seg className="sm" value={side} options={SIDES} onChange={setSide} label="Направление" />
              <Seg className="sm" value={result} options={RESULTS} onChange={setResult} label="Результат" />
            </div>
          </article>
          {all && <ByStrategy trades={trades} pick={setStrategy} picked={strategy} />}
          <Calendar />
          <ExecutionCard />
        </div>
        <div className="col">
          <article className="card jr-trades">
            <div className="card-h">
              <h2>Сделки</h2>
              {all && <span className="pill sp">{trades.length}</span>}
              {all && trades.length > 0 && (
                <button className="btn" type="button" title="Скачать показанные сделки в CSV" onClick={() => exportCsv(trades)}>
                  CSV
                </button>
              )}
            </div>
            {failed ? (
              <p className="empty">Не удалось загрузить журнал — бот не на связи.</p>
            ) : !all ? (
              <Skel rows={6} label="Загружаю журнал" />
            ) : trades.length ? (
              <TradeList key={`${period}${strategy}${side}${result}`} trades={trades} />
            ) : (
              <div className="empty">
                <p>{all.length ? "Под фильтр ничего не попало." : "За этот период закрытых сделок нет."}</p>
                {all.length > 0 ? (
                  <button
                    className="btn"
                    type="button"
                    onClick={() => {
                      setStrategy("");
                      setSide("all");
                      setResult("all");
                    }}
                  >
                    Сбросить фильтры
                  </button>
                ) : (
                  period !== "all" && (
                    <button className="btn" type="button" onClick={() => setPeriod("all")}>
                      Показать за всё время
                    </button>
                  )
                )}
              </div>
            )}
          </article>
        </div>
      </div>
    </section>
  );
}

function Totals({ trades, capped }: { trades: JournalTrade[]; capped: boolean }) {
  const total = net(trades);
  const fees = trades.reduce((s, t) => s + t.fees, 0);
  return (
    <>
      <div className="jr-tot">
        <div>
          <small>Итог</small>
          <b className={`num ${tone(total)}`}>{`${arrow(total)} ${usd(total)}`}</b>
        </div>
        <div>
          <small>Сделок</small>
          <b className="num">{trades.length}</b>
        </div>
        <div>
          <small>В плюс</small>
          <b className="num">{trades.length ? `${fmtN(winRate(trades) * 100, 0)}%` : "—"}</b>
        </div>
        <div>
          <small>Комиссии</small>
          <b className="num">{usd(-fees)}</b>
        </div>
      </div>
      {capped && <p className="jr-note">Показаны последние сделки периода — выберите период короче, чтобы увидеть все.</p>}
    </>
  );
}

function ByStrategy({ trades, pick, picked }: { trades: JournalTrade[]; pick: (s: string) => void; picked: string }) {
  const rows = useMemo(() => {
    const by = new Map<string, JournalTrade[]>();
    for (const t of trades) by.set(t.strategy, [...(by.get(t.strategy) ?? []), t]);
    return [...by].map(([name, t]) => ({ name, n: t.length, wr: winRate(t), net: net(t) })).sort((a, b) => b.net - a.net);
  }, [trades]);
  if (!rows.length) return null;
  return (
    <article className="card jr-by">
      <div className="card-h">
        <h2>По стратегиям</h2>
      </div>
      <ul className="strats">
        {rows.map((r) => (
          <li key={r.name}>
            <button
              type="button"
              className="jr-strat"
              aria-pressed={picked === r.name}
              title={picked === r.name ? "Показать все стратегии" : "Показать только эту стратегию"}
              onClick={() => pick(picked === r.name ? "" : r.name)}
            >
              <b>{stratName(r.name)}</b>
              <span>
                {r.n} · в плюс {fmtN(r.wr * 100, 0)}%
              </span>
              <b className={`num ${tone(r.net)}`}>{usd(r.net)}</b>
            </button>
          </li>
        ))}
      </ul>
    </article>
  );
}

/** Таблица сделок как в обозревателе блоков: строка — сделка, нажатие открывает её чек сбоку. */
function TradeList({ trades }: { trades: JournalTrade[] }) {
  const [shown, setShown] = useState(PAGE);
  const [t, setOpen] = useState<JournalTrade | null>(null);
  const closeBtn = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!t) return;
    closeBtn.current?.focus({ preventScroll: true });
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(null);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [t]);
  return (
    <>
      <div className="jr-head" aria-hidden="true">
        <span>ID</span>
        <span>Монета</span>
        <span>Вход → выход</span>
        <span>Результат</span>
        <span>Исход</span>
        <span>Стратегия</span>
        <span>Закрыта</span>
      </div>
      <ul className="jr-rows">
        {trades.slice(0, shown).map((x) => (
          <li key={`${x.id}-${x.closed}`}>
            <button type="button" className={`jr-tr ${tone(x.pnl)}`} onClick={() => setOpen(x)} aria-label={`Чек сделки ${x.symbol}: ${usd(x.pnl)}`}>
              <span className="jr-id num">#{x.id}</span>
              <span className="jr-coin">
                <span className="coin">{base(x.symbol)}</span>
                <b>{x.symbol}</b>
                <span className={`side ${x.side}`}>{x.side === "long" ? "LONG" : "SHORT"}</span>
              </span>
              <span className="jr-px num">
                {price(x.entry)} → {price(x.exit)}
              </span>
              <span>
                <span className={`jr-tag num ${tone(x.pnl)}`}>{usd(x.pnl)}</span>
              </span>
              <span className="jr-cell">{x.reason ? exitReason(x.reason) : "—"}</span>
              <span className="jr-cell">{stratName(x.strategy)}</span>
              <span className="jr-cell">{when(x.closed)}</span>
            </button>
          </li>
        ))}
      </ul>
      <button type="button" className={`pipe-scrim${t ? " open" : ""}`} tabIndex={-1} aria-hidden="true" onClick={() => setOpen(null)} />
      <aside className={`pipe-panel${t ? " open" : ""}`} role="dialog" aria-label="Чек сделки" aria-hidden={!t}>
        {t && (
          <>
            <div className="pipe-panel-h">
              <span className="coin">{base(t.symbol)}</span>
              <b>{t.symbol}</b>
              <span className={`side ${t.side}`}>{t.side === "long" ? "LONG" : "SHORT"}</span>
              <span className="label">чек сделки</span>
              <button ref={closeBtn} type="button" className="btn pipe-close" aria-label="Закрыть чек" onClick={() => setOpen(null)}>
                ✕
              </button>
            </div>
            <div className="pipe-body rc" key={`${t.id}-${t.closed}`}>
              <dl className="rc-g">
                <dt>ID сделки</dt>
                <dd className="num">#{t.id}</dd>
                <dt>Статус</dt>
                <dd>
                  <span className={`jr-tag ${tone(t.pnl)}`}>✓ закрыта{t.reason ? ` · ${exitReason(t.reason)}` : ""}</span>
                </dd>
                <dt>Открыта</dt>
                <dd>{when(t.opened)}</dd>
                <dt>Закрыта</dt>
                <dd>
                  {when(t.closed)} · длилась {lasted(t)}
                </dd>
              </dl>
              <dl className="rc-g">
                <dt>Направление</dt>
                <dd>{t.side === "long" ? "▲ LONG — ставка на рост" : "▼ SHORT — ставка на падение"}</dd>
                <dt>Вход → выход</dt>
                <dd className="num">
                  {price(t.entry)} → {price(t.exit)}
                </dd>
                <dt>Объём</dt>
                <dd className="num">${fmtN(t.entry * t.qty, 0)}</dd>
                <dt>Уверенность</dt>
                <dd className="num">{t.confidence.toFixed(2)}</dd>
              </dl>
              <dl className="rc-g">
                <dt>Результат</dt>
                <dd className={`num ${tone(t.pnl)}`}>
                  {usd(t.pnl)} · {pct(t.pnl_pct * 100)} от маржи
                </dd>
                <dt>Комиссии</dt>
                <dd className="num">${fmtN(t.fees)}</dd>
                <dt>Стратегия</dt>
                <dd>{stratName(t.strategy)}</dd>
                <dt>Почему открыта</dt>
                <dd>{t.reasoning || "бот не записал объяснение"}</dd>
                <dt>Почему закрыта</dt>
                <dd>{t.reason ? exitReason(t.reason) : "причина не записана (сделка до обновления)"}</dd>
              </dl>
              <div className="rc-steps">
                <h3>Как бот решил</h3>
                {t.trace?.length ? <TraceSteps steps={t.trace} /> : <p className="label">Шагов нет — сделка закрыта до обновления.</p>}
              </div>
            </div>
          </>
        )}
      </aside>
      {trades.length > shown && (
        <button className="btn jr-more-btn" type="button" onClick={() => setShown(shown + PAGE)}>
          Показать ещё {Math.min(PAGE, trades.length - shown)}
        </button>
      )}
    </>
  );
}
