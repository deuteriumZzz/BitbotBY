// Терминал: метрики торговли и качество исполнения (проскальзывание, задержка).

import { useEffect, useState } from "react";

import { fmtN, plural, tone, usd } from "../../lib/format";

import { useStore } from "../../store";

import { Tween } from "../../components/motion";
import { Q } from "../../components/tooltip";

// ── Метрики ───────────────────────────────────────────────────────────────────

export function MetricsCard() {
  const m = useStore((s) => s.metrics[s.period]);
  // На телефоне видны две главные метрики, остальные раскрываются кнопкой.
  const [open, setOpen] = useState(false);
  if (!m) return <article className="card metrics" aria-busy="true" />;
  const wr = m.win_rate ?? 0;
  const aw = m.avg_win ?? 0;
  const al = Math.abs(m.avg_loss ?? 0);
  const tot = aw + al || 1;
  return (
    <article className={`card metrics${open ? " open" : ""}`}>
      <div className="tile">
        <div className="t-h">
          <span>Доля прибыльных</span>
          <Q tip="Win rate — сколько сделок закрылись в плюс. 40% — нормально, если прибыльные сделки крупнее убыточных." />
        </div>
        <div className="v">
          {m.win_rate === null ? (
            "—"
          ) : (
            <Tween value={wr * 100} fmt={(v) => `${fmtN(v, 1)}%`} />
          )}
        </div>
        <div className="bar">
          <i style={{ background: "var(--up)", width: `${wr * 100}%` }} />
        </div>
        <small>
          {Math.round(wr * m.n)} из {m.n}{" "}
          {plural(m.n, ["сделки", "сделок", "сделок"])}
        </small>
      </div>
      <div className="tile">
        <div className="t-h">
          <span>Профит-фактор</span>
          <Q tip="Сумма всех прибылей, делённая на сумму всех убытков. Больше 1 — в плюсе, меньше 1 — в минусе." />
        </div>
        <div
          className={`v ${m.profit_factor === null ? "" : m.profit_factor >= 1 ? "up" : "down"}`}
        >
          {m.profit_factor === null ? (
            "—"
          ) : (
            <Tween value={m.profit_factor} fmt={(v) => fmtN(v, 2)} />
          )}
        </div>
        <small>
          {m.profit_factor === null
            ? "нет убыточных сделок"
            : m.profit_factor >= 1
              ? "прибыли больше убытков"
              : "убытки больше прибылей"}
        </small>
      </div>
      <div className="tile">
        <div className="t-h">
          <span>Средняя сделка</span>
          <Q tip="Средний размер прибыльной и убыточной сделки. Полоски показывают их соотношение." />
        </div>
        <div className="v small">
          <span className="up">
            {m.avg_win === null ? "—" : usd(aw)}
          </span>{" "}
          /{" "}
          <span className="down">
            {m.avg_loss === null ? "—" : usd(-al)}
          </span>
        </div>
        <div className="bar">
          <i
            style={{ background: "var(--up)", width: `${(aw / tot) * 100}%` }}
          />
          <i
            style={{ background: "var(--down)", width: `${(al / tot) * 100}%` }}
          />
        </div>
      </div>
      <div className="tile">
        <div className="t-h">
          <span>Риск : прибыль</span>
          <Q tip="R:R — во сколько раз средняя прибыльная сделка больше средней убыточной." />
        </div>
        <div className="v">
          {m.rr === null ? (
            "—"
          ) : (
            <Tween value={m.rr} fmt={(v) => `1 : ${fmtN(v, 2)}`} />
          )}
        </div>
        <small>прибыль на 1 $ риска</small>
      </div>
      <div className="tile">
        <div className="t-h">
          <span>Макс. просадка</span>
          <Q tip="Самое глубокое падение баланса от пика до дна за период. Столько бот уже терял в худшую полосу." />
        </div>
        <div className="v down">
          <Tween value={m.max_drawdown_pct} fmt={(v) => `−${fmtN(v, 1)}%`} />
        </div>
        <small>от пика баланса</small>
      </div>
      <div className="tile">
        <div className="t-h">
          <span>Коэф. Шарпа</span>
          <Q tip="Доходность с поправкой на её разброс, по дням, включая дни без сделок. Считаем от 50 закрытых сделок: на меньшем числе цифра случайна." />
        </div>
        {m.sharpe === null ? (
          <>
            <div className="v small muted">мало данных</div>
            <small>{m.n} из 50 сделок, нужных для расчёта</small>
          </>
        ) : (
          <>
            <div className={`v ${tone(m.sharpe)}`}>
              <Tween value={m.sharpe} fmt={(v) => fmtN(v, 2)} />
            </div>
            <small>по {m.sharpe_days ?? m.n} {m.sharpe_days ? "дням" : "сделкам"}</small>
          </>
        )}
      </div>
      <button
        className="btn metrics-more"
        type="button"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
      >
        {open ? "Скрыть метрики" : "Ещё 4 метрики"}
      </button>
    </article>
  );
}

// ── Качество исполнения: задержка ордера и проскальзывание ────────────────────

/** Качество исполнения — живёт в «Журнале». */
export function ExecutionCard() {
  const e = useStore((s) => s.execution[s.period]);
  const period = useStore((s) => s.period);
  // Грузим только здесь: на «Обзоре» эти цифры не показываются.
  useEffect(() => {
    if (!e) void useStore.getState().loadExecution(period).catch(() => undefined);
  }, [period]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!e) return <article className="card exec" aria-busy="true" />;
  const few = e.latency_ms_avg === null || e.slippage_pct_avg === null;
  return (
    <article className="card exec">
      <div className="card-h">
        <h2>Качество исполнения</h2>
        <Q tip="Задержка от отправки ордера до исполнения и насколько цена исполнения хуже цены сигнала. Считаем от 20 сделок." />
      </div>
      {few ? (
        <p className="muted">
          Мало данных: {e.n} из {e.min_n}{" "}
          {plural(e.min_n, ["сделки", "сделок", "сделок"])}, нужных для расчёта.
        </p>
      ) : (
        <div className="exec-rows">
          <div className="exec-row">
            <span className="label">Задержка ордера</span>
            <b>{fmtN(e.latency_ms_avg ?? 0, 0)} мс</b>
            <small>в среднем, от отправки до исполнения</small>
          </div>
          <div className="exec-row">
            <span className="label">Проскальзывание</span>
            <b className={tone(-(e.slippage_pct_avg ?? 0))}>
              {fmtN(e.slippage_pct_avg ?? 0, 3)}%
            </b>
            <small>
              в среднем, плюс — хуже цены сигнала; худшая сделка{" "}
              {fmtN(e.slippage_pct_worst ?? 0, 3)}%
            </small>
          </div>
          <small className="muted">
            {e.simulated
              ? `Paper: исполнение симулировано, по ${e.n} сделкам.`
              : `По ${e.n} сделкам.`}
          </small>
        </div>
      )}
    </article>
  );
}
