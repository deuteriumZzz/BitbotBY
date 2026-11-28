// Вкладка «Бот» в режиме «Простой»: четыре вопроса и полоса «как бот решает».
// Своих настроек здесь нет — те же карточки и переключатели, что в «Профи», только без деталей.

import { useStore } from "../store";
import { Q } from "../components/tooltip";
import { RiskCard, WhatCard, cfg } from "./bots";
import { ModeSeg, NODES, takesPart } from "./pipeline";
import { templateLine } from "./rules";
import { mark } from "./trace";
import { protectionLines } from "../lib/format";
import { KeysCard, useScoreboard } from "./ai";
import { UiSwitchRow } from "./uimode";

export function BotSimple() {
  const c = useStore((s) => s.config);
  const ai = useStore((s) => s.ai);
  const secrets = useStore((s) => s.secrets);
  const cycle = useStore((s) => s.lastCycle);
  const setUi = useStore((s) => s.setUi);
  const ops = useStore((s) => s.opsStatus);
  const score = useScoreboard();
  const mode = cfg<string>(c, "MODE", "ai");
  const fallback = cfg<boolean>(c, "LOCAL_FALLBACK", false);
  const chain = ai?.chain ?? [];
  const when = cycle ? new Date(cycle.at * 1000).toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" }) : "";
  return (
    <section className="view enter simple" aria-label="Бот">
      <UiSwitchRow />
      <WhatCard />
      <RiskCard simple />
      <article className="card">
        <div className="card-h">
          <h2>Кто решает</h2>
          <Q tip="Стратегии — индикаторы, без AI: бесплатно и работает сразу. AI — языковая модель проверяет сигнал перед сделкой, нужен ключ." />
        </div>
        <ModeSeg simple />
        {mode === "ai" && chain.length > 0 && (
          <p className="note-soon">
            {fallback ? "Если AI не ответит, решат стратегии." : "Если AI не ответит, сделки не будет."}
          </p>
        )}
        {mode === "ai" && ai && <KeysCard ai={ai} secrets={secrets ?? []} onSaved={() => undefined} />}
        {mode === "ai" && score && Object.keys(score.providers).length > 0 && (
          <p className="note-soon">
            Честность прогнозов:{" "}
            {Object.entries(score.providers)
              .map(([id, k]) => `${ai?.providers.find((p) => p.id === id)?.name ?? id} — ${k.forecast?.verdict ?? "нет данных"}`)
              .join(", ")}
            .
          </p>
        )}
      </article>
      <article className="card">
        <div className="card-h">
          <h2>Защита</h2>
          <Q tip="Когда бот останавливает входы: дневной лимит убытка, серия убытков, предельная просадка. Пороги ставит выбор «Как рискуем» выше." />
        </div>
        <p className="note-soon">{templateLine(c)}</p>
        {ops ? (
          <ul className="prot-list">
            {protectionLines(ops).map((p) => (
              <li key={p.key}>
                <span>
                  {p.label} <Q tip={p.hint} />
                </span>
                <span className={`pill sp ${p.state === "ok" ? "ok" : ""}`}>{p.text}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="note-soon">Остальные защиты появятся здесь после первого цикла бота.</p>
        )}
      </article>
      <article className="card simple-strip">
        <div className="card-h">
          <h2>Как бот решает</h2>
          <span className="pill sp">{when ? `последний цикл ${when}` : "цикла ещё не было"}</span>
        </div>
        <ol>
          {NODES.map((n) => {
            const res = cycle?.nodes[n.id];
            const [glyph, cls] = mark(res?.ok);
            return (
              <li key={n.id} className={takesPart(mode, n.id) ? "" : "idle"}>
                <b>{n.title}</b>
                <span className={`res ${cls}`}>
                  <i>{glyph}</i> {res?.text ?? (takesPart(mode, n.id) ? "—" : "не участвует")}
                </span>
              </li>
            );
          })}
        </ol>
      </article>
      <p className="note-soon simple-more">
        Нужны пороги, плечо или обучение?{" "}
        <button className="btn" type="button" onClick={() => setUi("pro")}>
          Открыть «Профи»
        </button>{" "}
        Всё, что выбрано здесь, там сохранится.
      </p>
    </section>
  );
}
