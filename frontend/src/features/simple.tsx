// Вкладка «Бот» в режиме «Простой»: четыре вопроса и полоса «как бот решает».
// Своих настроек здесь нет — те же карточки и переключатели, что в «Профи», только без деталей.

import { useStore } from "../store";
import { Q } from "../ui";
import { RiskCard, WhatCard, cfg } from "./bots";
import { ModeSeg, NODES, takesPart } from "./pipeline";
import { TemplateSeg, templateLine } from "./rules";
import { mark } from "./trace";
import { UiSwitchRow } from "./uimode";

export function BotSimple() {
  const c = useStore((s) => s.config);
  const ai = useStore((s) => s.ai);
  const cycle = useStore((s) => s.lastCycle);
  const setUi = useStore((s) => s.setUi);
  const mode = cfg<string>(c, "MODE", "ai");
  const chain = (ai?.chain ?? []).map((x) => ai?.providers.find((p) => p.id === x.provider)?.name ?? x.provider);
  const when = cycle ? new Date(cycle.at * 1000).toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" }) : "";
  return (
    <section className="view enter simple" aria-label="Бот">
      <UiSwitchRow />
      <WhatCard simple />
      <RiskCard simple />
      <article className="card">
        <div className="card-h">
          <h2>Кто решает</h2>
          <Q tip="Только стратегии — индикаторы, без AI: бесплатно и работает сразу. Стратегии + AI — языковая модель проверяет сигнал перед сделкой, нужен ключ провайдера." />
        </div>
        <ModeSeg simple />
        {mode === "ai" && (
          <p className="note-soon">
            {chain.length ? `AI по очереди: ${chain.join(" → ")}. Не ответит никто — решат стратегии.` : "Ключа AI ещё нет — добавьте его в «Профи» → AI-цепочка."}
          </p>
        )}
      </article>
      <article className="card">
        <div className="card-h">
          <h2>Защита</h2>
          <Q tip="Когда бот останавливает входы: дневной лимит убытка, серия убытков, предельная просадка. Подробные правила — в «Профи» → Риск." />
        </div>
        <TemplateSeg />
        <p className="note-soon">{templateLine(c)}</p>
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
        Нужны пороги, плечо, AI-цепочка или обучение?{" "}
        <button className="btn" type="button" onClick={() => setUi("pro")}>
          Открыть «Профи»
        </button>{" "}
        Всё, что выбрано здесь, там сохранится.
      </p>
    </section>
  );
}
