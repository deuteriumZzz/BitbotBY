// Сигналы для решения: что, кроме цены и индикаторов, влияет на сделку (как в SignalCombiner) и что из
// этого настраивается: ключи подключений, Chronos и пороги. Остальные сигналы рынка бесплатные и работают
// сами — без ключей и тумблеров. Дом порогов уверенности, настроения и Chronos (lib/where.ts).

import type { SecretItem } from "../lib/types";
import { plural } from "../lib/format";
import { actions, useStore } from "../store";
import { Q, Toggle } from "../ui";
import { cfg } from "./bots";
import { ParamList } from "./params";
import { FEEDS, STEP, chanList } from "./settings";

/** Фильтр рынка — 13 сигналов _apply_market_context_filter; key — источнику нужен ключ. */
export const MARKET: { name: string; rule: string; key?: string }[] = [
  { name: "Страх и жадность", rule: "≤ 15 — покупка против толпы, ≥ 85 гасит покупки" },
  { name: "Funding", rule: "> 0.1% — лонги перегреты, против покупки" },
  { name: "Открытый интерес", rule: "OI и цена падают — ликвидация лонгов" },
  { name: "Ликвидации", rule: "каскад — против входа" },
  { name: "Стакан", rule: "ask/bid > 1.4 — давление продаж" },
  { name: "Basis", rule: "премия фьючерса > 2% — жадность" },
  { name: "Опционы Deribit", rule: "PCR < 0.5 и перекос IV — все в коллах" },
  { name: "ETF-потоки", rule: "только BTC: отток < −$100M, приток > $150M" },
  { name: "Google Trends", rule: "> 75 — пик розничного FOMO" },
  { name: "Reddit · CryptoBERT", rule: "настроение постов r/CryptoCurrency", key: "reddit" },
  { name: "Стейблкоины", rule: "запас USDT падает — риск оттока", key: "glassnode" },
  { name: "Макро-календарь", rule: "FOMC, CPI, NFP рядом", key: "finnhub" },
];

/** Подключения сигналов, которые добавляются ключом, — в порядке важности для решения. */
export const SIGNAL_KEYS = ["news", "twitter", "telethon", "reddit", "glassnode", "finnhub"];

/** Единственное место, где меняется CHRONOS_ENABLED (тумблер в карточке и на узле холста). */
export const setChronos = (on: boolean) => void actions.patchConfig({ CHRONOS_ENABLED: on });

const state = (s?: SecretItem): string =>
  !s ? "—" : s.status === "ok" ? "▲ подключено" : s.status === "bad" ? "▼ не работает" : s.status === "unset" ? "не подключено" : "не проверено";

export function SignalsCard() {
  const config = useStore((s) => s.config);
  const secrets = useStore((s) => s.secrets);
  const fg = useStore((s) => s.market?.fear_greed ?? null);
  const openKey = useStore((s) => s.openKey);
  const mode = cfg<string>(config, "MODE", "ai");
  const chronos = cfg<boolean>(config, "CHRONOS_ENABLED", false);
  const chans = chanList(`${cfg(config, "TELEGRAM_SENTIMENT_CHANNELS", "")},${cfg(config, "TELEGRAM_TRUSTED_CHANNELS", "")}`).length;
  const keyed = SIGNAL_KEYS.map((id) => secrets?.find((s) => s.service === id)).filter((s): s is SecretItem => Boolean(s));
  const free = MARKET.filter((m) => !m.key);
  return (
    <article className="card sig">
      <div className="card-h">
        <h2>Сигналы для решения</h2>
        <Q tip="Новости, X и Telegram идут в промпт AI (режимы AI и Hybrid). 13 сигналов рынка — фильтр уверенности после решения в AI, SAC и Hybrid. Chronos — только Hybrid. В режиме Local решают только стратегии, соцсети читает стратегия «Настроение»." />
      </div>
      {mode === "local" && <p className="sig-note">Сейчас режим Local: решают стратегии, промпт AI, фильтр рынка и Chronos не работают. Соцсети читает стратегия «Настроение».</p>}

      <h3 className="sig-h">Подключения — добавляются ключом</h3>
      <ul className="sig-list">
        {keyed.map((s) => {
          const ok = s.service === "telethon" ? Boolean(s.signed_in) : s.status === "ok";
          return (
            <li key={s.service}>
              <div>
                <b>{s.service === "telethon" ? "Telegram-каналы" : s.name}</b>
                <small>{(FEEDS[s.service] ?? []).map(([what, step]) => (step ? `${what} → ${STEP[step]}` : what)).join(" · ")}</small>
              </div>
              <span className="sig-a">
                <span>{s.service === "telethon" ? (ok ? `▲ ${chans} ${plural(chans, ["канал", "канала", "каналов"])}` : "нужен вход") : state(s)}</span>
                <button type="button" className="btn" onClick={() => openKey(s.service)}>
                  {ok ? "Ключ →" : "Подключить →"}
                </button>
              </span>
            </li>
          );
        })}
      </ul>

      <h3 className="sig-h">Chronos — прогноз цены</h3>
      <div className="alert-row">
        <div>
          <b>Сравнивать сделку с прогнозом Chronos</b>
          <small>
            Совпал — уверенность +0.05, против — −0.15. Работает только в режиме Hybrid
            {chronos && mode !== "hybrid" ? " — сейчас другой режим, прогноз не используется" : ""}.
          </small>
        </div>
        <Toggle on={chronos} label="Прогноз Chronos" onChange={setChronos} />
      </div>

      <h3 className="sig-h">Работают сами — без ключей</h3>
      <p className="kf-h">
        {free.map((m) => m.name).join(", ")} — бесплатные источники (alternative.me, Bybit, Deribit и другие). Подключать и включать их не нужно, пороги
        заданы в коде бота: после решения в AI, SAC и Hybrid каждый сдвигает уверенность на ±0.05…0.15, крайние значения сами дают сигнал.
        {fg ? ` Индекс страха и жадности сейчас — ${fg.value} (${fg.label}).` : ""}
      </p>
      <details className="sig-more">
        <summary>Как каждый влияет</summary>
        <ul className="sig-list">
          {free.map((m) => (
            <li key={m.name}>
              <div>
                <b>{m.name}</b>
                <small>{m.rule}</small>
              </div>
            </li>
          ))}
        </ul>
      </details>

      <h3 className="sig-h">Пороги стратегии «Настроение»</h3>
      <ParamList keys={["SENTIMENT_BUY_THRESHOLD", "SENTIMENT_SELL_THRESHOLD"]} />
      <p className="sig-note">Порог уверенности сделки — в «Риск → Как рискуем → Настроить вручную → Фильтры входа».</p>
    </article>
  );
}
