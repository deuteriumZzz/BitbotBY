// Сигналы для решения: что, кроме цены и индикаторов, видит тот, кто решает, и с какой уверенности
// сделка идёт дальше в «Риск». Дом порогов уверенности и настроения (lib/where.ts).

import type { SecretItem, SettingDef } from "../lib/types";
import { plural } from "../lib/format";
import { actions, useStore } from "../store";
import { Q } from "../ui";
import { cfg } from "./bots";
import { FEEDS, STEP, chanList } from "./settings";

const state = (s?: SecretItem): string =>
  !s ? "—" : s.status === "ok" ? "▲ подключено" : s.status === "bad" ? "▼ не работает" : s.status === "unset" ? "не подключено" : "не проверено";

export function SignalsCard() {
  const config = useStore((s) => s.config);
  const schema = useStore((s) => s.schema);
  const secrets = useStore((s) => s.secrets);
  const paper = useStore((s) => s.bot?.paper ?? true);
  const chans = chanList(`${cfg(config, "TELEGRAM_SENTIMENT_CHANNELS", "")},${cfg(config, "TELEGRAM_TRUSTED_CHANNELS", "")}`).length;
  // Подключения — те же, что в «Настройки → Ключи и доступ»; куда каждое ведёт — карта FEEDS.
  const rows: [string, string, string][] = [
    ...(secrets ?? [])
      .filter((s) => s.group === "Данные и соцсети")
      .map((s): [string, string, string] => [
        s.service === "telethon" ? "Telegram-каналы" : s.name,
        (FEEDS[s.service] ?? []).map(([what, step]) => (step ? `${what} → ${STEP[step]}` : what)).join(" · "),
        s.service === "telethon" ? (s.signed_in ? `▲ ${chans} ${plural(chans, ["канал", "канала", "каналов"])}` : "нужен вход") : state(s),
      ]),
    [
      "Рынок без ключей",
      "Страх/Жадность, funding, открытый интерес, ликвидации, стакан, basis, опционы, ETF (BTC), Google Trends → фильтр уверенности: каждый сдвигает её на ±0.05…0.15",
      "▲ работает",
    ],
  ];
  const keys = paper
    ? ["MIN_SIGNAL_CONFIDENCE_PAPER", "MIN_SIGNAL_CONFIDENCE", "SENTIMENT_BUY_THRESHOLD", "SENTIMENT_SELL_THRESHOLD"]
    : ["MIN_SIGNAL_CONFIDENCE", "MIN_SIGNAL_CONFIDENCE_PAPER", "SENTIMENT_BUY_THRESHOLD", "SENTIMENT_SELL_THRESHOLD"];
  const params = keys.map((k) => schema.find((d) => d.key === k)).filter((d): d is SettingDef => Boolean(d));
  const now = paper ? "MIN_SIGNAL_CONFIDENCE_PAPER" : "MIN_SIGNAL_CONFIDENCE";
  return (
    <article className="card sig">
      <div className="card-h">
        <h2>Сигналы для решения</h2>
        <Q tip="Кроме цены и индикаторов тот, кто решает (AI или стратегии), видит новости, соцсети и рынок. Сделка идёт дальше в «Риск», только если её уверенность не ниже порога. Доверенные каналы Telegram сюда не входят — у них свой путь сразу в ордер." />
      </div>
      <p className="kf-h">
        В AI уходят новости (выше 0.3 — в пользу покупки, ниже −0.3 — продажи) и соцсети: оценка от −1 до 1 и сколько раз монету упомянули в свежих
        постах; нет данных — не «нейтрально». Telegram работает в двух режимах: все каналы — сигнал здесь, доверенные — ещё и сделка по посту в обход
        этого пути.
      </p>
      <ul className="sig-list">
        {rows.map(([name, to, st]) => (
          <li key={name}>
            <div>
              <b>{name}</b>
              <small>{to}</small>
            </div>
            <span>{st}</span>
          </li>
        ))}
      </ul>
      <div className="tgch-params">
        {params.map((d) => (
          <label className="cool" key={d.key}>
            <span>
              {d.label}
              {d.key === now && " · сейчас"}
            </span>
            <span className="numf">
              <input
                key={String(config[d.key]?.value)}
                type="number"
                inputMode="decimal"
                step={d.step ?? 0.01}
                min={d.min ?? undefined}
                max={d.max ?? undefined}
                defaultValue={String(config[d.key]?.value ?? d.default)}
                onBlur={(e) => {
                  const v = Number(e.target.value.replace(",", "."));
                  if (e.target.value && Number.isFinite(v) && v !== config[d.key]?.value) void actions.patchConfig({ [d.key]: v });
                }}
                onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
              />
            </span>
          </label>
        ))}
      </div>
    </article>
  );
}
