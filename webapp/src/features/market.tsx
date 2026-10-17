// Сезон рынка и индекс Страха/Жадности — то же, что показывает Telegram в /status.
// Данные бот загружает сам (CoinGecko и Fear & Greed), приложение только показывает.

import { useEffect, useState } from "react";

import { fmtN, pct } from "../lib/format";
import type { FearGreed, Season } from "../lib/types";
import { useStore } from "../store";
import { Q, Tween } from "../ui";

const ZONES: [number, number, string, string][] = [
  [0, 25, "Крайний страх", "var(--down)"],
  [25, 45, "Страх", "var(--brand)"],
  [45, 55, "Нейтрально", "var(--ink-3)"],
  [55, 75, "Жадность", "color-mix(in oklab, var(--up) 70%, var(--ink-3))"],
  [75, 100, "Крайняя жадность", "var(--up)"],
];

const zoneOf = (v: number) =>
  ZONES.find(([a, b]) => v >= a && v < b) ?? ZONES[ZONES.length - 1];

const R = 80;
const pt = (p: number, r = R): string => {
  const a = Math.PI * (1 - p / 100);
  return `${(100 + r * Math.cos(a)).toFixed(2)} ${(100 - r * Math.sin(a)).toFixed(2)}`;
};

/** Полукруг 0–100 с пятью зонами; стрелка плавно доезжает до значения. */
function FearGreedGauge({ data }: { data: FearGreed }) {
  const v = Math.max(0, Math.min(100, data.value));
  const [shown, setShown] = useState(0);
  useEffect(() => {
    const raf = requestAnimationFrame(() => setShown(v));
    return () => cancelAnimationFrame(raf);
  }, [v]);
  const [, , name, color] = zoneOf(v);
  return (
    <div className="fg">
      <svg
        className="fg-svg"
        viewBox="0 0 200 112"
        role="img"
        aria-label={`Страх и жадность: ${Math.round(v)} из 100, ${name}`}
      >
        {ZONES.map(([a, b, , c]) => (
          <path
            key={a}
            d={`M ${pt(a + 0.6)} A ${R} ${R} 0 0 1 ${pt(b - 0.6)}`}
            className="fg-z"
            style={{ stroke: c }}
          />
        ))}
        <line
          className="fg-n"
          x1="100"
          y1="100"
          x2="100"
          y2="34"
          style={{ transform: `rotate(${-90 + 1.8 * shown}deg)` }}
        />
        <circle cx="100" cy="100" r="5" className="fg-c" />
      </svg>
      <div className="fg-val">
        <Tween
          value={v}
          fmt={(x) => String(Math.round(x))}
          className="fg-num"
        />
        <b style={{ color }}>{name}</b>
      </div>
    </div>
  );
}

const SEASON_NAME = { altcoin: "Альтсезон", bluechip: "Сезон BTC" } as const;

function SeasonInfo({ data }: { data: Season }) {
  const idx = data.altcoin_index;
  const name = data.signal ? SEASON_NAME[data.signal] : "Нейтрально";
  const at = new Date(data.checked_at * 1000).toLocaleTimeString("ru-RU", {
    hour: "2-digit",
    minute: "2-digit",
  });
  return (
    <div className="season">
      <b className={`season-name ${data.signal ?? ""}`}>{name}</b>
      {idx != null && (
        <>
          <div
            className="season-bar"
            role="img"
            aria-label={`Индекс альтсезона ${Math.round(idx)} из 100`}
          >
            <i style={{ width: `${Math.max(0, Math.min(100, idx))}%` }} />
          </div>
          <span className="note-soon">
            Индекс альтсезона {Math.round(idx)} из 100: доля альткоинов из
            топ-100, которые за 30 дней выросли сильнее BTC. От 75 — альтсезон,
            до 25 — сезон BTC.
          </span>
        </>
      )}
      <dl className="season-kv">
        {data.btc_dominance != null && (
          <div>
            <dt>Доминация BTC</dt>
            <dd>{fmtN(data.btc_dominance, 1)}%</dd>
          </div>
        )}
        {data.btc_30d != null && (
          <div>
            <dt>BTC за 30 дней</dt>
            <dd>{pct(data.btc_30d, 1)}</dd>
          </div>
        )}
      </dl>
      <span className="note-soon">обновлено в {at}</span>
    </div>
  );
}

export function MarketPulse() {
  const market = useStore((s) => s.market);
  return (
    <article className="card pulse">
      <div className="pulse-col">
        <div className="card-h">
          <span className="label">
            Страх и жадность{" "}
            <Q tip="Индекс настроения крипторынка от 0 (крайний страх) до 100 (крайняя жадность). В страхе рынок часто перепродан, в жадности — перегрет; сам по себе сделок он не открывает." />
          </span>
        </div>
        {market?.fear_greed ? (
          <FearGreedGauge data={market.fear_greed} />
        ) : (
          <p className="note-soon">
            Данные загружаются — бот обновляет индекс раз в несколько часов.
          </p>
        )}
      </div>
      <div className="pulse-col">
        <div className="card-h">
          <span className="label">
            Сезон рынка{" "}
            <Q tip="Альтсезон — когда большинство крупных альткоинов растёт быстрее BTC; сезон BTC — наоборот. Данные CoinGecko, бот проверяет их несколько раз в день." />
          </span>
        </div>
        {market?.season ? (
          <SeasonInfo data={market.season} />
        ) : (
          <p className="note-soon">
            Данные загружаются — бот проверяет сезон раз в несколько часов.
          </p>
        )}
      </div>
    </article>
  );
}
