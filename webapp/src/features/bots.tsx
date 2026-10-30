// Карточки вкладки «Бот»: что торговать, стратегии, режим риска, тонкая настройка, открытие сделок.
// По узлам конвейера их раскладывает features/pipeline.tsx.

import { useEffect, useLayoutEffect, useRef, useState } from "react";

import { fmtN, plural, tone } from "../lib/format";
import { reducedMotion } from "../lib/platform";
import type { ConfigValues, RiskPreset, Strategy } from "../lib/types";
import { actions, useStore } from "../store";
import { Icon, Q, Seg, Toggle } from "../ui";

export const STRATEGY_NAME: Record<string, string> = {
  scalping: "Скальпинг",
  sentiment: "Настроение соцсетей",
  amd: "AMD: накопление → выброс",
  fibonacci_retracement: "Откат по Фибоначчи",
  liquidity_sweep: "Снятие ликвидности",
  fair_value_gap: "Fair Value Gap",
  bollinger_bands: "Полосы Боллинджера",
  swing_trading: "Свинг",
  cvd_divergence: "Дивергенция CVD",
  trend_following: "По тренду",
  support_resistance: "Поддержка и сопротивление",
  // Профильные варианты слиты в одну стратегию; имена остаются в истории сделок.
  volume_spike_meme: "Всплеск объёма",
  momentum_burst: "Импульс",
  inverse_fair_value_gap: "Обратный FVG",
  rsi_momentum: "RSI-моментум",
  ema_crossover: "Пересечение EMA",
  breakout: "Пробой",
  mean_reversion: "Возврат к среднему",
  volume_spike_altcoin: "Всплеск объёма",
  volume_spike_bluechip: "Всплеск объёма",
  volume_spike: "Всплеск объёма",
  macd_crossover: "Пересечение MACD",
  pump_exit: "Выход из пампа",
  supertrend: "SuperTrend",
  dual_thrust: "Dual Thrust",
  ttm_squeeze: "TTM Squeeze",
  adx_trend_birth: "Рождение тренда (ADX)",
  vwap_reversion: "Возврат к VWAP",
  crash_bounce: "Отскок после обвала",
  bear_hedge: "Медвежий шорт",
  ichimoku: "Ишимоку",
  parabolic_sar: "Parabolic SAR",
  vortex: "Vortex",
  stoch_rsi: "StochRSI",
  williams_r: "Williams %R",
  keltner_fade: "Кельтнер: возврат",
  obv_trend: "OBV-тренд",
  mfi_trend: "MFI-тренд",
  chaikin_osc: "Осциллятор Чайкина",
  coppock: "Кривая Коппока",
  coral_trend: "Coral Trend",
  kst: "KST",
  rvi: "RVI",
  waddah_attar: "Waddah Attar",
  zlsma: "ZLSMA",
  cci_aroon: "CCI + Aroon",
  engulfing_trend: "Поглощение по тренду",
  chaikin_money_flow: "Chaikin Money Flow",
  fisher_transform: "Fisher Transform",
  kdj: "KDJ",
  klinger: "Klinger",
  awesome_saucer: "Awesome Oscillator",
  heikin_ashi: "Heikin Ashi",
  rsi_divergence: "Дивергенция RSI",
  shooting_star: "Падающая звезда",
  cci_reversion: "CCI: возврат",
  ifr2: "IFR2",
  stochastic: "Стохастик",
  adx_dmi: "ADX/DMI",
  chandelier: "Chandelier Exit",
  halftrend: "HalfTrend",
  ssl_channel: "SSL Channel",
  ad_line_trend: "Линия A/D",
  ease_of_movement: "Ease of Movement",
  elder_force: "Индекс силы Элдера",
  vwap_trend: "VWAP-тренд",
  vw_macd: "MACD по VWMA",
  poc_reversion: "Возврат к POC",
  bollinger_reentry: "Возврат в Боллинджер",
  dual_momentum: "Двойной моментум",
  gtaa_ma: "Тайминг по SMA",
  spike_reversal: "Разворот после всплеска",
};
export const stratName = (k: string): string => STRATEGY_NAME[k] ?? k;

export const cfg = <T,>(c: ConfigValues, key: string, fallback: T): T => (c[key]?.value as T) ?? fallback;

type LevMode = RiskPreset["LEVERAGE_MODE"];
const ico = (d: string) => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d={d} />
  </svg>
);
/** Подписи пресетов. Сами значения — из бота (GET /api/risk/presets), одни на приложение и Telegram. */
const PRESET_META: Record<string, { name: string; d: string; icon: JSX.Element }> = {
  conservative: {
    name: "Консервативный",
    d: "Редкие сделки, маленький риск, строгая защита",
    icon: ico("M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"),
  },
  moderate: {
    name: "Умеренный",
    d: "Баланс между частотой сделок и риском",
    icon: ico("M12 3v18M7 21h10M3 7h18M6 7l-3 7a3 3 0 0 0 6 0zM18 7l-3 7a3 3 0 0 0 6 0z"),
  },
  aggressive: {
    name: "Агрессивный",
    d: "Больше сделок и риска, без защиты от просадки",
    icon: ico(
      "M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09zM12 15l-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2zM9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5",
    ),
  },
};

const PROFILES = [
  ["bluechip", "Bluechip"],
  ["altcoin", "Altcoin"],
  ["meme", "Meme"],
] as const;
export const profileName = (p: string): string => PROFILES.find(([v]) => v === p)?.[1] ?? p;
const PROFILE_NOTE: Record<string, string> = {
  bluechip: "Крупнейшие монеты: BTC, ETH, SOL и др. Спокойнее, меньше сделок.",
  altcoin: "Ликвидные альткоины. Больше движений и сигналов.",
  meme: "Мем-монеты и сигналы из Twitter и памп-каналов Telegram. Самый рискованный.",
};

// ── Как рискуем: пресеты + ручная настройка ───────────────────────────────────

/** Ключ пресета риска, совпадающего с настройками; undefined — свои настройки. */
export function presetOf(c: ConfigValues, presets: Record<string, RiskPreset>): string | undefined {
  const now: RiskPreset = {
    MAX_POSITIONS: cfg(c, "MAX_POSITIONS", 3),
    RISK_PER_TRADE: cfg(c, "RISK_PER_TRADE", 0.02),
    DRAWDOWN_SCALE_ENABLED: cfg(c, "DRAWDOWN_SCALE_ENABLED", true),
    LEVERAGE_MODE: cfg<LevMode>(c, "LEVERAGE_MODE", "volatility"),
  };
  return Object.entries(presets).find(
    ([, p]) =>
      p.MAX_POSITIONS === now.MAX_POSITIONS &&
      Math.abs(p.RISK_PER_TRADE - now.RISK_PER_TRADE) < 1e-9 &&
      p.DRAWDOWN_SCALE_ENABLED === now.DRAWDOWN_SCALE_ENABLED &&
      p.LEVERAGE_MODE === now.LEVERAGE_MODE,
  )?.[0];
}

export const presetName = (k: string | undefined): string => (k ? (PRESET_META[k]?.name ?? k) : "свои настройки");

/** Значение слайдера меняется сразу, а на сервер уходит через полсекунды после последнего движения. */
function useDebouncedPatch<T>(server: T, key: string, toServer: (v: T) => unknown = (v) => v): [T, (v: T) => void] {
  const [local, setLocal] = useState(server);
  const timer = useRef<ReturnType<typeof setTimeout>>();
  useEffect(() => setLocal(server), [server]);
  useEffect(() => () => clearTimeout(timer.current), []);
  const set = (v: T) => {
    setLocal(v);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => void actions.patchConfig({ [key]: toServer(v) }), 500);
  };
  return [local, set];
}

/**
 * Единственное место, где меняются сделки одновременно, риск на сделку и плечо.
 * «Уменьшать при просадке» — в «Правилах защиты» (пресет ставит и его).
 */
/** simple — режим «Простой»: только пресеты, без ручной настройки. */
export function RiskCard({ simple = false }: { simple?: boolean }) {
  const c = useStore((s) => s.config);
  const presets = useStore((s) => s.riskPresets);
  const balance = useStore((s) => s.balance);
  const paper = useStore((s) => s.bot?.paper ?? true);
  const active = presetOf(c, presets);
  const [open, setOpen] = useState(false);
  const manual = !simple && (open || (active === undefined && Object.keys(presets).length > 0));
  const spot = cfg<string>(c, "MARKET_TYPE", "linear") === "spot";
  const lev = cfg<LevMode>(c, "LEVERAGE_MODE", "volatility");
  const levText = spot ? "1×" : lev === "fixed" ? `${cfg(c, "LEVERAGE", 3)}×` : `до ${cfg(c, "LEVERAGE_MAX", 5)}×`;
  const [risk, setRisk] = useDebouncedPatch(+(cfg(c, "RISK_PER_TRADE", 0.02) * 100).toFixed(2), "RISK_PER_TRADE", (v) => +(v / 100).toFixed(4));
  const [slots, setSlots] = useDebouncedPatch(cfg(c, "MAX_POSITIONS", 3), "MAX_POSITIONS");
  const one = (balance * risk) / 100;
  return (
    <article className="card">
      <div className="card-h">
        <h2>Как рискуем</h2>
        <span className="pill sp">{active ? presetName(active) : simple ? "Свои настройки (Профи)" : "Свои настройки"}</span>
      </div>
      <div className="presets">
        {Object.entries(PRESET_META).map(([k, m]) => {
          const p = presets[k];
          return (
            <button key={k} className="preset" type="button" aria-pressed={k === active} disabled={!p} onClick={() => p && k !== active && void actions.patchConfig({ ...p })}>
              <span className="pi">{m.icon}</span>
              <b>{m.name}</b>
              <span className="pd">{m.d}</span>
              {p && (
                <dl>
                  <div>
                    <dt>Сделок сразу</dt>
                    <dd>{p.MAX_POSITIONS}</dd>
                  </div>
                  <div>
                    <dt>Риск на сделку</dt>
                    <dd>{+(p.RISK_PER_TRADE * 100).toFixed(2)}%</dd>
                  </div>
                  <div>
                    <dt>Защита просадки</dt>
                    <dd>{p.DRAWDOWN_SCALE_ENABLED ? "вкл" : "выкл"}</dd>
                  </div>
                </dl>
              )}
              {Icon.check}
            </button>
          );
        })}
      </div>
      <p className="preview">
        При балансе <b>${fmtN(balance, 0)}</b> одна сделка рискует до <b>${fmtN(one, 0)}</b>. Открыто может быть до <b>{slots}</b>{" "}
        {plural(slots, ["сделки", "сделок", "сделок"])} сразу — в худшем случае до <b>${fmtN(one * slots, 0)}</b>.
      </p>
      {!simple && (
        <button className="btn disclose" type="button" aria-expanded={manual} onClick={() => setOpen(!manual)}>
          {manual ? "Скрыть ручную настройку ▴" : "Настроить вручную ▾"}
        </button>
      )}
      {manual && (
        <div className="manual">
          <div className="field">
            <div className="field-h">
              <span>
                Риск на сделку <Q tip="Какую долю баланса бот готов потерять, если сделка закроется по стоп-лоссу. Размер позиции бот считает сам." />
              </span>
              <output className="num">
                {risk}% · ${fmtN(one, 0)}
              </output>
            </div>
            <input type="range" min="0.5" max="5" step="0.5" value={risk} aria-label="Риск на сделку, %" onChange={(e) => setRisk(+e.target.value)} />
            <div className="quick">
              {[1, 2, 4].map((v) => (
                <button key={v} className="btn" type="button" aria-pressed={risk === v} onClick={() => setRisk(v)}>
                  {v}%
                </button>
              ))}
            </div>
          </div>
          <div className="field">
            <div className="field-h">
              <span>
                Сделок одновременно{" "}
                <Q tip="Когда все места заняты, бот перестаёт искать новые сделки и только следит за открытыми. Так он экономит AI-запросы." />
              </span>
              <span className="stepper">
                <button className="btn" type="button" aria-label="Меньше" disabled={slots <= 1} onClick={() => setSlots(slots - 1)}>
                  −
                </button>
                <output>{slots}</output>
                <button className="btn" type="button" aria-label="Больше" disabled={slots >= 20} onClick={() => setSlots(slots + 1)}>
                  +
                </button>
              </span>
            </div>
            {paper && (
              <div className="quick">
                <button className="btn" type="button" aria-pressed={slots === 15} onClick={() => setSlots(15)}>
                  15 — больше статистики в Paper
                </button>
              </div>
            )}
          </div>
          <div className="field">
            <div className="field-h">
              <span>
                Плечо <Q tip="Фикс — всегда одно плечо. По волатильности — меньше плеча на резких монетах. Макс — максимальное плечо из настроек." />
              </span>
              <output className="num">{levText}</output>
            </div>
            <Seg
              className="full"
              value={lev}
              disabled={spot}
              options={[
                ["fixed", "Фикс"],
                ["volatility", "По волатильности"],
                ["full", "Макс"],
              ]}
              onChange={(v) => void actions.patchConfig({ LEVERAGE_MODE: v })}
            />
            {spot && <p>На споте плеча и шортов нет.</p>}
          </div>
          <p className="note-soon">Поменяли вручную — вверху «Свои настройки». Пресет можно выбрать снова в любой момент. Уменьшение сделок при просадке — в «Правилах защиты».</p>
        </div>
      )}
    </article>
  );
}

// ── Подтверждение сделок ──────────────────────────────────────────────────────

const DECISION_MIN = 10;
const DECISION_MAX = 120; // дольше ждать нельзя: цена уходит, сигнал устаревает
const DECISION_QUICK = [
  [15, "15 с"],
  [30, "30 с"],
  [60, "1 мин"],
  [120, "2 мин"],
] as const;

export function ConfirmCard() {
  const c = useStore((s) => s.config);
  const paper = useStore((s) => s.bot?.paper ?? true);
  const say = useStore((s) => s.say);
  const auto = cfg(c, "AUTO_EXECUTE", false);
  const timeout = cfg<number>(c, "TELEGRAM_CONFIRM_TIMEOUT", 60);
  const onTimeout = cfg<string>(c, "CONFIRM_ON_TIMEOUT", "execute");
  const setTimeoutSec = (raw: string) => {
    const v = Math.round(Number(raw));
    if (!Number.isFinite(v)) return;
    const clamped = Math.max(DECISION_MIN, Math.min(DECISION_MAX, v));
    if (clamped !== v) say(`Время на решение — от ${DECISION_MIN} до ${DECISION_MAX} секунд.`);
    if (clamped !== timeout) void actions.patchConfig({ TELEGRAM_CONFIRM_TIMEOUT: clamped });
  };
  return (
    <article className="card">
      <div className="card-h">
        <h2>Открытие сделок</h2>
      </div>
      <div className="field">
        <div className="field-h">
          <span>
            Как открывать <Q tip="Автоматически — бот нашёл сигнал и сразу открыл сделку. С подтверждением — каждая сделка приходит вам в Telegram и сюда." />
          </span>
        </div>
        <Seg
          className="full"
          value={auto ? "auto" : "confirm"}
          options={[
            ["auto", "Автоматически"],
            ["confirm", "С подтверждением"],
          ]}
          label="Как открывать сделки"
          onChange={(v) => void actions.patchConfig({ AUTO_EXECUTE: v === "auto" })}
        />
        <p>
          {auto
            ? "Нашёл сигнал — сразу открывает сделку, без ожидания: в торговле важна скорость."
            : "Бот присылает каждую сделку в Telegram и в приложение и ждёт вашего решения."}
        </p>
        {paper && <p className="note-soon">Сейчас Paper: бот не спрашивает ни в каком режиме — сделки виртуальные. Подтверждение работает в Live.</p>}
      </div>
      {!auto && (
        <>
          <div className="field">
            <div className="field-h">
              <span>
                Время на решение <Q tip="Не больше 2 минут: цена уходит, и сигнал устаревает." />
              </span>
              <output className="num">{timeout} с</output>
            </div>
            <div className="quick">
              {DECISION_QUICK.map(([v, label]) => (
                <button key={v} className="btn" type="button" aria-pressed={timeout === v} onClick={() => void actions.patchConfig({ TELEGRAM_CONFIRM_TIMEOUT: v })}>
                  {label}
                </button>
              ))}
              <span className="numf" style={{ flex: "none", width: 104 }}>
                <input
                  key={timeout}
                  type="number"
                  min={DECISION_MIN}
                  max={DECISION_MAX}
                  step="1"
                  inputMode="numeric"
                  defaultValue={timeout}
                  aria-label="Своё время на решение, секунд"
                  onBlur={(e) => setTimeoutSec(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
                />
                <span className="u">с</span>
              </span>
            </div>
            <p>
              От {DECISION_MIN} до {DECISION_MAX} секунд. Дольше ждать нельзя — цена уходит, и сигнал устаревает.
            </p>
          </div>
          <div className="field">
            <div className="field-h">
              <span>Если не ответил</span>
            </div>
            <Seg
              className="full"
              value={onTimeout}
              options={[
                ["execute", "Открыть сделку"],
                ["skip", "Пропустить"],
              ]}
              label="Если не ответил вовремя"
              onChange={(v) => void actions.patchConfig({ CONFIRM_ON_TIMEOUT: v })}
            />
            <p>
              Не ответили за {timeout} с — {onTimeout === "execute" ? "бот откроет сделку сам." : "сделка будет пропущена."}
            </p>
          </div>
        </>
      )}
    </article>
  );
}

// ── Что торговать ─────────────────────────────────────────────────────────────

/** simple — режим «Простой»: только профиль и авто-сезон. */
export function WhatCard({ simple = false }: { simple?: boolean }) {
  const c = useStore((s) => s.config);
  const profile = useStore((s) => s.bot?.profile ?? null);
  const seasonAuto = cfg<string>(c, "SEASON_SWITCH_MODE", "alert") === "auto";
  const chronos = cfg(c, "CHRONOS_ENABLED", false);
  return (
    <article className="card">
      <div className="card-h">
        <h2>Что торговать</h2>
      </div>
      <div className="field">
        <div className="field-h">
          <span>Профиль рынка</span>
          {seasonAuto && <span className="pill">авто по сезону</span>}
        </div>
        <Seg className="full" value={profile} options={PROFILES} onChange={(v) => void actions.setProfile(v)} label="Профиль рынка" />
        {profile && <p>{PROFILE_NOTE[profile]}</p>}
      </div>
      <div className="field">
        <div className="row-tgl">
          <span>
            Менять профиль по сезону{" "}
            <Q tip="Бот следит за сезоном рынка. Включено — переключает профиль сам. Выключено — присылает предложение в Telegram." />
          </span>
          <Toggle brand on={seasonAuto} label="Авто-сезон" onChange={(on) => void actions.patchConfig({ SEASON_SWITCH_MODE: on ? "auto" : "alert" })} />
        </div>
      </div>
      {!simple && (
        <>
      <div className="field">
        <div className="field-h">
          <span>Рынок</span>
        </div>
        <Seg
          className="full"
          value={cfg<string>(c, "MARKET_TYPE", "linear")}
          options={[
            ["linear", "Фьючерсы"],
            ["spot", "Спот"],
          ]}
          onChange={(v) => void actions.patchConfig({ MARKET_TYPE: v })}
        />
      </div>
      <div className="field">
        <div className="row-tgl">
          <span>
            Прогноз Chronos <Q tip="Модель Amazon Chronos прогнозирует цену и чуть повышает или понижает уверенность сигнала." />
          </span>
          <Toggle brand on={chronos} label="Chronos" onChange={(on) => void actions.patchConfig({ CHRONOS_ENABLED: on })} />
        </div>
      </div>
        </>
      )}
    </article>
  );
}

// ── Стратегии ─────────────────────────────────────────────────────────────────

/** Кто принимает решение о сделке: от этого зависит, насколько важны стратегии. */
const MODE_ROLE: Record<string, { title: string; text: string }> = {
  local: { title: "Стратегии", text: "" },
  ai: {
    title: "Запасные стратегии и подсказки для AI",
    text: "Сделку решает AI. Включённые стратегии он видит как список типов сетапов и подписывает ими свои сделки — это подсказка, а не проверка: код стратегии при этом не запускается. Отключённых AI не видит. Сами стратегии торгуют, только если включить запасной путь ниже, и только пока AI недоступен.",
  },
  hybrid: {
    title: "Запасные стратегии и подсказки для AI",
    text: "Сделку открывают, когда AI и нейросеть SAC согласны. Включённые стратегии AI видит как подсказку, а нейросеть их не использует вовсе.",
  },
  dqn: {
    title: "Запасные стратегии",
    text: "Сделку решает нейросеть SAC и стратегии не использует. Они нужны только если включить запасной путь ниже: тогда бот торгует по ним, пока модель SAC не загружена.",
  },
};

export function StrategiesCard() {
  const all0 = useStore((s) => s.strategies);
  const marketSpot = useStore((s) => cfg<string>(s.config, "MARKET_TYPE", "linear") === "spot");
  const mode = useStore((s) => s.bot?.mode ?? "ai");
  const fallback = useStore((s) => cfg<boolean>(s.config, "LOCAL_FALLBACK", false));
  const [all, setAll] = useState(false);
  const [view, setView] = useState<"futures" | "spot" | null>(null);
  const [q, setQ] = useState("");
  // Закреплённые стратегии идут первыми; хранится только в этом браузере.
  const [pins, setPins] = useState<string[]>(() => {
    try {
      return JSON.parse(localStorage.getItem("bitbot.pins") ?? "[]") as string[];
    } catch {
      return [];
    }
  });
  const togglePin = (name: string) =>
    setPins((cur) => {
      const next = cur.includes(name) ? cur.filter((x) => x !== name) : [...cur, name];
      try {
        localStorage.setItem("bitbot.pins", JSON.stringify(next));
      } catch {
        /* без хранилища закрепление живёт до перезагрузки */
      }
      return next;
    });
  // Строки плавно переезжают при смене рынка, фильтра и поиска (FLIP: было → стало).
  const listRef = useRef<HTMLUListElement>(null);
  const prevTop = useRef(new Map<string, number>());
  useLayoutEffect(() => {
    const ul = listRef.current;
    if (!ul) return;
    const next = new Map<string, number>();
    for (const li of Array.from(ul.children) as HTMLElement[]) {
      const key = li.dataset.name ?? "";
      const top = li.offsetTop - ul.offsetTop;
      next.set(key, top);
      const was = prevTop.current.get(key);
      if (was !== undefined && Math.abs(was - top) > 1 && !reducedMotion() && li.animate) {
        li.animate([{ transform: `translateY(${was - top}px)` }, { transform: "none" }], { duration: 280, easing: "cubic-bezier(.3,.7,.1,1)" });
      }
    }
    prevTop.current = next;
  });
  const [kind, setKind] = useState<"all" | "trending" | "ranging" | "volatile">("all");
  const [onlyOn, setOnlyOn] = useState(false);
  // Что показываем: по умолчанию рынок из настроек; переключатель только меняет вид списка.
  // На споте нет шортов, поэтому стратегии «только фьючерсы» там не работают.
  const shown: "futures" | "spot" = view ?? (marketSpot ? "spot" : "futures");
  const spot = shown === "spot";
  const fits = (s: Strategy, m: "futures" | "spot") => s.markets === "both" || s.markets === m;
  const counts = { futures: all0.filter((s) => fits(s, "futures")).length, spot: all0.filter((s) => fits(s, "spot")).length };
  const strategies = all0.filter((s) => fits(s, shown));
  const hidden = all0.length - strategies.length;
  const needle = q.trim().toLowerCase();
  const kindOf = (s: Strategy) => (s.type === "trending" || s.type === "ranging" ? s.type : "volatile");
  const found = strategies.filter(
    (s) =>
      (kind === "all" || kindOf(s) === kind) &&
      (!onlyOn || s.enabled) &&
      (!needle || stratName(s.name).toLowerCase().includes(needle) || s.name.includes(needle)),
  );
  const sorted = [...found].sort(
    (a, b) => Number(pins.includes(b.name)) - Number(pins.includes(a.name)) || (b.backtest?.sharpe ?? -9) - (a.backtest?.sharpe ?? -9),
  );
  const filtering = Boolean(needle) || kind !== "all" || onlyOn;
  const rows = all || filtering ? sorted : sorted.slice(0, 6);
  const on = strategies.filter((s) => s.enabled).length;
  const tested = strategies.filter((s) => s.backtest && s.backtest.trades > 0);
  const edge = tested.filter((s) => s.backtest!.trades >= 100 && s.backtest!.sharpe > 0);
  const role = MODE_ROLE[mode] ?? MODE_ROLE.ai;
  const body = (
    <>
      {role.text && <p className="note-soon role">{role.text}</p>}
      {mode !== "local" && (
        <div className="row-tgl">
          <span>
            Если AI или SAC недоступны — торговать по этим стратегиям{" "}
            <Q tip="Только при сбое: в AI не ответил ни один провайдер, в SAC модель не загружена. Если модель ответила, но входа не увидела, сделки нет — стратегии её не перебивают. Как только модель снова отвечает, решает она. Выключено: модель недоступна — новых сделок нет, открытые сопровождаются." />
          </span>
          <Toggle on={fallback} label="Запасной путь при недоступности AI или SAC" onChange={(v) => void actions.patchConfig({ LOCAL_FALLBACK: v })} />
        </div>
      )}
      {tested.length > 0 && !edge.length && (
        <div className="honest">
          {Icon.warn}
          <div>
            <b>Бэктест: ни одна из {tested.length} стратегий не показала устойчивого преимущества.</b> Пока edge не доказан, держите Paper.
            Kill-switch следит за каждой стратегией в live.
          </div>
        </div>
      )}
      <div className="strat-tools">
        <Seg
          className="sm"
          value={shown}
          options={[
            ["futures", `Фьючерсы · ${counts.futures}`],
            ["spot", `Спот · ${counts.spot}`],
          ]}
          onChange={setView}
          label="Для какого рынка показать стратегии"
        />
        <input className="inp" type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Найти стратегию" aria-label="Найти стратегию" autoComplete="off" />
      </div>
      <div className="strat-tools">
        <Seg
          className="sm"
          value={kind}
          options={[
            ["all", "Все"],
            ["trending", "Тренд"],
            ["ranging", "Возврат"],
            ["volatile", "Пробой"],
          ]}
          onChange={setKind}
          label="Тип стратегий"
        />
        <button className="chip-btn" type="button" aria-pressed={onlyOn} onClick={() => setOnlyOn(!onlyOn)}>
          Только включённые
        </button>
      </div>
      {view && view !== (marketSpot ? "spot" : "futures") && (
        <p className="note-soon">Это просмотр: бот сейчас торгует на рынке «{marketSpot ? "Спот" : "Фьючерсы"}». Переключить — выше, в «Рынок».</p>
      )}
      <ul className="strats" ref={listRef}>
        {rows.map((s) => {
          const bt = s.backtest;
          const st = !bt || !bt.trades ? ["", "нет данных"] : bt.trades < 100 ? ["", "мало сделок"] : bt.sharpe > 0 ? ["ok", "есть плюс"] : ["warn", "нет edge"];
          return (
            <li key={s.name} data-name={s.name} className={`strat${s.enabled ? "" : " off"}`}>
              <div className="nm">
                <b>{stratName(s.name)}</b>
                <code>{s.name}</code>
              </div>
              <div className="row-r">
                {s.experimental && (
                  <span className="pill" title="Добавлена из Argent-Systems без доказанного преимущества. Включайте осознанно.">
                    эксперимент
                  </span>
                )}
                <span className="pill" title={s.markets === "futures" ? "Сигналы только на продажу: открыть можно лишь шорт, на споте их нет" : "Покупки работают и на споте, продажи (шорты) — только на фьючерсах"}>
                  {s.markets === "futures" ? "только фьючерсы" : "спот и фьючерсы"}
                </span>
                <span className={`pill ${st[0]}`}>{st[1]}</span>
                <button className="star" type="button" aria-pressed={pins.includes(s.name)} aria-label={`Закрепить: ${stratName(s.name)}`} title="Закрепить вверху списка" onClick={() => togglePin(s.name)}>
                  {Icon.star}
                </button>
                <Toggle on={s.enabled} label={`Стратегия «${stratName(s.name)}»`} onChange={(v) => void actions.setStrategy(s.name, v)} />
              </div>
              <div className="nums">
                <span>
                  бэктест: прибыльных <b>{bt?.trades ? `${fmtN(bt.win_rate * 100, 1)}%` : "—"}</b>
                </span>
                <span>
                  сделок <b>{fmtN(bt?.trades ?? 0, 0)}</b>
                </span>
                <span>
                  Sharpe <b className={bt?.trades ? tone(bt.sharpe) : ""}>{bt?.trades ? bt.sharpe.toFixed(3) : "—"}</b>
                </span>
                {s.live && (
                  <span>
                    live: <b>{fmtN(s.live.win_rate * 100, 1)}%</b> · {s.live.trades} · EV <b className={tone(s.live.ev)}>{s.live.ev.toFixed(2)}</b>
                  </span>
                )}
              </div>
            </li>
          );
        })}
      </ul>
      {hidden > 0 && (
        <p className="note-soon">
          Скрыто {hidden}: {plural(hidden, ["стратегия не подходит", "стратегии не подходят", "стратегий не подходят"])} рынку «{spot ? "Спот" : "Фьючерсы"}».
        </p>
      )}
      <div className="btn-row" style={{ marginTop: 8 }}>
        {found.length > 6 && !filtering && (
          <button className="btn" type="button" onClick={() => setAll(!all)}>
            {all ? "Свернуть" : `Показать все ${strategies.length}`}
          </button>
        )}
        <button className="btn" type="button" disabled={strategies.filter((s) => !s.experimental).every((s) => s.enabled)} title="Включить основные стратегии; экспериментальные останутся выключенными" onClick={() => void actions.resetStrategies()}>
          Включить основные
        </button>
      </div>
    </>
  );
  // У стратегий свой узел в конвейере: кто его открыл, хочет видеть список, поэтому не сворачиваем.
  return (
    <article className="card">
      <div className="card-h">
        <h2>{role.title}</h2>
        <span className="pill sp">
          включено {on} из {strategies.length}
        </span>
      </div>
      {body}
    </article>
  );
}

// ── Выход из сделки ───────────────────────────────────────────────────────────

const TP_AT: readonly (readonly [number, string])[] = [
  [0.5, "50%"],
  [0.6, "60%"],
  [0.75, "75%"],
];
const TP_PART: readonly (readonly [number, string])[] = [
  [0.3, "30%"],
  [0.5, "50%"],
  [0.7, "70%"],
];
const TRAIL: readonly (readonly [number, string])[] = [
  [0.5, "0.5 × ATR"],
  [1, "1 × ATR"],
  [1.5, "1.5 × ATR"],
  [2, "2 × ATR"],
];
const preset = (v: number, opts: readonly (readonly [number, string])[]) => opts.find(([x]) => Math.abs(x - v) < 1e-9)?.[0] ?? null;
const pctOf = (v: number) => `${+(v * 100).toFixed(1)}%`;

/** Частичная фиксация прибыли и трейлинг — ровно так, как их делает PositionMonitor. */
export function TrackCard() {
  const c = useStore((s) => s.config);
  const schema = useStore((s) => s.schema);
  const setTab = useStore((s) => s.setTab);
  const searchParams = useStore((s) => s.searchParams);
  const on = cfg(c, "PARTIAL_TP_ENABLED", true);
  const at = cfg(c, "PARTIAL_TP_TRIGGER", 0.6);
  const part = cfg(c, "PARTIAL_TP_FRACTION", 0.5);
  const trail = cfg(c, "TRAILING_STOP_ATR_MULT", 1);
  // Эти параметры бот читает при запуске — честно говорим, что нужен перезапуск.
  const restart = schema.some((d) => (d.key.startsWith("PARTIAL_TP_") || d.key === "TRAILING_STOP_ATR_MULT") && !d.hot);
  return (
    <article className="card">
      <div className="card-h">
        <h2>Выход из сделки</h2>
        {restart && <span className="tag rst sp">после перезапуска</span>}
      </div>
      <div className="row-tgl">
        <span>
          Частичная фиксация прибыли{" "}
          <Q tip="Когда цена проходит заданную часть пути до тейк-профита, бот закрывает часть позиции и переносит стоп-лосс на цену входа." />
        </span>
        <Toggle on={on} label="Частичная фиксация прибыли" onChange={(v) => void actions.patchConfig({ PARTIAL_TP_ENABLED: v })} />
      </div>
      {on && (
        <>
          <div className="field">
            <div className="field-h">
              <span>Когда фиксировать — пройдено пути до тейк-профита</span>
            </div>
            <Seg className="full" value={preset(at, TP_AT)} options={TP_AT} onChange={(v) => void actions.patchConfig({ PARTIAL_TP_TRIGGER: v })} label="Когда фиксировать" />
          </div>
          <div className="field">
            <div className="field-h">
              <span>Какую часть позиции закрыть</span>
            </div>
            <Seg className="full" value={preset(part, TP_PART)} options={TP_PART} onChange={(v) => void actions.patchConfig({ PARTIAL_TP_FRACTION: v })} label="Какую часть закрыть" />
          </div>
          <p className="note-soon">
            Цена прошла {pctOf(at)} пути до тейк-профита — бот закрывает {pctOf(part)} позиции и переносит стоп-лосс на цену входа (безубыток). Остаток идёт к
            тейк-профиту, а если цена развернётся — закроется около цены входа.
          </p>
        </>
      )}
      <div className="field">
        <div className="field-h">
          <span>
            Отступ трейлинг-стопа <Q tip="Стоп-лосс подтягивается за ценой на это расстояние в ATR — среднем ходе цены за свечу. Меньше — выход раньше, больше — сделке больше места." />
          </span>
          {preset(trail, TRAIL) === null && <span className="pill sp">сейчас {trail} × ATR</span>}
        </div>
        <Seg className="full" value={preset(trail, TRAIL)} options={TRAIL} onChange={(v) => void actions.patchConfig({ TRAILING_STOP_ATR_MULT: v })} label="Отступ трейлинг-стопа" />
      </div>
      <p className="note-soon">Трейлинг можно выключить у отдельной позиции — тумблером в «Терминале». Точные значения — в «Все параметры».</p>
      <div className="btn-row">
        <button className="btn" type="button" onClick={() => setTab("terminal")}>
          Позиции в Терминале →
        </button>
        <button className="btn" type="button" onClick={() => searchParams("PARTIAL_TP")}>
          Частичный TP в «Все параметры» →
        </button>
      </div>
    </article>
  );
}
