// Вкладка «Бот»: режим риска, тонкая настройка, подтверждение сделок, что торговать, стратегии.

import { useEffect, useRef, useState } from "react";

import { fmtN, plural, tone } from "../lib/format";
import type { ConfigValues } from "../lib/types";
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
};
export const stratName = (k: string): string => STRATEGY_NAME[k] ?? k;

export const cfg = <T,>(c: ConfigValues, key: string, fallback: T): T => (c[key]?.value as T) ?? fallback;

type LevMode = "fixed" | "volatility" | "full";
interface Preset {
  name: string;
  d: string;
  slots: number;
  risk: number; // доля баланса
  dd: boolean;
  lev: LevMode;
  icon: JSX.Element;
}
const ico = (d: string) => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d={d} />
  </svg>
);
const PRESETS: Record<string, Preset> = {
  conservative: {
    name: "Консервативный",
    d: "Редкие сделки, маленький риск, строгая защита",
    slots: 2,
    risk: 0.01,
    dd: true,
    lev: "fixed",
    icon: ico("M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"),
  },
  moderate: {
    name: "Умеренный",
    d: "Баланс между частотой сделок и риском",
    slots: 3,
    risk: 0.02,
    dd: true,
    lev: "volatility",
    icon: ico("M12 3v18M7 21h10M3 7h18M6 7l-3 7a3 3 0 0 0 6 0zM18 7l-3 7a3 3 0 0 0 6 0z"),
  },
  aggressive: {
    name: "Агрессивный",
    d: "Больше сделок и риска, без защиты от просадки",
    slots: 5,
    risk: 0.04,
    dd: false,
    lev: "full",
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

export function Bots() {
  return (
    <section className="view enter" aria-label="Бот">
      <div className="cols">
        <div className="col">
          <RiskCard />
          <TuneCard />
          <ConfirmCard />
        </div>
        <div className="col">
          <WhatCard />
          <StrategiesCard />
        </div>
      </div>
    </section>
  );
}

// ── Режим риска ───────────────────────────────────────────────────────────────

/** Ключ пресета риска, совпадающего с настройками; undefined — свои настройки. */
export function presetOf(c: ConfigValues): string | undefined {
  const slots = cfg(c, "MAX_POSITIONS", 3);
  const risk = cfg(c, "RISK_PER_TRADE", 0.02);
  const dd = cfg(c, "DRAWDOWN_SCALE_ENABLED", true);
  const lev = cfg<LevMode>(c, "LEVERAGE_MODE", "volatility");
  return Object.entries(PRESETS).find(([, p]) => p.slots === slots && Math.abs(p.risk - risk) < 1e-9 && p.dd === dd && p.lev === lev)?.[0];
}

export const presetName = (k: string | undefined): string => (k ? PRESETS[k].name : "свои настройки");

export function RiskCard() {
  const c = useStore((s) => s.config);
  const balance = useStore((s) => s.balance);
  const slots = cfg(c, "MAX_POSITIONS", 3);
  const risk = cfg(c, "RISK_PER_TRADE", 0.02);
  const active = presetOf(c);
  const one = balance * risk;
  const apply = async (k: string) => {
    const p = PRESETS[k];
    await actions.patchConfig({ MAX_POSITIONS: p.slots, RISK_PER_TRADE: p.risk, DRAWDOWN_SCALE_ENABLED: p.dd, LEVERAGE_MODE: p.lev });
  };
  return (
    <article className="card">
      <div className="card-h">
        <h2>Режим риска</h2>
        <span className="pill sp">{active ? PRESETS[active].name : "Свои настройки"}</span>
      </div>
      <div className="presets">
        {Object.entries(PRESETS).map(([k, p]) => (
          <button key={k} className="preset" type="button" aria-pressed={k === active} onClick={() => k !== active && void apply(k)}>
            <span className="pi">{p.icon}</span>
            <b>{p.name}</b>
            <span className="pd">{p.d}</span>
            <dl>
              <div>
                <dt>Сделок сразу</dt>
                <dd>{p.slots}</dd>
              </div>
              <div>
                <dt>Риск на сделку</dt>
                <dd>{p.risk * 100}%</dd>
              </div>
              <div>
                <dt>Защита просадки</dt>
                <dd>{p.dd ? "вкл" : "выкл"}</dd>
              </div>
            </dl>
            {Icon.check}
          </button>
        ))}
      </div>
      <p className="preview">
        При балансе <b>${fmtN(balance, 0)}</b> одна сделка рискует до <b>${fmtN(one, 0)}</b>. Открыто может быть до <b>{slots}</b>{" "}
        {plural(slots, ["сделки", "сделок", "сделок"])} сразу — в худшем случае до <b>${fmtN(one * slots, 0)}</b>.
      </p>
    </article>
  );
}

// ── Тонкая настройка ──────────────────────────────────────────────────────────

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

function TuneCard() {
  const c = useStore((s) => s.config);
  const paper = useStore((s) => s.bot?.paper ?? true);
  const spot = cfg<string>(c, "MARKET_TYPE", "linear") === "spot";
  const lev = cfg<LevMode>(c, "LEVERAGE_MODE", "volatility");
  const levText = spot ? "1×" : lev === "fixed" ? `${cfg(c, "LEVERAGE", 3)}×` : `до ${cfg(c, "LEVERAGE_MAX", 5)}×`;
  const [risk, setRisk] = useDebouncedPatch(+(cfg(c, "RISK_PER_TRADE", 0.02) * 100).toFixed(2), "RISK_PER_TRADE", (v) => +(v / 100).toFixed(4));
  const [slots, setSlots] = useDebouncedPatch(cfg(c, "MAX_POSITIONS", 3), "MAX_POSITIONS");
  const dd = cfg(c, "DRAWDOWN_SCALE_ENABLED", true);
  return (
    <article className="card">
      <div className="card-h">
        <h2>Тонкая настройка</h2>
      </div>
      <div className="field">
        <div className="field-h">
          <span>
            Риск на сделку <Q tip="Какую долю баланса бот готов потерять, если сделка закроется по стоп-лоссу. Размер позиции бот считает сам." />
          </span>
          <output className="num">{risk}%</output>
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
      <div className="field">
        <div className="row-tgl">
          <span>
            Защита при просадке <Q tip="Во время просадки бот уменьшает размер новых сделок, а при сильной — останавливает входы." />
          </span>
          <Toggle on={dd} label="Защита при просадке" onChange={(on) => void actions.patchConfig({ DRAWDOWN_SCALE_ENABLED: on })} />
        </div>
      </div>
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

function ConfirmCard() {
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

function WhatCard() {
  const c = useStore((s) => s.config);
  const profile = useStore((s) => s.bot?.profile ?? null);
  const seasonAuto = cfg<string>(c, "SEASON_SWITCH_MODE", "alert") === "auto";
  const mode = cfg<string>(c, "MODE", "ai");
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
        <div className="field-h">
          <span>
            Кто принимает решение{" "}
            <Q tip="Local — только индикаторы. AI — языковая модель. SAC — обученная нейросеть. Hybrid — SAC и AI должны согласиться." />
          </span>
        </div>
        <Seg
          className="full"
          value={mode}
          options={[
            ["local", "Local"],
            ["ai", "AI"],
            ["dqn", "SAC"],
            ["hybrid", "Hybrid"],
          ]}
          onChange={(v) => void actions.patchConfig({ MODE: v })}
        />
        <div className="row-tgl">
          <span>
            Прогноз Chronos <Q tip="Модель Amazon Chronos прогнозирует цену и чуть повышает или понижает уверенность сигнала." />
          </span>
          <Toggle brand on={chronos} label="Chronos" onChange={(on) => void actions.patchConfig({ CHRONOS_ENABLED: on })} />
        </div>
      </div>
    </article>
  );
}

// ── Стратегии ─────────────────────────────────────────────────────────────────

/** Кто принимает решение о сделке: от этого зависит, насколько важны стратегии. */
const MODE_ROLE: Record<string, { title: string; text: string }> = {
  local: { title: "Стратегии", text: "" },
  ai: {
    title: "Запасные стратегии и подсказки для AI",
    text: "Сделку решает AI. Включённые стратегии он видит как список типов сетапов и подписывает ими свои сделки — это подсказка, а не проверка: код стратегии при этом не запускается. Отключённых AI не видит.",
  },
  hybrid: {
    title: "Запасные стратегии и подсказки для AI",
    text: "Сделку открывают, когда AI и нейросеть SAC согласны. Включённые стратегии AI видит как подсказку, а нейросеть их не использует вовсе.",
  },
  dqn: {
    title: "Запасные стратегии",
    text: "Сделку решает нейросеть SAC и стратегии не использует. Они нужны только как запасной путь, если SAC не дала сигнала.",
  },
};

function StrategiesCard() {
  const all0 = useStore((s) => s.strategies);
  const spot = useStore((s) => cfg<string>(s.config, "MARKET_TYPE", "linear") === "spot");
  const mode = useStore((s) => s.bot?.mode ?? "ai");
  const fallback = useStore((s) => cfg<boolean>(s.config, "LOCAL_FALLBACK", true));
  const [all, setAll] = useState(false);
  // Показываем только то, что подходит текущему рынку: на споте нет шортов,
  // стратегии «только фьючерсы» там не работают, и наоборот.
  const strategies = all0.filter((s) => s.markets === "both" || s.markets === (spot ? "spot" : "futures"));
  const hidden = all0.length - strategies.length;
  const sorted = [...strategies].sort((a, b) => (b.backtest?.sharpe ?? -9) - (a.backtest?.sharpe ?? -9));
  const rows = all ? sorted : sorted.slice(0, 6);
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
            Если решение не пришло — торговать по этим стратегиям{" "}
            <Q tip="Бывают циклы, когда AI или SAC не дали ни одного сигнала. Включено: тогда бот сам ищет вход по включённым стратегиям ниже. Выключено: нет сигнала от модели — нет сделки." />
          </span>
          <Toggle on={fallback} label="Запасной путь по локальным стратегиям" onChange={(v) => void actions.patchConfig({ LOCAL_FALLBACK: v })} />
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
      <ul className="strats">
        {rows.map((s) => {
          const bt = s.backtest;
          const st = !bt || !bt.trades ? ["", "нет данных"] : bt.trades < 100 ? ["", "мало сделок"] : bt.sharpe > 0 ? ["ok", "есть плюс"] : ["warn", "нет edge"];
          return (
            <li key={s.name} className={`strat${s.enabled ? "" : " off"}`}>
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
                {s.markets === "futures" && (
                  <span className="pill" title="Сигналы только на продажу: открыть можно лишь шорт">
                    только фьючерсы
                  </span>
                )}
                <span className={`pill ${st[0]}`}>{st[1]}</span>
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
        {strategies.length > 6 && (
          <button className="btn" type="button" onClick={() => setAll(!all)}>
            {all ? "Свернуть" : `Показать все ${strategies.length}`}
          </button>
        )}
        <button className="btn" type="button" disabled={on === strategies.length} onClick={() => void actions.resetStrategies()}>
          Включить все
        </button>
      </div>
    </>
  );
  // Когда решают AI или SAC, стратегии вторичны — список свёрнут, чтобы не отвлекать.
  return (
    <article className="card">
      <div className="card-h">
        <h2>{role.title}</h2>
        <span className="pill sp">
          включено {on} из {strategies.length}
        </span>
      </div>
      {mode === "local" ? (
        body
      ) : (
        <details className="more">
          <summary>Что это и как влияет</summary>
          {body}
        </details>
      )}
    </article>
  );
}
