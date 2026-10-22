// Правила защиты «если → то» — те же защиты, что в коде бота, с порогами из настроек.
// Своих правил нет: бот выполняет только эти, поэтому и показываем только их.

import { useEffect, useRef, useState } from "react";

import type { ConfigValues, SettingDef } from "../lib/types";
import { actions, useStore } from "../store";
import { Q, Seg, Toggle } from "../ui";
import { cfg } from "./bots";
import { NumField, shown } from "./settings";

export interface Rule {
  id: string;
  cat: "Капитал" | "Позиция" | "Рынок" | "Бот";
  /** Порог — ключ настройки; правится прямо в строке. */
  key?: string;
  /** Тумблер правила — ключ настройки (bool) или тревога. */
  on?: string;
  alert?: string;
  ifm: string;
  cmp: string;
  /** Порог без ключа (не настраивается). */
  fixed?: string;
  then: (c: ConfigValues) => string;
  /** В Paper бот только пишет в Telegram, а не останавливает торговлю. */
  paperNotify?: boolean;
  /** Не проверяется, если включено «Paper без защит». */
  guardsOff?: boolean;
  /** 0 в пороге — правило выключено. */
  zeroOff?: boolean;
}

export const RULES: Rule[] = [
  {
    id: "daily",
    cat: "Капитал",
    key: "DAILY_LOSS_LIMIT",
    ifm: "Убыток за сутки (UTC)",
    cmp: "≥",
    then: () => "новые входы закрыты до 00:00 UTC или пока баланс не восстановится; открытые позиции живут по своим SL/TP",
    paperNotify: true,
    guardsOff: true,
  },
  {
    id: "streak",
    cat: "Капитал",
    key: "CIRCUIT_BREAKER_LOSSES",
    ifm: "Убыточных сделок подряд",
    cmp: "≥",
    then: () => "бот останавливается; запустить снова — вручную",
    paperNotify: true,
    zeroOff: true,
  },
  {
    id: "dd",
    cat: "Капитал",
    key: "MAX_DRAWDOWN_PERCENT",
    ifm: "Просадка от пика баланса",
    cmp: "≥",
    then: (c) =>
      `убыточные позиции закрыть, у прибыльных стоп в безубыток, новых входов нет ${cfg(c, "DRAWDOWN_HALT_HOURS", 4)} ч (если держится ${cfg(c, "DRAWDOWN_CONFIRM_CYCLES", 3)} цикла подряд)`,
    paperNotify: true,
    guardsOff: true,
  },
  {
    id: "ddscale",
    cat: "Капитал",
    key: "DRAWDOWN_SCALE_THRESHOLD",
    on: "DRAWDOWN_SCALE_ENABLED",
    ifm: "Просадка от пика баланса",
    cmp: "≥",
    then: (c) => `новые сделки меньше: размер × ${cfg(c, "DRAWDOWN_SCALE_FACTOR", 0.5)}`,
  },
  {
    id: "sl",
    cat: "Позиция",
    key: "STOP_LOSS_PERCENT",
    ifm: "Цена ушла против позиции",
    cmp: "≥",
    then: () => "закрыть по стоп-лоссу (у профиля рынка может быть свой)",
  },
  {
    id: "slots",
    cat: "Позиция",
    key: "MAX_POSITIONS",
    ifm: "Открыто сделок",
    cmp: "≥",
    then: () => "новые не открывать, только сопровождать открытые",
  },
  {
    id: "exposure",
    cat: "Позиция",
    key: "MAX_PORTFOLIO_EXPOSURE",
    ifm: "Нагрузка портфеля",
    cmp: ">",
    then: () => "новые не открывать",
    guardsOff: true,
    zeroOff: true,
  },
  {
    id: "cooldown",
    cat: "Позиция",
    key: "SYMBOL_COOLDOWN_HOURS",
    ifm: "С прошлой сделки по монете прошло",
    cmp: "<",
    then: () => "в эту монету не входить",
    guardsOff: true,
    zeroOff: true,
  },
  {
    id: "atr",
    cat: "Рынок",
    key: "ATR_SPIKE_MULT",
    ifm: "Волатильность выше обычной",
    cmp: "≥",
    then: () => "закрыть все позиции и пропустить цикл",
    guardsOff: true,
  },
  {
    id: "macro",
    cat: "Рынок",
    on: "MACRO_BLACKOUT_ENABLED",
    ifm: "До или после FOMC, CPI, NFP",
    cmp: "<",
    fixed: "30 мин",
    then: () => "не входить (нужен ключ Finnhub)",
    guardsOff: true,
  },
  {
    id: "silent",
    cat: "Бот",
    key: "SILENT_DEATH_HOURS",
    ifm: "Бот не открывал сделок",
    cmp: "≥",
    then: () => "тревога в Telegram",
  },
  {
    id: "stalled",
    cat: "Бот",
    alert: "bot_stalled",
    ifm: "Бот не отвечает или цикл не начинался",
    cmp: ">",
    fixed: "интервал + 15 мин",
    then: () => "тревога в Telegram, потом «снова работает» (проверяет панель — работает, пока включён компьютер)",
  },
];

/** Наборы порогов. «Как задумано» — значения по умолчанию из схемы настроек. */
const TEMPLATES = {
  strict: { DAILY_LOSS_LIMIT: 0.03, CIRCUIT_BREAKER_LOSSES: 2, MAX_DRAWDOWN_PERCENT: 0.1, SYMBOL_COOLDOWN_HOURS: 6, DRAWDOWN_SCALE_THRESHOLD: 0.05 },
  base: null,
  loose: { DAILY_LOSS_LIMIT: 0.08, CIRCUIT_BREAKER_LOSSES: 5, MAX_DRAWDOWN_PERCENT: 0.3, SYMBOL_COOLDOWN_HOURS: 1, DRAWDOWN_SCALE_THRESHOLD: 0.15 },
} as const;
type Tpl = keyof typeof TEMPLATES;
const TPL_KEYS = Object.keys(TEMPLATES.strict) as (keyof typeof TEMPLATES.strict)[];
const TPL_OPTS = [
  ["strict", "Строже"],
  ["base", "Как задумано"],
  ["loose", "Свободнее"],
] as const;

export function templateValues(t: Tpl, schema: SettingDef[]): Record<string, unknown> {
  const v = TEMPLATES[t];
  return v ?? Object.fromEntries(TPL_KEYS.map((k) => [k, schema.find((d) => d.key === k)?.default]));
}

export function templateOf(c: ConfigValues, schema: SettingDef[]): Tpl | null {
  const same = (a: unknown, b: unknown) => Math.abs(Number(a) - Number(b)) < 1e-9;
  return (Object.keys(TEMPLATES) as Tpl[]).find((t) => Object.entries(templateValues(t, schema)).every(([k, v]) => same(cfg(c, k, NaN), v))) ?? null;
}

/** Живёт в узле «Риск» вкладки «Бот». */
export function RulesCard() {
  const c = useStore((s) => s.config);
  const schema = useStore((s) => s.schema);
  const alerts = useStore((s) => s.alerts);
  const paper = useStore((s) => s.bot?.paper ?? true);
  const guardsOff = paper && cfg(c, "PAPER_GUARDS_OFF", false);
  const restart = RULES.some((r) => r.key && schema.find((d) => d.key === r.key)?.hot === false);
  return (
    <article className="card">
      <div className="card-h">
        <h2>Правила защиты</h2>
        <Q tip="Каждое правило — «если → то»: бот проверяет их перед ордером и в каждом цикле. Пороги — те же параметры, что в «Все параметры». Своих правил добавить нельзя: бот выполняет только эти." />
        {restart && <span className="tag rst sp">после перезапуска</span>}
      </div>
      <div className="field">
        <div className="field-h">
          <span>Шаблон порогов</span>
          {templateOf(c, schema) === null && <span className="pill sp">свои пороги</span>}
        </div>
        <Seg
          className="full"
          value={templateOf(c, schema)}
          options={TPL_OPTS}
          label="Шаблон порогов"
          onChange={(t) => void actions.patchConfig(templateValues(t, schema))}
        />
        <p className="note-soon">Меняет дневной лимит, серию убытков, предельную просадку, паузу по монете и порог уменьшения сделок.</p>
      </div>
      <ul className="rules">
        {RULES.map((r) => (
          <RuleRow key={r.id} r={r} c={c} schema={schema} paper={paper} guardsOff={guardsOff} alertOn={alerts.find((a) => a.kind === r.alert)?.enabled} />
        ))}
      </ul>
    </article>
  );
}

function RuleRow({ r, c, schema, paper, guardsOff, alertOn }: { r: Rule; c: ConfigValues; schema: SettingDef[]; paper: boolean; guardsOff: boolean; alertOn?: boolean }) {
  const d = schema.find((s) => s.key === r.key);
  const enabled = r.on ? cfg(c, r.on, true) : r.alert ? alertOn !== false : true;
  const zero = r.zeroOff && Number(cfg(c, r.key ?? "", 1)) === 0;
  const skipped = guardsOff && r.guardsOff;
  const active = enabled && !zero && !skipped;
  const note = skipped
    ? "не проверяется: Paper без защит"
    : zero
      ? "выключено: порог 0"
      : paper && r.paperNotify
        ? "в Paper — только сообщение в Telegram"
        : "";
  return (
    <li className={`rule${active ? "" : " off"}`}>
      <span className="rule-cat">{r.cat}</span>
      <div className="rule-if">
        <b>Если</b> {r.ifm} <span className="rule-cmp">{r.cmp}</span> {d ? <Threshold d={d} value={c[d.key]?.value} /> : <span className="rule-val">{r.fixed}</span>}
      </div>
      <div className="rule-then">
        <b>то</b> {r.then(c)}
      </div>
      {note && <p className="rule-note">{note}</p>}
      {(r.on || r.alert) && (
        <span className="rule-tgl">
          <Toggle
            on={enabled}
            label={`Правило: ${r.ifm}`}
            onChange={(v) => void (r.on ? actions.patchConfig({ [r.on]: v }) : actions.patchAlert(r.alert ?? "", { enabled: v }))}
          />
        </span>
      )}
    </li>
  );
}

/** Порог правится прямо в строке; на сервер уходит через 0.6 с после последней цифры. */
function Threshold({ d, value }: { d: SettingDef; value: unknown }) {
  const timer = useRef<ReturnType<typeof setTimeout>>();
  const [bad, setBad] = useState(false);
  useEffect(() => () => clearTimeout(timer.current), []);
  const change = (text: string) => {
    clearTimeout(timer.current);
    const n = parseFloat(text.replace(",", "."));
    const v = d.type === "pct" ? +(n / 100).toFixed(6) : n;
    const ok = Number.isFinite(n) && (d.min === null || v >= d.min) && (d.max === null || v <= d.max);
    setBad(!ok);
    if (ok) timer.current = setTimeout(() => void actions.patchConfig({ [d.key]: d.type === "int" ? Math.round(v) : v }), 600);
  };
  return (
    <span className={`rule-val${bad ? " bad" : ""}`} title={bad ? `от ${shown(d, d.min)} до ${shown(d, d.max)}` : undefined}>
      <NumField id={`rule-${d.key}`} d={d} value={Number(shown(d, value ?? d.default))} disabled={d.readonly} onChange={change} />
    </span>
  );
}
