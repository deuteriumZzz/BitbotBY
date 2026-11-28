// Правила защиты «если → то» — те же защиты, что в коде бота, с порогами из настроек.
// Своих правил нет: бот выполняет только эти, поэтому и показываем только их.

import { useEffect, useRef, useState, type ReactNode } from "react";
import { Bell, CalendarX, ChartLineDown, Hourglass, Lightning, Newspaper, Scales, ShieldWarning, Stack, TrendDown } from "@phosphor-icons/react";

import type { ConfigValues, SettingDef } from "../lib/types";
import { actions } from "../store/actions";
import { useStore } from "../store";
import { Seg, Toggle } from "../components/controls";
import { Q } from "../components/tooltip";
import { IRow } from "../components/irow";
import { cfg } from "./bots";
import { shown } from "./params";
import { NumField, fmtVal } from "./settings/params";

export interface Rule {
  id: string;
  cat: "Капитал" | "Позиция" | "Рынок" | "Бот";
  /** Порог — ключ настройки; правится прямо в строке. */
  key?: string;
  /** Тумблер правила — ключ настройки (bool) или тревога. */
  on?: string;
  alert?: string;
  ifm: string;
  icon: ReactNode;
  /** Что сделает бот — коротко, для строки; полное условие и действие — в «?». */
  short: string;
  /** Короткая подпись для пилюли на «Цепочке», где полное условие обрезается. */
  chip?: string;
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
  /** Порог правится в другом месте — здесь только значение и ссылка туда. */
  elsewhere?: string;
}

export const RULES: Rule[] = [
  {
    id: "daily",
    icon: <CalendarX />,
    short: "стоп входов до 00:00 UTC",
    cat: "Капитал",
    key: "DAILY_LOSS_LIMIT",
    ifm: "Убыток за сутки (UTC)",
    chip: "Дневной убыток",
    cmp: "≥",
    then: () => "новые входы закрыты до 00:00 UTC или пока баланс не восстановится; открытые позиции живут по своим SL/TP",
    paperNotify: true,
    guardsOff: true,
  },
  {
    id: "streak",
    icon: <TrendDown />,
    short: "бот остановится",
    cat: "Капитал",
    key: "CIRCUIT_BREAKER_LOSSES",
    ifm: "Убыточных сделок подряд",
    chip: "Серия убытков",
    cmp: "≥",
    then: () => "бот останавливается; запустить снова — вручную",
    paperNotify: true,
    zeroOff: true,
  },
  {
    id: "dd",
    icon: <ChartLineDown />,
    short: "закрыть убыточные, пауза",
    cat: "Капитал",
    key: "MAX_DRAWDOWN_PERCENT",
    ifm: "Просадка от пика баланса",
    chip: "Просадка: стоп",
    cmp: "≥",
    then: (c) =>
      `убыточные позиции закрыть, у прибыльных стоп в безубыток, новых входов нет ${cfg(c, "DRAWDOWN_HALT_HOURS", 4)} ч (если держится ${cfg(c, "DRAWDOWN_CONFIRM_CYCLES", 3)} цикла подряд)`,
    paperNotify: true,
    guardsOff: true,
  },
  {
    id: "ddscale",
    icon: <Scales />,
    short: "меньше ставка",
    cat: "Капитал",
    key: "DRAWDOWN_SCALE_THRESHOLD",
    on: "DRAWDOWN_SCALE_ENABLED",
    ifm: "Просадка от пика: уменьшить ставку",
    chip: "Снижение ставки",
    cmp: "≥",
    then: (c) => `новые сделки меньше: размер × ${cfg(c, "DRAWDOWN_SCALE_FACTOR", 0.5)}`,
  },
  {
    id: "sl",
    icon: <ShieldWarning />,
    short: "закрыть по стопу",
    cat: "Позиция",
    key: "STOP_LOSS_PERCENT",
    ifm: "Цена ушла против позиции",
    chip: "Против позиции",
    cmp: "≥",
    then: () => "закрыть по стоп-лоссу (у профиля рынка может быть свой)",
  },
  {
    id: "slots",
    icon: <Stack />,
    short: "новые не открывать",
    cat: "Позиция",
    key: "MAX_POSITIONS",
    elsewhere: "меняется в «Как рискуем»",
    ifm: "Открыто сделок",
    cmp: "≥",
    then: () => "новые не открывать, только сопровождать открытые",
  },
  {
    id: "exposure",
    icon: <Stack />,
    short: "новые не открывать",
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
    icon: <Hourglass />,
    short: "в монету не входить",
    cat: "Позиция",
    key: "SYMBOL_COOLDOWN_HOURS",
    ifm: "С прошлой сделки по монете прошло",
    chip: "Пауза по монете",
    cmp: "<",
    then: () => "в эту монету не входить",
    guardsOff: true,
    zeroOff: true,
  },
  {
    id: "atr",
    icon: <Lightning />,
    short: "закрыть всё, пропустить цикл",
    cat: "Рынок",
    key: "ATR_SPIKE_MULT",
    ifm: "Волатильность выше обычной",
    chip: "Всплеск волатильности",
    cmp: "≥",
    then: () => "закрыть все позиции и пропустить цикл",
    guardsOff: true,
  },
  {
    id: "macro",
    icon: <Newspaper />,
    short: "не входить",
    cat: "Рынок",
    on: "MACRO_BLACKOUT_ENABLED",
    ifm: "До или после FOMC, CPI, NFP",
    chip: "Макро-новости",
    cmp: "<",
    fixed: "30 мин",
    then: () => "не входить (нужен ключ Finnhub)",
    guardsOff: true,
  },
  {
    id: "silent",
    icon: <Bell />,
    short: "тревога в Telegram",
    cat: "Бот",
    key: "SILENT_DEATH_HOURS",
    ifm: "Бот не открывал сделок",
    chip: "Нет сделок",
    cmp: "≥",
    then: () => "тревога в Telegram",
  },
  {
    id: "stalled",
    icon: <Bell />,
    short: "тревога в Telegram",
    cat: "Бот",
    alert: "bot_stalled",
    ifm: "Бот не отвечает или цикл не начинался",
    chip: "Бот молчит",
    cmp: ">",
    fixed: "интервал + 15 мин",
    then: () => "тревога в Telegram, потом «снова работает» (проверяет панель — работает, пока включён компьютер)",
  },
];

// «Paper без защит» — общий тумблер в шапке карточки (раньше была отдельная карточка).
export const GUARDS_OFF = [
  "Пауза по монете после сделки",
  "Пауза и закрытие позиций при всплеске волатильности",
  "Пауза вокруг макро-новостей",
  "Фильтр корреляции с открытыми позициями",
  "Лимит общей нагрузки портфеля",
  "Kill-switch убыточных стратегий",
  "Предупреждения о просадке, дневном лимите и серии убытков (в Paper они и так ничего не останавливают)",
];
export const GUARDS_KEPT = [
  "Стоп-лосс и тейк-профит на бирже",
  "Проверка ликвидности (спред и объём)",
  "Риск на сделку, лимит позиций, масштабирование при просадке",
  "Живой режим: переключатель в нём не действует вообще",
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

/** Шаблон порогов — одно место для набора порогов; в «Простом» это вопрос «Защита». */
export function TemplateSeg() {
  const c = useStore((s) => s.config);
  const schema = useStore((s) => s.schema);
  return (
    <Seg
      className="full"
      value={templateOf(c, schema)}
      options={TPL_OPTS}
      label="Шаблон порогов"
      onChange={(t) => void actions.patchConfig(templateValues(t, schema))}
    />
  );
}

/** Одна строка о текущих порогах — для «Простого». */
export function templateLine(c: ConfigValues): string {
  const pct = (k: string, d: number) => `${+(cfg(c, k, d) * 100).toFixed(1)}%`;
  return `Дневной лимит ${pct("DAILY_LOSS_LIMIT", 0.05)} · стоп после ${cfg(c, "CIRCUIT_BREAKER_LOSSES", 3)} убытков подряд · предельная просадка ${pct("MAX_DRAWDOWN_PERCENT", 0.2)}`;
}

/** Живёт в узле «Риск» вкладки «Бот». */
export function RulesCard() {
  const c = useStore((s) => s.config);
  const schema = useStore((s) => s.schema);
  const alerts = useStore((s) => s.alerts);
  const paper = useStore((s) => s.bot?.paper ?? true);
  const guardsOff = paper && cfg(c, "PAPER_GUARDS_OFF", false);
  const restart = RULES.some((r) => r.key && !r.elsewhere && schema.find((d) => d.key === r.key)?.hot === false);
  return (
    <article className="card">
      <div className="card-h">
        <h2>Правила защиты</h2>
        <Q tip="Бот проверяет эти правила перед каждым ордером и в каждом цикле. Пороги можно менять, добавить своё правило нельзя." />
        {restart && <span className="tag rst">после перезапуска</span>}
        <span className="rules-master sp">
          {paper ? (
            <>
              <span>Все защиты</span>
              <Toggle on={!guardsOff} label="Все защиты" onChange={(on) => void actions.patchConfig({ PAPER_GUARDS_OFF: !on })} />
            </>
          ) : (
            <span className="tag">в Live всегда вкл</span>
          )}
        </span>
      </div>
      {guardsOff && (
        <div className="guards-off">
          <p>
            <b>Paper без защит:</b> бот входит чаще, чтобы быстрее набрать опыт для обучения. В Live не действует.
          </p>
          <details>
            <summary>Что именно отключается</summary>
            <b>Отключается в Paper</b>
            <ul>
              {GUARDS_OFF.map((t) => (
                <li key={t}>{t}</li>
              ))}
            </ul>
            <b>Остаётся всегда</b>
            <ul>
              {GUARDS_KEPT.map((t) => (
                <li key={t}>{t}</li>
              ))}
            </ul>
          </details>
        </div>
      )}
      {paper && !guardsOff && <p className="note-soon">В Paper правила капитала только присылают сообщение в Telegram.</p>}
      <div className="irows">
        {RULES.filter((r) => r.cat !== "Бот").map((r) => (
          <RuleRow key={r.id} r={r} c={c} schema={schema} guardsOff={guardsOff} alertOn={alerts.find((a) => a.kind === r.alert)?.enabled} />
        ))}
      </div>
    </article>
  );
}

/** «Бот не открывал сделок N ч» — тревога, живёт в уведомлениях одной строкой. «Бот не отвечает» —
 * это уведомление «Бот перестал работать», отдельной строки не нужно. */
export function BotRules() {
  const c = useStore((s) => s.config);
  const schema = useStore((s) => s.schema);
  const d = schema.find((x) => x.key === "SILENT_DEATH_HOURS");
  if (!d) return null;
  return (
    <IRow icon={<Hourglass />} title="Нет сделок дольше" sub="тревога в Telegram">
      <Threshold d={d} value={c[d.key]?.value} />
    </IRow>
  );
}

function RuleRow({ r, c, schema, guardsOff, alertOn }: { r: Rule; c: ConfigValues; schema: SettingDef[]; guardsOff: boolean; alertOn?: boolean }) {
  const d = schema.find((s) => s.key === r.key);
  const enabled = r.on ? cfg(c, r.on, true) : r.alert ? alertOn !== false : true;
  const zero = r.zeroOff && Number(cfg(c, r.key ?? "", 1)) === 0;
  const skipped = guardsOff && r.guardsOff;
  const active = enabled && !zero && !skipped;
  const sub = skipped ? "не проверяется: Paper без защит" : zero ? "выключено: порог 0" : r.short;
  return (
    <IRow
      icon={r.icon}
      title={r.chip ?? r.ifm}
      sub={sub}
      tip={`Если ${r.ifm.toLowerCase()} ${r.cmp} порога — ${r.then(c)}.${r.elsewhere ? ` Порог ${r.elsewhere}.` : ""}`}
      off={!active}
    >
      {d && !r.elsewhere ? <Threshold d={d} value={c[d.key]?.value} /> : <span className="rule-val ro">{d ? fmtVal(d, c[d.key]?.value ?? d.default) : r.fixed}</span>}
      {r.on && <Toggle on={enabled} label={`Правило: ${r.ifm}`} onChange={(v) => void actions.patchConfig({ [r.on as string]: v })} />}
    </IRow>
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
