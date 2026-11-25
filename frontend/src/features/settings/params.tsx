// Настройки → Все параметры: таблица настроек с источником значения и правкой.

import { useEffect, useMemo, useState } from "react";

import { OWN_HOME } from "../../lib/where";
import { fmtN, plural } from "../../lib/format";

import type { SettingDef } from "../../lib/types";
import { actions, type PatchResult } from "../../store/actions";
import { useStore } from "../../store";
import { Toggle } from "../../components/controls";
import { Q } from "../../components/tooltip";

import { shown, unitOf } from "../params";
import { Pane } from "./index";
import { SettingsTools } from "./tools";

// ══ Все параметры ════════════════════════════════════════════════════════════

const NUMERIC = ["int", "float", "pct"];

export function fmtVal(d: SettingDef, v: unknown): string {
  if (d.type === "bool") return v ? "вкл" : "выкл";
  if (d.type === "choice") return d.options.find((o) => o.value === v)?.label ?? String(v);
  if (d.type === "str") return v ? String(v) : "пусто";
  const n = Number(shown(d, v));
  return `${Math.abs(n) >= 10000 ? fmtN(n, 0) : n}${unitOf(d) ? ` ${unitOf(d)}` : ""}`;
}

const SOURCE: Record<string, string> = {
  live: "задано в приложении",
  ui: "задано в приложении",
  env: "из файла .env",
  default: "",
};

export function AllParams({ go }: { go: (pane: Pane) => void }) {
  const schema = useStore((s) => s.schema);
  const config = useStore((s) => s.config);
  const jump = useStore((s) => s.paramSearch);
  const [q, setQ] = useState(jump ?? "");
  useEffect(() => {
    if (jump === null) return;
    setQ(jump);
    useStore.getState().clearParamSearch();
    setTimeout(() => document.getElementById("param-search")?.focus(), 60);
  }, [jump]);
  const [pending, setPending] = useState<Record<string, unknown>>({});
  const [bad, setBad] = useState<Record<string, string>>({});
  const [restartKeys, setRestartKeys] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [open, setOpen] = useState<Set<string>>(() => new Set(schema.slice(0, 1).map((d) => d.group)));

  const groups = useMemo(() => {
    const out: [string, SettingDef[]][] = [];
    for (const d of schema) {
      const g = out.find(([name]) => name === d.group);
      if (g) g[1].push(d);
      else out.push([d.group, [d]]);
    }
    return out;
  }, [schema]);
  const needle = q.trim().toLowerCase();
  // Поиск ищет по всем параметрам; без поиска — основные или все.
  const hit = (d: SettingDef) =>
    needle ? `${d.label} ${d.key} ${d.hint}`.toLowerCase().includes(needle) : !(d.key in OWN_HOME) && (advanced || d.basic || d.key in pending);
  const found = schema.filter(hit).length;
  const elsewhere = schema.filter((d) => d.key in OWN_HOME).length;

  const change = (d: SettingDef, raw: unknown) => {
    const cur = config[d.key]?.value;
    let v = raw;
    let err = "";
    if (NUMERIC.includes(d.type)) {
      const x = parseFloat(String(raw).replace(",", "."));
      const u = unitOf(d) ? ` ${unitOf(d)}` : "";
      const min = d.min === null ? null : Number(shown(d, d.min));
      const max = d.max === null ? null : Number(shown(d, d.max));
      if (!Number.isFinite(x)) err = "Введите число.";
      else if (min !== null && x < min) err = `Не меньше ${min}${u}.`;
      else if (max !== null && x > max) err = `Не больше ${max}${u}.`;
      else if (d.type === "int" && !Number.isInteger(x)) err = "Нужно целое число.";
      v = d.type === "pct" ? +(x / 100).toFixed(6) : x;
    }
    const nextBad = { ...bad };
    const nextPending = { ...pending };
    if (err) {
      nextBad[d.key] = err;
      delete nextPending[d.key];
    } else {
      delete nextBad[d.key];
      const same = typeof v === "number" && typeof cur === "number" ? Math.abs(v - cur) < 1e-9 : v === cur;
      if (same) delete nextPending[d.key];
      else nextPending[d.key] = v;
    }
    setBad(nextBad);
    setPending(nextPending);
  };

  const [fx, setFx] = useState<Record<string, "ok" | "bad">>({});
  const flashKeys = (keys: string[], kind: "ok" | "bad") => {
    setFx(Object.fromEntries(keys.map((k) => [k, kind])));
    setTimeout(() => setFx({}), 1500);
  };
  const apply = async () => {
    setBusy(true);
    const keys = Object.keys(pending);
    const r: PatchResult | null = await actions.patchConfig(pending);
    setBusy(false);
    if (!r) return flashKeys(keys, "bad");
    flashKeys(keys, "ok");
    setPending({});
    setRestartKeys((k) => [...new Set([...k, ...r.pending_restart])]);
  };
  const restart = async () => {
    if (await actions.restart()) setRestartKeys([]);
  };

  const nPending = Object.keys(pending).length;
  const nBad = Object.keys(bad).length;

  return (
    <div className="set-pane view enter">
      <div className="cfg-top">
        <input id="param-search" className="inp" type="search" placeholder="Поиск: плечо, стоп, RISK_PER_TRADE…  (Ctrl/⌘ K)" aria-label="Поиск параметра" autoComplete="off" value={q} onChange={(e) => setQ(e.target.value)} />
        <span className="label">{needle ? `найдено ${found}` : `${found} из ${schema.length}`}</span>
      </div>
      <div className="row-tgl">
        <span>
          Показать расширенные <Q tip="По умолчанию видны основные параметры. Расширенные — тонкая настройка фильтров, выхода, обучения и сервера." />
        </span>
        <Toggle on={advanced} label="Показать расширенные параметры" onChange={setAdvanced} />
      </div>
      <p className="cfg-legend">
        <span className="tag hot">сразу</span> применяется без перезапуска · <span className="tag rst">перезапуск</span> вступит в силу после
        перезапуска бота. Ключи — в «Подключения», ключи AI — в Бот → AI-цепочка, Paper/Live — в «Основное».
      </p>
      {!needle && elsewhere > 0 && (
        <p className="cfg-legend">
          {elsewhere} {plural(elsewhere, ["настройка", "настройки", "настроек"])} здесь не показано: у них есть свои разделы (риск, открытие сделок, режим, часы, AI,
          обучение). Найти любую и перейти к ней можно поиском.
        </p>
      )}
      <div className="cfg">
        {groups.map(([g, defs]) => {
          const rows = defs.filter(hit);
          if (!rows.length) return null;
          const changed = rows.filter((d) => d.key in pending).length;
          return (
            <details
              className="grp"
              key={g}
              open={Boolean(needle) || open.has(g)}
              onToggle={(e) => {
                if (needle) return;
                const next = new Set(open);
                if ((e.currentTarget as HTMLDetailsElement).open) next.add(g);
                else next.delete(g);
                setOpen(next);
              }}
            >
              <summary>
                <span>{g}</span>
                <span className="cnt">{rows.length}</span>
                {changed > 0 && <span className="chgb">изменено {changed}</span>}
              </summary>
              {rows.map((d) =>
                OWN_HOME[d.key] ? (
                  <HomeRow key={d.key} d={d} go={go} />
                ) : (
                <ParamRow key={d.key} d={d} value={d.key in pending ? pending[d.key] : config[d.key]?.value} source={config[d.key]?.source ?? "default"} changed={d.key in pending} err={bad[d.key]} fx={fx[d.key]} onChange={(v) => change(d, v)} onReset={() => change(d, rawOf(d, d.default))} />
                ),
              )}
            </details>
          );
        })}
      </div>
      <SettingsTools schema={schema} config={config} onImport={(v) => Object.entries(v).forEach(([k, x]) => {
        const d = schema.find((s) => s.key === k);
        if (d && !d.readonly) change(d, rawOf(d, x));
      })} />
      {(nPending > 0 || nBad > 0) && (
        <div className="applybar">
          <span>
            {nPending > 0 && (
              <>
                Изменено <b>{nPending}</b> {plural(nPending, ["параметр", "параметра", "параметров"])}
              </>
            )}
            {nBad > 0 && <span className="down">{`${nPending ? " · " : ""}исправьте ошибки: ${nBad}`}</span>}
          </span>
          <span className="sp">
            <button
              className="btn"
              type="button"
              onClick={() => {
                setPending({});
                setBad({});
              }}
            >
              Отменить
            </button>
            <button className="btn primary" type="button" disabled={busy || nBad > 0 || !nPending} onClick={() => void apply()}>
              Применить
            </button>
          </span>
        </div>
      )}
      {!nPending && !nBad && restartKeys.length > 0 && (
        <div className="applybar">
          <span>
            <b>{restartKeys.length}</b> {plural(restartKeys.length, ["настройка вступит", "настройки вступят", "настроек вступят"])} в силу после
            перезапуска бота
          </span>
          <span className="sp">
            <button className="btn primary" type="button" onClick={() => void restart()}>
              Перезапустить бота
            </button>
          </span>
        </div>
      )}
    </div>
  );
}

/** Параметр с собственным разделом: вместо второго поля — ссылка туда. */
function HomeRow({ d, go }: { d: SettingDef; go: (pane: Pane) => void }) {
  const home = OWN_HOME[d.key];
  const setTab = useStore((s) => s.setTab);
  const openNode = useStore((s) => s.openNode);
  const open = () => {
    if (home.tab === "settings" && home.pane) go(home.pane);
    else if (home.node) openNode(home.node);
    else setTab(home.tab);
  };
  return (
    <div className="prm home">
      <div className="prm-l">
        <label>{d.label}</label>
        <code>{d.key}</code>
        {d.hint && <p>{d.hint}</p>}
      </div>
      <div className="prm-c">
        <span className="tag hot">{home.where}</span>
        <button className="btn" type="button" onClick={open}>
          Открыть
        </button>
      </div>
    </div>
  );
}

/** Значение в том виде, в каком поле его показывает (проценты — в %, а не долей). */
const rawOf = (d: SettingDef, v: unknown): unknown => (d.type === "pct" && typeof v === "number" ? +(v * 100).toFixed(6) : v);

export const sameVal = (a: unknown, b: unknown): boolean =>
  typeof a === "number" && typeof b === "number" ? Math.abs(a - b) < 1e-9 : a === b;

function ParamRow(props: { d: SettingDef; value: unknown; source: string; changed: boolean; err?: string; fx?: "ok" | "bad"; onChange: (v: unknown) => void; onReset: () => void }) {
  const { d, value } = props;
  const id = `c-${d.key}`;
  let control: JSX.Element;
  if (d.type === "bool") control = <Toggle on={Boolean(value)} label={d.label} disabled={d.readonly} onChange={props.onChange} />;
  else if (d.type === "choice")
    control = (
      <select className="sel" id={id} value={String(value ?? "")} disabled={d.readonly} onChange={(e) => props.onChange(e.target.value)}>
        {d.options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    );
  else if (d.type === "str")
    control = <input className="inp" id={id} value={String(value ?? "")} spellCheck={false} autoComplete="off" disabled={d.readonly} onChange={(e) => props.onChange(e.target.value)} />;
  else
    control = (
      <NumField id={id} d={d} value={Number(shown(d, value))} disabled={d.readonly} onChange={props.onChange} />
    );
  return (
    <div className={`prm${props.changed ? " chg" : ""}${props.err ? " bad" : ""}${props.fx ? ` fx-${props.fx}` : ""}`}>
      <div className="prm-l">
        <label htmlFor={id}>{d.label}</label>
        <code>{d.key}</code>
        {d.hint && <p>{d.hint}</p>}
        <p className="def">
          по умолчанию: {fmtVal(d, d.default)}
          {SOURCE[props.source] ? ` · ${SOURCE[props.source]}` : ""}
        </p>
      </div>
      <div className="prm-c">
        {d.readonly ? <span className="tag rst">только .env</span> : <span className={`tag ${d.hot ? "hot" : "rst"}`}>{d.hot ? "сразу" : "перезапуск"}</span>}
        {control}
        {props.fx === "ok" && (
          <svg className="prm-tick" viewBox="0 0 24 24" width="18" height="18" aria-label="Применено" role="img">
            <path d="M5 12.5l4.5 4.5L19 7.5" />
          </svg>
        )}
        {!d.readonly && !sameVal(value, d.default) && (
          <button className="btn reset" type="button" title={`Вернуть по умолчанию: ${fmtVal(d, d.default)}`} aria-label={`${d.label}: вернуть по умолчанию`} onClick={props.onReset}>
            ↺
          </button>
        )}
      </div>
      {props.err && <p className="err">{props.err}</p>}
    </div>
  );
}

/** Число редактируется как текст: промежуточное «0.» или пустое поле не сбрасывается. */
export function NumField({ id, d, value, disabled, onChange }: { id: string; d: SettingDef; value: number; disabled: boolean; onChange: (v: string) => void }) {
  const [text, setText] = useState(String(value));
  useEffect(() => {
    if (parseFloat(text.replace(",", ".")) !== value) setText(String(value));
  }, [value]); // eslint-disable-line react-hooks/exhaustive-deps
  const step = d.type === "pct" ? (d.step ?? 0.0001) * 100 : (d.step ?? (d.type === "int" ? 1 : 0.01));
  return (
    <div className="numf">
      <input
        id={id}
        type="number"
        inputMode="decimal"
        step={step}
        value={text}
        disabled={disabled}
        onChange={(e) => {
          setText(e.target.value);
          onChange(e.target.value);
        }}
      />
      <span className="u">{unitOf(d)}</span>
    </div>
  );
}
