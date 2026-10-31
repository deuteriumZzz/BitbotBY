// Поля настроек списком — числом или тумблером — для карточек, где у параметров свой дом
// («Как рискуем», «Сигналы для решения», «Каналы Telegram»). Отдельный модуль, чтобы
// bots.tsx и settings.tsx могли его брать без циклического импорта.

import type { SettingDef } from "../lib/types";
import { actions, useStore } from "../store";
import { Toggle } from "../ui";

/** Значение так, как его вводят: проценты — в процентах. */
export const shown = (d: SettingDef, v: unknown) => (d.type === "pct" ? +(Number(v) * 100).toFixed(4) : v);
export const unitOf = (d: SettingDef) => (d.type === "pct" ? "%" : d.unit);

/** Те, что применяются только после перезапуска бота, — со значком ↻. mark — «· сейчас» (значение текущего счёта). */
export function ParamList({ keys, mark }: { keys: string[]; mark?: string }) {
  const config = useStore((s) => s.config);
  const schema = useStore((s) => s.schema);
  const defs = keys.map((k) => schema.find((d) => d.key === k)).filter((d): d is SettingDef => Boolean(d));
  return (
    <div className="tgch-params">
      {defs.map((d) => {
        const v = config[d.key]?.value ?? d.default;
        const label = (
          <span title={d.hint || undefined}>
            {d.label}
            {d.key === mark && " · сейчас"}
            {!d.hot && (
              <small className="prm-restart" title="Заработает после перезапуска бота" aria-label="после перезапуска">
                {" "}
                ↻
              </small>
            )}
          </span>
        );
        if (d.type === "bool")
          return (
            <div className="cool" key={d.key}>
              {label}
              <Toggle on={Boolean(v)} label={d.label} onChange={(on) => void actions.patchConfig({ [d.key]: on })} />
            </div>
          );
        return (
          <label className="cool" key={d.key}>
            {label}
            <span className="numf">
              <input
                key={String(config[d.key]?.value)}
                type="number"
                inputMode="decimal"
                step={d.type === "pct" ? (d.step ?? 0.0001) * 100 : (d.step ?? (d.type === "int" ? 1 : 0.01))}
                min={d.min === null ? undefined : Number(shown(d, d.min))}
                max={d.max === null ? undefined : Number(shown(d, d.max))}
                defaultValue={String(shown(d, v))}
                onBlur={(e) => {
                  const n = Number(e.target.value.replace(",", "."));
                  const raw = d.type === "pct" ? +(n / 100).toFixed(6) : n;
                  if (e.target.value && Number.isFinite(n) && raw !== config[d.key]?.value) void actions.patchConfig({ [d.key]: raw });
                }}
                onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
              />
              <span className="u">{unitOf(d)}</span>
            </span>
          </label>
        );
      })}
    </div>
  );
}
