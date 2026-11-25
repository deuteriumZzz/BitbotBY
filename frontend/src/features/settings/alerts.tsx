// Настройки → Основное: уведомления — тихие часы, уровни, cooldown тревог.

import { actions } from "../../store/actions";
import { useStore } from "../../store";
import { Q, Seg, Toggle } from "../../ui";
import { cfg } from "../bots";

// Тихие часы выбираются по местному времени, а бот хранит их в UTC (QUIET_HOURS).
const TZ = Math.round(-new Date().getTimezoneOffset() / 60);

const shiftHour = (h: number, by: number) => (((h + by) % 24) + 24) % 24;

const shiftRange = (r: string, by: number) => (r ? r.split("-").map((h) => shiftHour(Number(h), by)).join("-") : "");

const QUIET = [
  ["", "Без тишины"],
  ["23-8", "23–08"],
  ["0-7", "00–07"],
  ["22-9", "22–09"],
] as const;

const LEVEL: Record<string, string> = { critical: "приходит всегда", trade: "сделки", info: "сведения" };

export function AlertsCard() {
  const alerts = useStore((s) => s.alerts);
  const quietUtc = useStore((s) => cfg<string>(s.config, "QUIET_HOURS", ""));
  const quietLocal = shiftRange(quietUtc, TZ);
  const preset = QUIET.find(([v]) => v === quietLocal)?.[0] ?? null;
  return (
    <article className="card">
      <div className="card-h">
        <h2>Уведомления в Telegram</h2>
        <Q tip="Пауза не даёт одному и тому же уведомлению сыпаться чаще, чем раз в N минут." />
      </div>
      <div className="field">
        <div className="field-h">
          <span>
            Тихие часы, по вашему времени{" "}
            <Q tip="В эти часы приходят только уведомления с пометкой «приходит всегда»: бот не отвечает, просадка, дневной лимит, всплеск волатильности, позиция без защиты. Сделки и сигналы — утром." />
          </span>
          {quietUtc && <span className="pill sp">{preset === null ? `свои: ${quietLocal.replace("-", "–")}` : `UTC ${quietUtc.replace("-", "–")}`}</span>}
        </div>
        <Seg className="full" value={preset} options={QUIET} label="Тихие часы" onChange={(v) => void actions.patchConfig({ QUIET_HOURS: shiftRange(v, -TZ) })} />
      </div>
      <div className="list-tgl">
        {alerts.map((a) => (
          <div className={`alert-row${a.enabled ? "" : " off"}`} key={a.kind}>
            <div>
              <b>{a.label}</b>
              {a.level && <span className={`lvl ${a.level}`}>{LEVEL[a.level]}</span>}
            </div>
            <Toggle on={a.enabled} label={a.label} onChange={(on) => void actions.patchAlert(a.kind, { enabled: on })} />
            <label className="cool">
              <span>Не чаще раза в</span>
              <span className="numf">
                <input
                  key={a.cooldown_min}
                  type="number"
                  min="0"
                  max="1440"
                  step="1"
                  defaultValue={a.cooldown_min}
                  inputMode="numeric"
                  disabled={!a.enabled}
                  aria-label={`Пауза для «${a.label}», минут`}
                  onBlur={(e) => {
                    const v = Math.round(Number(e.target.value));
                    if (v !== a.cooldown_min) void actions.patchAlert(a.kind, { cooldown_min: v });
                  }}
                  onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
                />
                <span className="u">мин</span>
              </span>
              <span>{a.cooldown_min ? "" : "каждый раз"}</span>
            </label>
          </div>
        ))}
      </div>
    </article>
  );
}
