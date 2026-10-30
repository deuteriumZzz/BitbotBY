// Фоновые задачи бота — всё, что есть в меню Telegram: бэктест, обучение и тюнинг SAC,
// откат модели, отчёты. Бот продолжает торговать, пока задача идёт.

import { useEffect, useLayoutEffect, useRef, useState } from "react";

import { base, fmtN, pct } from "../lib/format";
import type { JobDone, JobId, JobState, ProfileInfo } from "../lib/types";
import { actions, useStore } from "../store";
import { Q } from "../ui";
import { stratName } from "./bots";

interface JobDef {
  id: JobId;
  /** Задачи одной группы не идут одновременно (обучение и тюнинг — одна группа). */
  group: string;
  title: string;
  what: string;
  /** Что стоит знать перед запуском — показывается в подтверждении. */
  warn: string;
  button: string;
}

const JOBS: JobDef[] = [
  {
    id: "backtest",
    group: "backtest",
    title: "Бэктест стратегий",
    what: "Прогоняет все стратегии по истории свечей и показывает, как они работали. В торговле ничего не меняет.",
    warn: "Займёт несколько минут и загрузит процессор сервера. Загрузит свечи с биржи и заменит прежние результаты бэктеста.",
    button: "Запустить бэктест",
  },
  {
    id: "train",
    group: "sac",
    title: "Обучение SAC",
    what: "Переобучает нейросеть SAC на свежих данных профиля монет.",
    warn: "Около 60 минут на процессоре. Новая модель заменит текущую; прежняя сохранится, и её можно вернуть кнопкой «Откатить модель».",
    button: "Обучить модель",
  },
  {
    id: "tune",
    group: "sac",
    title: "Тюнинг и обучение SAC",
    what: "Optuna перебирает 30 вариантов параметров нейросети, затем обучает модель с лучшими. Рекомендуется раз в 1–2 месяца.",
    warn: "Около 3 часов на процессоре, два этапа. По итогу модель заменится новой, прежнюю можно будет откатить.",
    button: "Запустить тюнинг",
  },
  {
    id: "alpha_report",
    group: "alpha_report",
    title: "Значимость альфы",
    what: "Проверяет на ваших закрытых Paper-сделках, есть ли у стратегий настоящее преимущество (bootstrap Sharpe, Wilcoxon, поправка Bonferroni).",
    warn: "Несколько секунд, ничего не меняет и ничего не тратит.",
    button: "Проверить",
  },
  {
    id: "live_divergence",
    group: "live_divergence",
    title: "Live против бэктеста",
    what: "Сравнивает цену, по которой исполнялись сделки, с моделью проскальзывания из бэктеста.",
    warn: "Несколько секунд, ничего не меняет и ничего не тратит.",
    button: "Сравнить",
  },
  {
    id: "ai_isolation",
    group: "ai_isolation",
    title: "Даёт ли AI преимущество",
    what: "Отделяет вклад AI от голых индикаторов: даёт ли языковая модель что-то сверх них.",
    warn: "Делает около 150 реальных запросов к AI-провайдеру и тратит его дневной лимит. Несколько минут.",
    button: "Проверить AI",
  },
];

const num = (v: unknown): number | undefined =>
  typeof v === "number" && Number.isFinite(v) ? v : undefined;

/** Одна строка про ход: «34% · осталось ~12 мин · BTC · Свинг». */
function progressText(id: JobId, info: Record<string, unknown>): string {
  const pct = num(info.pct);
  const parts = [pct !== undefined ? `${Math.round(pct)}%` : "запускаю…"];
  const eta = num(info.eta_min);
  if (eta !== undefined) parts.push(`осталось ~${Math.round(eta)} мин`);
  if (id === "backtest" && typeof info.strategy === "string")
    parts.push(
      `${base(String(info.symbol ?? ""))} · ${stratName(info.strategy)}`,
    );
  if (id === "tune" && info.stage === "train")
    parts.push("этап 2 из 2: обучение");
  else if (id === "tune" && num(info.trial) !== undefined)
    parts.push(`попытка ${info.trial}/${info.total ?? "?"}`);
  if (id === "train" && num(info.step) !== undefined)
    parts.push(
      `шаг ${fmtN(Number(info.step), 0)}/${fmtN(Number(info.total ?? 0), 0)}`,
    );
  const pnl = num(info.best_pnl);
  if (id === "tune" && pnl !== undefined)
    parts.push(`лучший PnL $${fmtN(pnl, 0)}`);
  return parts.join(" · ");
}

function resultText(id: JobId, d: JobDone): string {
  if (!d.ok) return `Не получилось: ${d.error ?? "ошибка бота"}.`;
  if (id === "backtest")
    return `Готово: ${d.n} стратегий${d.best ? `. Лучший Sharpe — ${stratName(d.best.strategy)} (${d.best.sharpe}, сделок ${d.best.trades})` : ""}. Sharpe ниже нуля значит, что стратегия не заработала на истории.`;
  if (id === "train" || id === "tune") {
    const cmp =
      d.sac_pct != null && d.bh_pct != null
        ? ` На тестовой выборке SAC ${pct(d.sac_pct, 1)}, просто держать (Buy&Hold) ${pct(d.bh_pct, 1)}.`
        : "";
    return `Модель обучена.${cmp}${d.worse ? " Результат заметно хуже рынка — лучше откатить модель или проверить бэктестом." : ""}`;
  }
  return "Готово.";
}

function Ring({ pct }: { pct: number }) {
  const r = 14;
  const c = 2 * Math.PI * r;
  const v = Math.max(0, Math.min(100, pct));
  return (
    <svg className="ring" viewBox="0 0 36 36" width="40" height="40" role="img" aria-label={`${Math.round(v)}%`}>
      <circle className="ring-bg" cx="18" cy="18" r={r} />
      <circle className="ring-fg" cx="18" cy="18" r={r} strokeDasharray={c} strokeDashoffset={c * (1 - v / 100)} />
      <text x="18" y="21.5" textAnchor="middle">
        {Math.round(v)}
      </text>
    </svg>
  );
}

function JobRow({ def, profile }: { def: JobDef; profile?: ProfileInfo }) {
  const job: JobState | undefined = useStore((s) => s.jobs[def.id]);
  const jobs = useStore((s) => s.jobs);
  const groupSource = useStore((s) => s.jobGroups[def.group]);
  const online = useStore((s) => s.alive);
  const [confirm, setConfirm] = useState(false);
  // Обучение и тюнинг делят один замок: пока идёт одно, второе недоступно.
  const sibling = JOBS.some(
    (j) => j.group === def.group && j.id !== def.id && jobs[j.id]?.running,
  );
  // Группа занята, а у этой задачи своего прогресса нет — её запустили из Telegram (или другого окна).
  const foreign = groupSource !== undefined && !job?.running && !sibling;
  const d = job?.done;
  return (
    <li className="job">
      <div className="job-h">
        <b>{def.title}</b>
        <Q tip={def.what} />
      </div>
      {job?.running ? (
        <>
          {def.id === "train" || def.id === "tune" ? (
            <div className="ring-row">
              <Ring pct={num(job.info.pct) ?? 0} />
              <p className="note-soon">
                {progressText(def.id, job.info)}. Закрыть приложение можно — бот
                доработает сам.
              </p>
            </div>
          ) : (
            <>
              <div
                className="prog"
                role="progressbar"
                aria-valuenow={Math.round(num(job.info.pct) ?? 0)}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-label={def.title}
              >
                <i style={{ width: `${num(job.info.pct) ?? 0}%` }} />
              </div>
              <p className="note-soon">
                {progressText(def.id, job.info)}. Закрыть приложение можно — бот
                доработает сам.
              </p>
            </>
          )}
        </>
      ) : (
        <>
          {d && (
            <div className={`job-res ${d.ok ? "" : "down"}`}>
              <p className="note-soon">
                {d.ok && (
                  <svg className="chk" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
                    <circle cx="12" cy="12" r="10" />
                    <path d="M7 12.5l3.2 3.2L17 9" />
                  </svg>
                )}
                {resultText(def.id, d)}
              </p>
              {d.ok && d.output && <pre className="job-out">{d.output}</pre>}
            </div>
          )}
          {foreign && (
            <p className="note-soon">
              Идёт, запущено{" "}
              {groupSource === "telegram" ? "из Telegram" : "в другом окне"} —
              дождитесь конца.
            </p>
          )}
          {confirm ? (
            <div className="confirm-box">
              <span>
                {profile && <>Профиль «{profile.label}». </>}
                {def.warn}
              </span>
              <div className="btn-row">
                <button
                  className="btn"
                  type="button"
                  onClick={() => setConfirm(false)}
                >
                  Отмена
                </button>
                <button
                  className="btn primary"
                  type="button"
                  onClick={() => {
                    setConfirm(false);
                    void actions.runJob(def.id, profile?.id);
                  }}
                >
                  Запустить
                </button>
              </div>
            </div>
          ) : (
            <button
              className="btn"
              type="button"
              disabled={!online || sibling || foreign}
              title={
                sibling || foreign
                  ? "Идёт другая задача этой группы"
                  : undefined
              }
              onClick={() => setConfirm(true)}
            >
              {def.button}
            </button>
          )}
        </>
      )}
    </li>
  );
}

const MODEL_JOBS: JobId[] = ["train", "tune"];
const ALPHA_JOBS: JobId[] = [
  "backtest",
  "alpha_report",
  "live_divergence",
  "ai_isolation",
];

/** Нейросеть SAC участвует в решениях только в режимах SAC и Hybrid. */
function sacRole(mode: string): { used: boolean; text: string } {
  if (mode === "dqn")
    return {
      used: true,
      text: "Сейчас сделки решает именно эта нейросеть — качество обучения напрямую влияет на торговлю.",
    };
  if (mode === "hybrid")
    return {
      used: true,
      text: "Сделка открывается, когда нейросеть и AI согласны, поэтому плохая модель режет и хорошие сигналы AI.",
    };
  return {
    used: false,
    text: "Сейчас режим не использует SAC, и обучение ни на что не повлияет, пока вы не включите режим SAC или Hybrid (раздел «Цепочка» → «Кто принимает решение» вверху).",
  };
}

const when = (iso: string | null): string =>
  iso
    ? new Date(iso).toLocaleDateString("ru-RU", {
        day: "numeric",
        month: "long",
      })
    : "";

/** Выбор профиля: у каждого своя модель, таймфрейм и число монет, обучаем ту, что выбрана. */
function ProfilePicker({
  profiles,
  value,
  onChange,
}: {
  profiles: ProfileInfo[];
  value: string;
  onChange: (id: string) => void;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState<{
    x: number;
    y: number;
    w: number;
    h: number;
  } | null>(null);
  // Рамка выбранной карточки плавно переезжает на новую; следим и за шириной окна.
  const measure = () => {
    const el = wrap.current?.querySelector<HTMLElement>(
      '[aria-checked="true"]',
    );
    setBox(
      el
        ? {
            x: el.offsetLeft,
            y: el.offsetTop,
            w: el.offsetWidth,
            h: el.offsetHeight,
          }
        : null,
    );
  };
  useLayoutEffect(measure, [value, profiles.length]);
  useEffect(() => {
    if (!wrap.current || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(measure);
    ro.observe(wrap.current);
    return () => ro.disconnect();
  }, []);
  return (
    <div
      className="prof-pick"
      role="radiogroup"
      aria-label="Профиль для обучения"
      ref={wrap}
    >
      <i
        className="prof-ind"
        aria-hidden="true"
        style={
          box
            ? {
                transform: `translate(${box.x}px, ${box.y}px)`,
                width: box.w,
                height: box.h,
              }
            : { display: "none" }
        }
      />
      {profiles.map((p) => (
        <button
          key={p.id}
          type="button"
          role="radio"
          aria-checked={p.id === value}
          className={`prof ${p.id === value ? "on" : ""}`}
          onClick={() => onChange(p.id)}
        >
          <b>{p.label}</b>
          <span>
            {p.timeframe} · {p.train_top_n} монет
          </span>
          <span className="prof-m">
            {p.model_exists ? `модель от ${when(p.trained_at)}` : "не обучена"}
          </span>
          {p.active && <i className="pill ok">торгует сейчас</i>}
        </button>
      ))}
    </div>
  );
}

export function ModelCard() {
  const profiles = useStore((s) => s.profiles);
  const backupProfile = useStore((s) => s.backupProfile);
  const activeId = profiles.find((p) => p.active)?.id ?? profiles[0]?.id ?? "";
  const [picked, setPicked] = useState<string | null>(null);
  const chosen = profiles.find((p) => p.id === (picked ?? activeId));
  const mode = useStore((s) => s.bot?.mode ?? "ai");
  const canRollback = useStore((s) => s.rollbackAvailable);
  const sacBusy = useStore(
    (s) =>
      "sac" in s.jobGroups ||
      Boolean(s.jobs.train?.running || s.jobs.tune?.running),
  );
  const online = useStore((s) => s.alive);
  const [confirm, setConfirm] = useState(false);
  const role = sacRole(mode);
  return (
    <article className="card">
      <div className="card-h">
        <h2>Нейросеть SAC</h2>
        <span className={`pill sp ${role.used ? "ok" : ""}`}>
          {role.used ? "участвует в решениях" : "сейчас не используется"}
        </span>
      </div>
      <p className="note-soon role">{role.text}</p>
      <ul className="jobs">
        {chosen && (
          <li className="job">
            <div className="job-h">
              <b>Какой профиль обучать</b>
              <Q tip="У каждого профиля рынка своя модель, свой таймфрейм и своё число монет в обучении. Обучение и тюнинг ниже работают с выбранным профилем; торговать продолжает активный." />
            </div>
            <ProfilePicker
              profiles={profiles}
              value={chosen.id}
              onChange={setPicked}
            />
            {!chosen.active && (
              <p className="note-soon">
                Этот профиль сейчас не торгует: новая модель начнёт работать,
                когда вы переключитесь на него (раздел «Цепочка»).
              </p>
            )}
          </li>
        )}
        {MODEL_JOBS.map((id) => (
          <JobRow
            key={id}
            def={JOBS.find((j) => j.id === id)!}
            profile={chosen}
          />
        ))}
        <li className="job">
          <div className="job-h">
            <b>Откат модели SAC</b>
            <Q tip="Возвращает модель, которая была до последнего обучения. Бот подхватит её в следующем цикле." />
          </div>
          {!canRollback ? (
            <p className="note-soon">
              Резервной копии пока нет: она появляется после первого
              переобучения.
            </p>
          ) : confirm ? (
            <div className="confirm-box">
              <span>
                Модель профиля «
                {profiles.find((p) => p.id === backupProfile)?.label ??
                  "активного"}
                » будет заменена той, что была до последнего обучения.
              </span>
              <div className="btn-row">
                <button
                  className="btn"
                  type="button"
                  onClick={() => setConfirm(false)}
                >
                  Отмена
                </button>
                <button
                  className="btn danger"
                  type="button"
                  onClick={() => {
                    setConfirm(false);
                    void actions.rollbackModel();
                  }}
                >
                  Откатить
                </button>
              </div>
            </div>
          ) : (
            <button
              className="btn"
              type="button"
              disabled={!online || sacBusy}
              title={sacBusy ? "Идёт обучение" : undefined}
              onClick={() => setConfirm(true)}
            >
              Откатить модель
            </button>
          )}
        </li>
      </ul>
    </article>
  );
}

function AlphaCard() {
  return (
    <article className="card">
      <div className="card-h">
        <h2>Поиск преимущества (альфы)</h2>
        <Q tip="Альфа — это то, что стратегия зарабатывает сверх случайности. Эти проверки отвечают на вопрос: есть ли у ваших стратегий и у AI настоящее преимущество, или пока везение." />
      </div>
      <p className="note-soon role">
        Бэктест прогоняет стратегии по истории; значимость альфы проверяет ваши
        закрытые Paper-сделки статистикой; остальные две сравнивают реальное
        исполнение с моделью и отделяют вклад AI от индикаторов. Всё в фоне, бот
        продолжает торговать.
      </p>
      <ul className="jobs">
        {ALPHA_JOBS.map((id) => (
          <JobRow key={id} def={JOBS.find((j) => j.id === id)!} />
        ))}
      </ul>
    </article>
  );
}

/** Вкладка «Обучение»: нейросеть и поиск преимущества — всё, что можно улучшить и проверить. */
export function Learn() {
  return (
    <section className="view enter" aria-label="Обучение">
      <div className="cols">
        <div className="col">
          <ModelCard />
        </div>
        <div className="col">
          <AlphaCard />
        </div>
      </div>
    </section>
  );
}
