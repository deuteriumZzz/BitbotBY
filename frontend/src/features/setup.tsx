// Первый запуск: карточка «Готовность» на терминале и мастер из пяти шагов.
// Шаги собраны из готовых карточек: ключи — из «Подключений», AI — из раздела AI,
// риск — из «Ботов».

import { useEffect, useState, type ReactNode } from "react";

import type { AiState, SecretItem, SetupInfo, SetupStep } from "../lib/types";
import { actions } from "../store/actions";
import { useStore } from "../store";
import { Icon } from "../components/icon";
import { Skel } from "../components/motion";
import { KeysCard } from "./ai";
import { RiskCard, presetName, presetOf } from "./bots";
import { KeyCard } from "./settings/keys";

const STEPS: SetupStep[] = ["bybit", "telegram", "ai", "risk", "mode"];
const SEEN = "bitbot.setup.seen";
const HIDDEN = "bitbot.setup.hidden";

export interface Check {
  step: SetupStep;
  title: string;
  ok: boolean;
  /** Без этого бот не работает как надо (не просто «желательно»). */
  need: boolean;
  text: string;
}

/** Что уже настроено — по данным сервера. */
export function readiness(info: SetupInfo | null, ai: AiState | null, preset: string, paper: boolean, mode = "ai"): Check[] {
  const n = ai?.chain.length ?? 0;
  const local = mode === "local";
  return [
    {
      step: "bybit",
      title: "Биржа Bybit",
      ok: Boolean(info?.bybit),
      need: !paper,
      text: info?.bybit ? "ключи на месте" : paper ? "нужны только для реальной торговли" : "без ключей Live не торгует",
    },
    {
      step: "telegram",
      title: "Telegram",
      ok: Boolean(info?.telegram),
      need: false,
      text: info?.telegram ? "уведомления и подтверждения в чате" : "необязательно — приложение работает и без него",
    },
    {
      step: "ai",
      title: "AI-анализ",
      ok: local || n > 0,
      need: !local,
      text: local
        ? "не нужен: решают локальные стратегии"
        : n
          ? `подключено провайдеров: ${n}`
          : "не подключён — сигналы без проверки моделью",
    },
    { step: "risk", title: "Риск", ok: true, need: false, text: preset },
    { step: "mode", title: "Счёт", ok: true, need: false, text: paper ? "Paper — виртуальные деньги" : "Live — реальные деньги" },
  ];
}

function useChecks(): Check[] {
  const info = useStore((s) => s.setupInfo);
  const ai = useStore((s) => s.ai);
  const config = useStore((s) => s.config);
  const paper = useStore((s) => s.bot?.paper ?? true);
  const mode = useStore((s) => s.bot?.mode ?? "ai");
  const presets = useStore((s) => s.riskPresets);
  return readiness(info, ai, presetName(presetOf(config, presets)), paper, mode);
}

// Флаги только для этого устройства: без хранилища считаем, что мастер уже видели,
// чтобы не навязывать его при каждом запуске.
const flag = (key: string): boolean => {
  try {
    return localStorage.getItem(key) === "1";
  } catch {
    return true;
  }
};
const setFlag = (key: string): void => {
  try {
    localStorage.setItem(key, "1");
  } catch {
    /* нет хранилища — не страшно */
  }
};

// ── Чип в строке статуса на «Обзоре» ─────────────────────────────────────────

/** Готовность к работе: «3 из 5 →» открывает мастер на первом незаполненном шаге.
 *  Совсем новому боту мастер открывается сам — один раз на устройстве. */
export function ReadinessChip() {
  const info = useStore((s) => s.setupInfo);
  const ai = useStore((s) => s.ai);
  const open = useStore((s) => s.openSetup);
  const checks = useChecks();
  const missing = checks.filter((c) => !c.ok);
  const needed = missing.some((c) => c.need);
  const [hidden, setHidden] = useState(() => flag(HIDDEN));

  useEffect(() => {
    if (!info || !ai || flag(SEEN)) return;
    setFlag(SEEN);
    if (!info.bybit && !info.telegram && !ai.chain.length) {
      // Совсем новый человек — начинает в «Простом»; «Профи» в шапке, одним нажатием.
      useStore.getState().setUi("simple");
      open("bybit");
    }
  }, [info, ai, open]);

  // Скрыть можно, только когда остались необязательные пункты.
  if (!info || !missing.length || (hidden && !needed)) return null;
  const done = checks.length - missing.length;
  return (
    <li className="health-act">
      <span className={`dot ${needed ? "bad" : "warn"}`} />
      <small>Готовность</small>
      <button type="button" title={missing.map((c) => `${c.title}: ${c.text}`).join("\n")} onClick={() => open(missing[0].step)}>
        {done} из {checks.length} · {needed ? "настроить" : missing[0].title.toLowerCase()} →
      </button>
      {!needed && (
        <button
          type="button"
          className="health-x"
          aria-label="Скрыть готовность"
          onClick={() => {
            setFlag(HIDDEN);
            setHidden(true);
          }}
        >
          ×
        </button>
      )}
    </li>
  );
}

// ── Мастер ────────────────────────────────────────────────────────────────────

export function Setup() {
  const step = useStore((s) => s.setup) ?? "bybit";
  const open = useStore((s) => s.openSetup);
  const checks = useChecks();
  const i = STEPS.indexOf(step);
  const check = checks[i];
  const last = i === STEPS.length - 1;
  return (
    <section className="view enter setup" aria-label="Первый запуск">
      <div className="setup-h">
        <span className="label">
          Первый запуск · шаг {i + 1} из {STEPS.length}
        </span>
        <button className="btn sp" type="button" onClick={() => open(null)}>
          Позже
        </button>
      </div>
      <ol className="setup-dots">
        {STEPS.map((s, k) => (
          <li key={s}>
            <button
              type="button"
              aria-current={k === i ? "step" : undefined}
              className={checks[k].ok ? "ok" : ""}
              aria-label={`${k + 1}. ${checks[k].title}`}
              onClick={() => open(s)}
            />
          </li>
        ))}
      </ol>
      <h2 className="setup-t">
        {check.title}
        {check.ok && <span className="pill ok">{check.text}</span>}
      </h2>
      <StepBody step={step} />
      <div className="setup-f">
        {i > 0 && (
          <button className="btn" type="button" onClick={() => open(STEPS[i - 1])}>
            Назад
          </button>
        )}
        <span className="sp" />
        {last ? (
          <button className="btn primary" type="button" onClick={() => open(null)}>
            Готово
          </button>
        ) : (
          <button className="btn primary" type="button" onClick={() => open(STEPS[i + 1])}>
            {check.ok || !check.need ? "Дальше" : "Пропустить"}
          </button>
        )}
      </div>
    </section>
  );
}

function StepBody({ step }: { step: SetupStep }) {
  const info = useStore((s) => s.setupInfo);
  const secrets = useStore((s) => s.secrets);
  const ai = useStore((s) => s.ai);
  const paper = useStore((s) => s.bot?.paper ?? true);
  const mode = useStore((s) => s.bot?.mode ?? "ai");
  const key = (service: string, envs: string): ReactNode => {
    const item = secrets?.find((s: SecretItem) => s.service === service);
    if (info?.keys_in_app && item) return <KeyCard item={item} />;
    return (
      <p className="setup-p">
        Сохранять ключи из приложения нельзя: на сервере не задан <code>BITBOT_SECRETS_KEY</code>. Впишите <code>{envs}</code> в{" "}
        <code>.env</code> и перезапустите бота.
      </p>
    );
  };

  switch (step) {
    case "bybit":
      return (
        <>
          <p className="setup-p">
            Ключи нужны, чтобы бот торговал на реальном счёте. В Paper цены берутся с биржи без ключа — шаг можно пропустить и вернуться
            перед переходом на Live.
          </p>
          {key("bybit", "BYBIT_API_KEY и BYBIT_API_SECRET")}
        </>
      );
    case "telegram":
      return (
        <>
          <p className="setup-p">
            Бот пишет в Telegram о сделках и защите, а при подтверждении сделок присылает кнопки «Trade / Skip». Необязательно: всё это есть и в
            приложении.
          </p>
          {key("telegram", "TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID")}
        </>
      );
    case "ai": {
      if (mode === "local")
        return (
          <>
            <p className="setup-p">
              Сейчас бот работает <b>без AI</b>: сделки открывают только локальные стратегии по индикаторам. Это бесплатно и предсказуемо, но
              ни одна из стратегий пока не показала доказанного преимущества — держите Paper.
            </p>
            <button className="btn" type="button" onClick={() => void actions.patchConfig({ MODE: "ai" })}>
              Вернуть AI
            </button>
          </>
        );
      if (!ai) return <Skel label="Загружаю список провайдеров" />;
      // Бесплатные провайдеры, к которым ещё нет ключа (ключ вводится прямо здесь).
      return (
        <>
          <p className="setup-p">
            Перед сделкой бот спрашивает языковую модель. Хватит одного бесплатного ключа; чем их больше, тем реже бот остаётся без AI из-за
            дневных лимитов. Порядок бот выбирает сам: сначала бесплатные.
          </p>
          <button className="btn" type="button" onClick={() => void actions.patchConfig({ MODE: "local" })}>
            Работать без AI: только локальные стратегии
          </button>
          {info?.keys_in_app && <KeysCard ai={ai} secrets={secrets ?? []} onSaved={() => undefined} />}
        </>
      );
    }
    case "risk":
      return <RiskCard />;
    case "mode":
      return (
        <>
          <p className="setup-p">
            <b>Paper</b> — тренировочный счёт: настоящие цены, виртуальные деньги. Посмотрите журнал сделок за одну-две недели, прежде чем
            включать Live.
          </p>
          {!paper && (
            <div className="honest">
              {Icon.warn}
              <div>
                <b>Сейчас включён Live — бот торгует реальными деньгами.</b> Вернуть Paper можно в Настройки → Счёт и компьютер.
              </div>
            </div>
          )}
          <p className="setup-p">Новые ключи начинают работать после перезапуска бота.</p>
          <button className="btn" type="button" onClick={() => void actions.restart()}>
            Перезапустить бота
          </button>
        </>
      );
  }
}
