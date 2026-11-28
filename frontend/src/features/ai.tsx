// Раздел «AI»: два ключа (основной и запасной) и табло честности прогнозов (живёт в «Обучении»).
// Провайдера бот узнаёт по ключу, порядок и модель выбирает сам (bitbot.config.ai_providers.chain).

import { useEffect, useState } from "react";

import { ApiError, api } from "../lib/api";
import { EP } from "../api/endpoints";
import { fmtN, plural } from "../lib/format";
import { detectProvider } from "../lib/keys";
import type { AiProvider, AiScoreboard, AiState, ScoreKind, SecretItem } from "../lib/types";
import { actions } from "../store/actions";
import { useStore } from "../store";
import { Icon } from "../components/icon";
import { Skel } from "../components/motion";
import { Q } from "../components/tooltip";
import { Avatar, STATUS, hhmm } from "./chainops";

export function AiPane() {
  const ai = useStore((s) => s.ai);
  const secrets = useStore((s) => s.secrets);
  const [needRestart, setNeedRestart] = useState(false);
  const local = useStore((s) => s.bot?.mode === "local");
  if (!ai) return <Skel rows={3} label="Загружаю ключи AI" />;
  return (
    <div className="set-pane view enter">
      {local && (
        <div className="honest">
          {Icon.warn}
          <div>
            <b>Сейчас AI не используется.</b> Включён режим без AI, ключи ниже бот не спрашивает.{" "}
            <button className="btn" type="button" onClick={() => void actions.patchConfig({ MODE: "ai" })}>
              Вернуть AI
            </button>
          </div>
        </div>
      )}
      <KeysCard ai={ai} secrets={secrets ?? []} onSaved={() => setNeedRestart(true)} />
      {needRestart && (
        <div className="applybar">
          <span>Новые ключи заработают после перезапуска бота</span>
          <span className="sp">
            <button
              className="btn primary"
              type="button"
              onClick={() => void actions.restart().then((ok) => ok && setNeedRestart(false))}
            >
              Перезапустить бота
            </button>
          </span>
        </div>
      )}
    </div>
  );
}

/** Ключи AI: основной и запасной. Провайдера бот узнаёт по ключу, модель и порядок
 * (сначала бесплатные) выбирает сам — bitbot.config.ai_providers.chain. */
function KeysCard(props: { ai: AiState; secrets: SecretItem[]; onSaved: () => void }) {
  const { ai } = props;
  const order = ai.chain.map((c) => c.provider);
  const withKey = order.map((id) => ai.providers.find((p) => p.id === id)!).filter(Boolean);
  const free = ai.providers.filter((p) => p.free_tier && !p.key_set && p.signup);
  return (
    <article className="card">
      <div className="card-h">
        <h2>Ключи AI</h2>
        <Q tip="Вставьте ключ любого провайдера: Groq, Gemini, DeepSeek, OpenAI или Claude. Бот сам узнает провайдера, проверит ключ и выберет модель. Второй ключ — запасной, на случай лимита или сбоя." />
      </div>
      {withKey.length > 0 && (
        <ul className="ai-list">
          {withKey.map((p, i) => {
            const [cls, text] = STATUS[p.status] ?? ["warn", p.status];
            const paused = p.cooldown_until && p.cooldown_until * 1000 > Date.now();
            return (
              <li key={p.id}>
                <Avatar id={p.id} name={p.name} />
                <div className="ai-main">
                  <b>{p.name}</b> <small className="muted">{i === 0 ? "основной" : "запасной"}</small>
                  <br />
                  <small className="muted">
                    сегодня {p.calls_today} {plural(p.calls_today, ["запрос", "запроса", "запросов"])}
                    {p.tokens_today > 0 && ` · ${fmtN(p.tokens_today, 0)} токенов`}
                  </small>
                </div>
                <span className={`pill sp ${paused ? "warn" : cls}`}>{paused ? `пауза до ${hhmm(p.cooldown_until!)}` : text}</span>
                <button className="btn" type="button" onClick={() => void actions.deleteSecret(p.id)}>
                  Убрать
                </button>
              </li>
            );
          })}
        </ul>
      )}
      {withKey.length < 2 && (
        <KeyInput ai={ai} secrets={props.secrets} label={withKey.length ? "Запасной ключ" : "Основной ключ"} onSaved={props.onSaved} />
      )}
      {!withKey.length && free.length > 0 && (
        <p className="note-soon">
          Бесплатный ключ:{" "}
          {free.map((p, i) => (
            <span key={p.id}>
              {i > 0 && ", "}
              <a href={p.signup} target="_blank" rel="noopener noreferrer">
                {p.name} ↗
              </a>
            </span>
          ))}
          . Подписка Claude Code не подходит, нужен API-ключ.
        </p>
      )}
    </article>
  );
}

// ── Честность прогнозов ───────────────────────────────────────────────────────

/** Табло честности одной строкой: проверено, угадывает, вердикт словами. */
export function scoreLine(k: ScoreKind, minN: number): string {
  const share = `${Math.round(k.hit_rate * 100)}%`;
  if (k.verdict === "мало данных") return `проверено ${k.n} из ${minN} — пока рано судить`;
  const skill = `${k.skill > 0 ? "+" : ""}${Math.round(k.skill * 100)}%`;
  return `${k.verdict} · ${k.n} проверок · исходов вверх ${share} · точность против 50/50: ${skill}`;
}

/** Табло честности: каждый прогноз AI сверяется с тем, что потом сделала цена. */
export function useScoreboard(): AiScoreboard | null {
  const [s, setS] = useState<AiScoreboard | null>(null);
  useEffect(() => {
    api<AiScoreboard>("GET", EP.aiScoreboard).then(setS).catch(() => setS(null));
  }, []);
  return s;
}

export function HonestyCard() {
  const ai = useStore((s) => s.ai);
  const s = useScoreboard();
  const rows = s ? Object.entries(s.providers) : [];
  const name = (id: string) => ai?.providers.find((p) => p.id === id)?.name ?? id;
  return (
    <article className="card">
      <div className="card-h">
        <h2>Честность прогнозов</h2>
        <Q tip="AI отвечает вероятностями: «цена сначала пройдёт +ATR, а не −ATR, за 16 свечей — 58%». Через 16 свечей бот проверяет, что было на самом деле. Если прогнозы провайдера на 200+ проверках не точнее, чем «всегда 50/50», он теряет право открывать сделки (может только промолчать)." />
      </div>
      {!s || !rows.length ? (
        <p className="note-soon">Прогнозов пока нет: табло заполнится, когда бот поработает в paper с новым запросом к AI.</p>
      ) : (
        <ul className="prot-list">
          {rows.map(([id, kinds]) => (
            <li key={id}>
              <span>
                <b>{name(id)}</b>
                <br />
                <small>{kinds.forecast ? scoreLine(kinds.forecast, s.min_n) : "прогнозов нет"}</small>
                {kinds.trade && (
                  <>
                    <br />
                    <small>сделки: {scoreLine(kinds.trade, s.min_n)}</small>
                  </>
                )}
              </span>
              <span className={`pill sp ${kinds.forecast?.verdict === "лучше монетки" ? "ok" : ""}`}>
                {kinds.forecast?.verdict ?? "—"}
              </span>
            </li>
          ))}
        </ul>
      )}
    </article>
  );
}


// ── Поле ключа: вставили — провайдер определился сам ──────────────────────────

/** Сохраняет ключ первому подходящему провайдеру. Для `sk-` (OpenAI или DeepSeek)
 * пробует по очереди: сервер отклоняет чужой ключ (422 key_rejected) и не сохраняет его. */
async function saveKey(candidates: string[], key: string, secrets: SecretItem[]): Promise<string | null> {
  for (const id of candidates) {
    const s = secrets.find((x) => x.service === id);
    const field = s?.fields.find((f) => f.secret) ?? s?.fields[0];
    if (!field) continue;
    try {
      await api("PUT", EP.secret(id), { fields: { [field.env]: key } });
      return id;
    } catch (e) {
      if (!(e instanceof ApiError) || e.code !== "key_rejected" || id === candidates[candidates.length - 1]) throw e;
    }
  }
  return null;
}

function KeyInput(props: { ai: AiState; secrets: SecretItem[]; label: string; onSaved: () => void }) {
  const { ai } = props;
  const [val, setVal] = useState("");
  const [pick, setPick] = useState("");
  const [busy, setBusy] = useState(false);
  const [over, setOver] = useState(false);
  const say = useStore((s) => s.say);
  const guess = detectProvider(val);
  const candidates = guess.length ? guess : pick ? [pick] : [];
  const take = async (e: React.DragEvent) => {
    e.preventDefault();
    setOver(false);
    const file = e.dataTransfer.files[0];
    const text = file ? await file.text() : e.dataTransfer.getData("text");
    if (text.trim()) setVal(text.trim().split(/\s+/)[0]);
  };
  const save = async () => {
    setBusy(true);
    try {
      const id = await saveKey(candidates, val.trim(), props.secrets);
      if (id) {
        await useStore.getState().loadSecrets();
        say(`Ключ ${ai.providers.find((p) => p.id === id)?.name ?? id} сохранён. Заработает после перезапуска бота.`);
        setVal("");
        setPick("");
        props.onSaved();
      }
    } catch (e) {
      say(e instanceof ApiError ? e.message : "Не удалось сохранить ключ.");
    } finally {
      setBusy(false);
    }
  };
  const name = candidates.length === 1 ? ai.providers.find((p) => p.id === candidates[0])?.name : null;
  return (
    <div className={`quick${over ? " over" : ""}`} onDragOver={(e) => (e.preventDefault(), setOver(true))} onDragLeave={() => setOver(false)} onDrop={(e) => void take(e)}>
      <div className="quick-row">
        <svg className="lock" viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
          <rect x="5" y="11" width="14" height="9" rx="2" />
          <path className="shackle" d="M8 11V8a4 4 0 0 1 8 0v3" />
        </svg>
        <input
          className="inp"
          type="password"
          autoComplete="off"
          spellCheck={false}
          placeholder={`${props.label}: вставьте или перетащите`}
          aria-label={props.label}
          value={val}
          onChange={(e) => setVal(e.target.value)}
        />
      </div>
      {val.trim().length >= 8 && (
        <div className="quick-guess" aria-live="polite">
          {name && <span>Это {name}.</span>}
          {guess.length > 1 && <span>Это OpenAI или DeepSeek, проверю оба.</span>}
          {guess.length === 0 && (
            <>
              <span>Провайдер не узнан, выберите:</span>
              <select className="sel" value={pick} aria-label="Провайдер" onChange={(e) => setPick(e.target.value)}>
                <option value="">—</option>
                {ai.providers.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
            </>
          )}
          <button className="btn primary" type="button" disabled={busy || !candidates.length} onClick={() => void save()}>
            {busy ? (
              <>
                <span className="spin" />
                Проверяю…
              </>
            ) : (
              "Проверить и сохранить"
            )}
          </button>
        </div>
      )}
    </div>
  );
}

export function AddCard({ p, secret, onSaved }: { ai?: AiState; p: AiProvider; secret?: SecretItem; onSaved: () => void }) {
  const [vals, setVals] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const say = useStore((s) => s.say);
  const fields = secret?.fields ?? [];
  const save = async () => {
    const filled = Object.fromEntries(
      Object.entries(vals)
        .filter(([, v]) => v.trim())
        .map(([k, v]) => [k, v.trim()]),
    );
    if (!Object.keys(filled).length) return say("Вставьте ключ.");
    setBusy(true);
    const ok = await actions.putSecret(p.id, filled);
    setBusy(false);
    if (ok) {
      setVals({});
      onSaved();
    }
  };
  return (
    <div className="kcard">
      <div className="kc-h">
        <b>{p.name}</b>
        <span className={`tag ${p.free_tier ? "hot" : "rst"}`}>{p.free_tier ? "бесплатно" : "платно"}</span>
      </div>
      <p className="kc-d">
        {p.note}{" "}
        {p.signup && (
          <a href={p.signup} target="_blank" rel="noopener noreferrer">
            Где взять ключ ↗
          </a>
        )}
      </p>
      {fields.map((f) => (
        <div className="kf" key={f.env}>
          <label htmlFor={`ai-${f.env}`}>
            <span>{f.label}</span>
            <span className="saved">{f.set ? (f.secret ? `сохранён ••••${f.last4 ?? ""}` : f.value) : f.optional ? "необязательно" : "не задан"}</span>
          </label>
          <input
            className="inp"
            id={`ai-${f.env}`}
            type={f.secret ? "password" : "text"}
            autoComplete="off"
            spellCheck={false}
            placeholder={f.secret ? "Вставьте ключ" : ""}
            value={vals[f.env] ?? ""}
            onChange={(e) => setVals({ ...vals, [f.env]: e.target.value })}
          />
        </div>
      ))}
      <div className="kc-a">
        {fields.length > 0 && (
          <button className="btn primary" type="button" disabled={busy} onClick={() => void save()}>
            {busy ? (
              <>
                <span className="spin" />
                Проверяю…
              </>
            ) : (
              "Проверить и сохранить"
            )}
          </button>
        )}
      </div>
      {secret?.message && <p className={`kc-msg ${secret.status === "ok" ? "ok" : secret.status === "bad" ? "bad" : ""}`}>{secret.message}</p>}
    </div>
  );
}
