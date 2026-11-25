// Раздел «AI»: ключи, кто отвечает и какой моделью, честность прогнозов.
// Порядок опроса выбирает бот: сначала бесплатные, потом платные (bitbot.config.ai_providers.chain).

import { useEffect, useState } from "react";

import { api } from "../lib/api";
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

/** Модели, выбранные в приложении: PUT заменяет карту целиком. */
const appModels = (ai: AiState) =>
  Object.fromEntries(ai.providers.filter((p) => p.model_source === "app").map((p) => [p.id, p.model]));

export function AiPane() {
  const ai = useStore((s) => s.ai);
  const secrets = useStore((s) => s.secrets);
  const [needRestart, setNeedRestart] = useState(false);
  const local = useStore((s) => s.bot?.mode === "local");
  if (!ai) return <Skel rows={4} label="Загружаю список провайдеров" />;
  const secretOf = (id: string) => secrets?.find((s) => s.service === id);
  return (
    <div className="set-pane view enter">
      {local && (
        <div className="honest">
          {Icon.warn}
          <div>
            <b>Сейчас AI не используется.</b> Включён режим «Локальные стратегии (без AI)» — провайдеры ниже подключены, но бот их не
            спрашивает.{" "}
            <button className="btn" type="button" onClick={() => void actions.patchConfig({ MODE: "ai" })}>
              Вернуть AI
            </button>
          </div>
        </div>
      )}
      <ProvidersCard ai={ai} secretOf={secretOf} onSaved={() => setNeedRestart(true)} />
      <HonestyCard ai={ai} />
      <QuickAdd ai={ai} secretOf={secretOf} onSaved={() => setNeedRestart(true)} />
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

/** Кто отвечает: по очереди, сначала бесплатные. У каждого — статус, модель (выбор в «Профи») и расход за сегодня. */
function ProvidersCard(props: { ai: AiState; secretOf: (id: string) => SecretItem | undefined; onSaved: () => void }) {
  const { ai } = props;
  const [adding, setAdding] = useState("");
  const pro = useStore((s) => s.ui === "pro");
  const order = ai.chain.map((c) => c.provider);
  const withKey = order.map((id) => ai.providers.find((p) => p.id === id)!).filter(Boolean);
  const without = ai.providers.filter((p) => !p.usable);
  return (
    <article className="card">
      <div className="card-h">
        <h2>Кто отвечает</h2>
        <Q tip="Бот спрашивает по очереди: сначала бесплатных, потом платных — платного только если бесплатные не ответили (лимит, нет связи). Не ответил никто — решают стратегии. Порядок выбирает бот." />
      </div>
      {!withKey.length && <p className="note-soon">Пока ни одного ключа AI — решают стратегии. Добавьте бесплатный Groq ниже или вставьте любой ключ.</p>}
      <ul className="ai-list">
        {withKey.map((p, i) => {
          const [cls, text] = STATUS[p.status] ?? ["warn", p.status];
          const paused = p.cooldown_until && p.cooldown_until * 1000 > Date.now();
          return (
            <li key={p.id}>
              <Avatar id={p.id} name={p.name} />
              <div className="ai-main">
                <b>{p.name}</b> <small className="muted">{i === 0 ? "основной" : "запасной"}{p.free ? " · бесплатно" : " · платно"}</small>
                <div className="ai-model">
                  {pro ? (
                    <select
                      className="sel"
                      value={p.model}
                      aria-label={`Модель ${p.name}`}
                      onChange={(e) => void actions.putAiChain({ models: { ...appModels(ai), [p.id]: e.target.value } }, `${p.name}: ${e.target.value}`)}
                    >
                      {[...new Set([p.model, ...p.models.map((m) => m.id)])].map((id) => (
                        <option key={id} value={id}>
                          {id}
                          {p.models.find((m) => m.id === id)?.recommended ? " (рекомендуется)" : ""}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <span>модель {p.model}</span>
                  )}
                </div>
                <small className="muted">
                  сегодня {p.calls_today} {plural(p.calls_today, ["запрос", "запроса", "запросов"])}
                  {p.tokens_today > 0 && ` · ${fmtN(p.tokens_today, 0)} токенов`}
                </small>
              </div>
              <span className={`pill sp ${paused ? "warn" : cls}`}>{paused ? `пауза до ${hhmm(p.cooldown_until!)}` : text}</span>
            </li>
          );
        })}
        {without.map((p) => (
          <li key={p.id} className="ai-off">
            <Avatar id={p.id} name={p.name} />
            <div className="ai-main">
              <b>{p.name}</b> <small className="muted">{p.free_tier ? "бесплатно" : "платно"} · нет ключа</small>
              {adding === p.id && <AddCard p={p} secret={props.secretOf(p.id)} onSaved={() => (setAdding(""), props.onSaved())} />}
            </div>
            {adding !== p.id && (
              <button className="btn" type="button" onClick={() => setAdding(p.id)}>
                Добавить ключ
              </button>
            )}
          </li>
        ))}
      </ul>
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

function HonestyCard({ ai }: { ai: AiState }) {
  const s = useScoreboard();
  const rows = s ? Object.entries(s.providers) : [];
  const name = (id: string) => ai.providers.find((p) => p.id === id)?.name ?? id;
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


// ── Быстрое добавление: вставили ключ — провайдер определился сам ─────────────

function QuickAdd(props: { ai: AiState; secretOf: (id: string) => SecretItem | undefined; onSaved: () => void }) {
  const { ai } = props;
  const [val, setVal] = useState("");
  const [pick, setPick] = useState("");
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);
  const [over, setOver] = useState(false);
  const guess = detectProvider(val);
  const chosen = guess.length === 1 ? guess[0] : pick;
  const provider = ai.providers.find((p) => p.id === chosen);
  const field = props.secretOf(chosen)?.fields.find((f) => f.secret) ?? props.secretOf(chosen)?.fields[0];
  const take = async (e: React.DragEvent) => {
    e.preventDefault();
    setOver(false);
    const file = e.dataTransfer.files[0];
    const text = file ? await file.text() : e.dataTransfer.getData("text");
    if (text.trim()) setVal(text.trim().split(/\s+/)[0]);
  };
  const save = async () => {
    if (!provider || !field) return;
    setBusy(true);
    const ok = await actions.putSecret(provider.id, { [field.env]: val.trim() });
    setBusy(false);
    if (ok) {
      setVal("");
      setPick("");
      setDone(true);
      setTimeout(() => setDone(false), 1800);
      props.onSaved();
    }
  };
  return (
    <article className={`card quick${over ? " over" : ""}${done ? " done" : ""}`} onDragOver={(e) => (e.preventDefault(), setOver(true))} onDragLeave={() => setOver(false)} onDrop={(e) => void take(e)}>
      <div className="card-h">
        <h2>Вставьте ключ</h2>
        <Q tip="Определю провайдера по началу ключа (например, gsk_ — Groq). Ключ уходит на сервер, проверяется запросом к сервису и хранится зашифрованным. Можно перетащить сюда текст или файл с ключом." />
      </div>
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
          placeholder="Вставьте или перетащите ключ"
          aria-label="Ключ нового провайдера"
          value={val}
          onChange={(e) => setVal(e.target.value)}
        />
      </div>
      {val.trim().length >= 8 && (
        <div className="quick-guess" aria-live="polite">
          {guess.length === 1 && provider && (
            <span>
              Похоже на <b>{provider.name}</b>
              {provider.key_set && " — ключ уже есть, он будет заменён"}.
            </span>
          )}
          {guess.length > 1 && (
            <>
              <span>Начало подходит нескольким — выберите:</span>
              {guess.map((id) => (
                <button key={id} type="button" className="chip-btn" aria-pressed={pick === id} onClick={() => setPick(id)}>
                  {ai.providers.find((p) => p.id === id)?.name ?? id}
                </button>
              ))}
            </>
          )}
          {guess.length === 0 && (
            <>
              <span>Формат не узнал — выберите провайдера:</span>
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
          <button className="btn primary" type="button" disabled={busy || !provider || !field} onClick={() => void save()}>
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
    </article>
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
