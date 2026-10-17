// Раздел «AI» в настройках: режим и порядок цепочки, модели, статусы, новые провайдеры.

import { useLayoutEffect, useRef, useState } from "react";

import { fmtN, plural } from "../lib/format";
import { reducedMotion } from "../lib/platform";
import type { AiMode, AiModel, AiProvider, AiState, SecretItem } from "../lib/types";
import { actions, useStore } from "../store";
import { Icon, Q, Seg, Toggle } from "../ui";

const EASE = "cubic-bezier(.2,.8,.2,1)";

const MODES: readonly (readonly [AiMode, string])[] = [
  ["free", "Бесплатные"],
  ["auto", "Смешанный"],
  ["paid", "Платные"],
];
const MODE_NOTE: Record<AiMode, string> = {
  free: "Только бесплатные модели. Бот никогда сам не начнёт тратить деньги: платные провайдеры не используются.",
  auto: "Сначала бесплатные, платные — только когда бесплатные кончились или на паузе.",
  paid: "Только платные провайдеры: бесплатные не используются.",
};

const STATUS: Record<string, [string, string]> = {
  ok: ["ok", "работает"],
  rate_limit: ["warn", "лимит"],
  no_balance: ["bad", "нет денег"],
  bad_key: ["bad", "ключ не принят"],
  bad_model: ["bad", "нет такой модели"],
  bad_format: ["warn", "ломает формат ответа"],
  bad_request: ["warn", "отклонил запрос"],
  error: ["warn", "нет связи"],
  unknown: ["", "ещё не отвечал"],
};

const hhmm = (epoch: number) =>
  new Date(epoch * 1000).toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" });

const modelLabel = (m: AiModel) =>
  `${m.id}${m.recommended ? " ★" : ""} · ${m.free ? "бесплатно" : "платно"}${m.strong ? "" : " · слабая"}`;

/** Модели, выбранные в приложении: PUT заменяет карту целиком. */
const appModels = (ai: AiState) =>
  Object.fromEntries(ai.providers.filter((p) => p.model_source === "app").map((p) => [p.id, p.model]));

export function AiPane() {
  const ai = useStore((s) => s.ai);
  const secrets = useStore((s) => s.secrets);
  const [needRestart, setNeedRestart] = useState(false);
  if (!ai) return <p className="empty">Загружаю список провайдеров…</p>;
  const inChain = ai.providers.filter((p) => p.usable);
  const rest = ai.providers.filter((p) => !p.usable);
  const secretOf = (id: string) => secrets?.find((s) => s.service === id);
  return (
    <div className="set-pane view enter">
      <ChainCard ai={ai} />
      <article className="card">
        <div className="card-h">
          <h2>В цепочке</h2>
          <Q tip="Бот спрашивает провайдеров сверху вниз. Ошибка или лимит — сразу следующий; после лимита провайдер отдыхает и потом возвращается." />
          <span className="pill sp">{ai.chain.length}</span>
        </div>
        {inChain.length ? <ChainList ai={ai} list={inChain} /> : <p className="empty">Пока ни одного провайдера. Добавьте бесплатный ниже — хватит одного ключа.</p>}
      </article>
      <AddProviders ai={ai} list={rest} secretOf={secretOf} onSaved={() => setNeedRestart(true)} />
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

// ── Режим и что сейчас отвечает ───────────────────────────────────────────────

function ChainCard({ ai }: { ai: AiState }) {
  const byId = Object.fromEntries(ai.providers.map((p) => [p.id, p]));
  const last = byId[ai.last];
  const pinned = ai.pinned !== "auto" ? byId[ai.pinned] : undefined;
  return (
    <article className="card">
      <div className="card-h">
        <h2>Цепочка AI</h2>
        {last && (
          <span className="pill ok sp">
            <span className="dot ok live" />
            отвечает {last.name}
          </span>
        )}
      </div>
      <Seg className="full" value={ai.mode} options={MODES} label="Режим цепочки" onChange={(mode) => void actions.putAiChain({ mode }, MODES.find(([m]) => m === mode)?.[1])} />
      <p className="note-soon" style={{ marginTop: 8 }}>
        {MODE_NOTE[ai.mode]}
      </p>
      <p className="ai-route">
        {ai.chain.length ? (
          ai.chain.map((c, i) => (
            <span key={c.provider}>
              {i > 0 && " → "}
              <b>{byId[c.provider]?.name ?? c.provider}</b>
            </span>
          ))
        ) : (
          <span className="down">Нет доступных провайдеров — бот работает на локальных стратегиях.</span>
        )}
      </p>
      {pinned && (
        <div className="row-tgl" style={{ marginTop: 8 }}>
          <span className="note-soon">
            Первым всегда идёт <b>{pinned.name}</b> — выбран в /provider.
          </span>
          <button className="btn" type="button" onClick={() => void actions.patchConfig({ AI_PROVIDER: "auto" })}>
            Открепить
          </button>
        </div>
      )}
    </article>
  );
}

// ── Подключённые провайдеры ───────────────────────────────────────────────────

/** Карточки плавно переезжают при перестановке (FLIP по offsetTop). */
function useListFlip(ref: React.RefObject<HTMLElement>): void {
  const last = useRef(new Map<string, number>());
  useLayoutEffect(() => {
    const host = ref.current;
    if (!host) return;
    const next = new Map<string, number>();
    host.querySelectorAll<HTMLElement>("[data-pid]").forEach((el) => {
      const k = el.dataset.pid!;
      next.set(k, el.offsetTop);
      const prev = last.current.get(k);
      if (prev !== undefined && prev !== el.offsetTop && !reducedMotion())
        el.animate([{ transform: `translateY(${prev - el.offsetTop}px)` }, { transform: "none" }], { duration: 380, easing: EASE });
    });
    last.current = next;
  });
}

function ChainList({ ai, list }: { ai: AiState; list: AiProvider[] }) {
  const host = useRef<HTMLOListElement>(null);
  useListFlip(host);
  const position = new Map(ai.chain.map((c, i) => [c.provider, i + 1]));
  const move = (p: AiProvider, dir: -1 | 1) => {
    const j = list.indexOf(p) + dir;
    if (j < 0 || j >= list.length) return;
    const ids = ai.providers.map((x) => x.id);
    const a = ids.indexOf(p.id);
    const b = ids.indexOf(list[j].id);
    [ids[a], ids[b]] = [ids[b], ids[a]];
    void actions.putAiChain({ order: ids });
  };
  return (
    <ol className="aichain" ref={host}>
      {list.map((p, i) => (
        <ProviderRow key={p.id} ai={ai} p={p} n={position.get(p.id)} first={i === 0} lastRow={i === list.length - 1} onMove={(d) => move(p, d)} />
      ))}
    </ol>
  );
}

function why(ai: AiState, p: AiProvider): string {
  if (p.id === ai.pinned) return "";
  if (p.disabled) return "выключен вами";
  if (ai.mode === "free" && !p.free) return "платная модель — не используется в режиме «Бесплатные»";
  if (ai.mode === "paid" && p.free) return "бесплатная модель — не используется в режиме «Платные»";
  return "";
}

function ProviderRow(props: { ai: AiState; p: AiProvider; n?: number; first: boolean; lastRow: boolean; onMove: (dir: -1 | 1) => void }) {
  const { ai, p } = props;
  const [cls, text] = STATUS[p.status] ?? ["warn", p.status];
  const paused = p.cooldown_until && p.cooldown_until * 1000 > Date.now();
  const skip = why(ai, p);
  const disabledIds = ai.providers.filter((x) => x.disabled).map((x) => x.id);
  const keyless = ai.providers.filter((x) => x.keyless_on).map((x) => x.id);
  return (
    <li className={`aip${props.n ? "" : " off"}`} data-pid={p.id}>
      <div className="aip-h">
        <span className="n">{props.n ?? "—"}</span>
        <b>{p.name}</b>
        <span className={`tag ${p.free ? "hot" : "rst"}`}>{p.free ? "бесплатно" : "платно"}</span>
        {!p.strong && <span className="tag rst" title="Мелкая модель чаще ломает формат ответа и хуже решает">слабая</span>}
        {p.id === ai.pinned && <span className="tag hot">первый</span>}
        <span className="sp">
          <span className={`pill ${cls}`}>
            <span className={`dot ${cls}`} />
            {paused ? `пауза до ${hhmm(p.cooldown_until!)}` : text}
          </span>
          <span className="mv">
            <button className="btn" type="button" aria-label={`Выше: ${p.name}`} disabled={props.first} onClick={() => props.onMove(-1)}>
              ↑
            </button>
            <button className="btn" type="button" aria-label={`Ниже: ${p.name}`} disabled={props.lastRow} onClick={() => props.onMove(1)}>
              ↓
            </button>
          </span>
          <Toggle
            on={!p.disabled}
            label={`${p.name} в цепочке`}
            onChange={(on) => void actions.putAiChain({ disabled: on ? disabledIds.filter((id) => id !== p.id) : [...disabledIds, p.id] })}
          />
        </span>
      </div>
      <ModelPicker ai={ai} p={p} />
      <small>
        {skip && <span className="down">{skip}. </span>}
        {p.model_source === "env" && "Модель из .env — выберите здесь, чтобы заменить. "}
        {!p.verified && "ID моделей не сверены — нажмите «Загрузить список». "}
        сегодня {p.calls_today} {plural(p.calls_today, ["запрос", "запроса", "запросов"])}
        {p.tokens_today > 0 && ` · ${fmtN(p.tokens_today, 0)} токенов`}
      </small>
      <div className="btn-row">
        {p.id !== ai.pinned && (
          <button className="btn" type="button" onClick={() => void actions.patchConfig({ AI_PROVIDER: p.id })}>
            Сделать первым
          </button>
        )}
        {p.keyless_on && (
          <button className="btn" type="button" onClick={() => void actions.putAiChain({ keyless: keyless.filter((id) => id !== p.id) }, `${p.name} отключён`)}>
            Отключить (без ключа)
          </button>
        )}
      </div>
    </li>
  );
}

function ModelPicker({ ai, p }: { ai: AiState; p: AiProvider }) {
  const [live, setLive] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const known = new Set(p.models.map((m) => m.id));
  const extra = [...new Set([...(known.has(p.model) ? [] : [p.model]), ...live.filter((id) => !known.has(id))])];
  const pick = (model: string) => void actions.putAiChain({ models: { ...appModels(ai), [p.id]: model } }, `${p.name}: ${model}`);
  return (
    <div className="aip-row">
      <select className="sel" value={p.model} aria-label={`Модель ${p.name}`} onChange={(e) => pick(e.target.value)}>
        <optgroup label="Каталог">
          {p.models.map((m) => (
            <option key={m.id} value={m.id}>
              {modelLabel(m)}
            </option>
          ))}
        </optgroup>
        {extra.length > 0 && (
          <optgroup label={live.length ? `У провайдера: ${live.length}` : "Своя"}>
            {extra.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </optgroup>
        )}
      </select>
      <button
        className="btn"
        type="button"
        disabled={busy}
        title="Спросить у провайдера актуальный список моделей (лимит не тратит)"
        onClick={async () => {
          setBusy(true);
          setLive(await actions.aiModels(p.id));
          setBusy(false);
        }}
      >
        {busy ? <span className="spin" /> : "Загрузить список"}
      </button>
    </div>
  );
}

// ── Новые провайдеры ──────────────────────────────────────────────────────────

function AddProviders(props: { ai: AiState; list: AiProvider[]; secretOf: (id: string) => SecretItem | undefined; onSaved: () => void }) {
  const free = props.list.filter((p) => p.free_tier);
  const paid = props.list.filter((p) => !p.free_tier);
  if (!props.list.length) return null;
  return (
    <article className="card">
      <div className="card-h">
        <h2>Добавить провайдера</h2>
        <Q tip="Чем больше бесплатных провайдеров с ключами, тем реже бот остаётся без AI: у каждого свой дневной лимит." />
      </div>
      {[
        ["Бесплатные", free],
        ["Платные", paid],
      ].map(([title, list]) =>
        (list as AiProvider[]).length ? (
          <section className="kgroup" key={title as string}>
            <h3>{title as string}</h3>
            <div className="addp">
              {(list as AiProvider[]).map((p) => (
                <AddCard key={p.id} ai={props.ai} p={p} secret={props.secretOf(p.id)} onSaved={props.onSaved} />
              ))}
            </div>
          </section>
        ) : null,
      )}
    </article>
  );
}

function AddCard({ ai, p, secret, onSaved }: { ai: AiState; p: AiProvider; secret?: SecretItem; onSaved: () => void }) {
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
  const keyless = ai.providers.filter((x) => x.keyless_on).map((x) => x.id);
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
        {p.key_optional && (
          <button className="btn" type="button" onClick={() => void actions.putAiChain({ keyless: [...keyless, p.id] }, `${p.name} включён`)}>
            {Icon.check}
            Включить без ключа
          </button>
        )}
      </div>
      {secret?.message && <p className={`kc-msg ${secret.status === "ok" ? "ok" : secret.status === "bad" ? "bad" : ""}`}>{secret.message}</p>}
    </div>
  );
}
