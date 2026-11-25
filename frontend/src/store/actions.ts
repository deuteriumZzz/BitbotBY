// Действия приложения: команды боту, правки настроек с отменой — вызываются из компонентов.

import { ApiError, api, command } from "../lib/api";
import { base, plural } from "../lib/format";
import type {
  AiChainChanges,
  AiState,
  Alert,
  JobId,
  Position,
  Symbols,
  TgStatus,
  TgStep,
} from "../lib/types";
import { useStore } from "./index";
import { EP } from "../api/endpoints";

// ── Действия (вызываются из компонентов) ──────────────────────────────────────

export const st = () => useStore.getState();

export interface PatchResult {
  applied_hot: string[];
  pending_restart: string[];
}

export const actions = {
  pause: () => st().act(() => command("POST", EP.botPause), "Бот на паузе"),
  resume: () => st().act(() => command("POST", EP.botResume), "Бот снова работает"),
  panic: () =>
    st().act(async () => {
      const r = await command("POST", EP.botPanic);
      const c = Number(r.closed ?? 0);
      st().say(`Закрыто ${c} ${plural(c, ["позиция", "позиции", "позиций"])}. Бот на паузе.`);
    }),
  restart: () => st().act(() => command("POST", EP.botRestart), "Бот перезапускается…"),
  runJob: (id: JobId, profile?: string) =>
    st().act(async () => {
      await command("POST", EP.job(id), profile ? { profile } : undefined);
      useStore.setState((s) => ({ jobs: { ...s.jobs, [id]: { running: true, info: { pct: 0 } } } }));
    }),
  resetConfig: () => st().act(() => api("POST", EP.configReset), "Настройки сброшены к значениям из .env"),
  rollbackModel: () => st().act(() => command("POST", EP.modelRollback), "Модель SAC откатана — бот подхватит её в следующем цикле"),
  closePosition: (p: Position, fraction = 1) =>
    st().act(
      () => command("POST", EP.positionClose(p.trade_id), { fraction }),
      fraction === 1 ? `${base(p.symbol)} закрыт по рынку` : `Закрыто ${Math.round(fraction * 100)}% ${base(p.symbol)}`,
    ),
  updatePosition: (p: Position, changes: { sl?: number; tp?: number; trailing?: boolean }) =>
    st().act(() => command("PATCH", EP.position(p.trade_id), changes), "Сохранено"),
  answerSignal: (id: string, confirm: boolean) =>
    st().act(
      () => command("POST", EP.signal(id, confirm)),
      confirm ? "Сделка подтверждена" : "Сделка пропущена",
    ),
  accountMode: (paper: boolean, pin = "") =>
    st().act(() => command("POST", EP.accountMode, { paper, pin }), paper ? "Вернулись в Paper" : "Включён Live"),
  async patchConfig(changes: Record<string, unknown>, opts?: { undo?: boolean }): Promise<PatchResult | null> {
    // Прежние значения нужны для «Отменить»: возвращаем их тем же запросом.
    const before = Object.fromEntries(
      Object.keys(changes)
        .map((k) => [k, st().config[k]?.value] as const)
        .filter(([, v]) => v !== undefined),
    );
    let result: PatchResult | null = null;
    const ok = await st().act(async () => {
      result = await api<PatchResult>("PATCH", EP.config, changes);
    });
    if (!ok || !result) return null;
    const r: PatchResult = result;
    const text = r.pending_restart.length
      ? `Сразу: ${r.applied_hot.length}. После перезапуска: ${r.pending_restart.length}.`
      : "Применено — бот уже работает с новыми настройками";
    if (opts?.undo !== false && Object.keys(before).length) st().sayUndo(text, () => void actions.patchConfig(before, { undo: false }));
    else st().say(text);
    await st().loadConfig().catch(() => undefined);
    return r;
  },
  setProfile: (name: string) =>
    st().act(() => api("POST", EP.profile, { name }).then(() => st().loadState()), "Профиль рынка переключён"),
  setStrategy: (name: string, enabled: boolean) =>
    st().act(() => api("PATCH", EP.strategy(name), { enabled }).then(() => st().loadStrategies())),
  resetStrategies: () =>
    st().act(() => api("POST", EP.strategiesReset).then(() => st().loadStrategies()), "Основные стратегии включены, экспериментальные выключены"),
  putSymbols: (symbols: Symbols) =>
    st().act(() => api<Symbols>("PUT", EP.symbols, symbols).then((v) => useStore.setState({ symbols: v }))),
  patchAlert: (kind: string, changes: Partial<Alert>) =>
    st().act(() => api("PATCH", EP.alert(kind), changes).then(() => st().loadAlerts())),
  putSecret: (service: string, fields: Record<string, string>, model?: string) =>
    st().act(
      () => api("PUT", EP.secret(service), { fields, ...(model ? { model } : {}) }).then(() => st().loadSecrets()),
      "Ключ сохранён. Заработает после перезапуска бота.",
    ),
  addXAccount: (body: { login: string; password: string; email: string; email_password: string }) =>
    st().act(() => api("POST", EP.twitterAccounts, body).then(() => st().loadSecrets()), "Аккаунт добавлен. Заработает после перезапуска бота."),
  removeXAccount: (login: string) =>
    st().act(() => api("DELETE", EP.twitterAccount(login)).then(() => st().loadSecrets()), "Аккаунт удалён."),
  putAiChain: (changes: AiChainChanges, ok?: string) =>
    st().act(() => api<AiState>("PUT", EP.aiChain, changes).then((ai) => useStore.setState({ ai })), ok),
  /** Живой список моделей провайдера по сохранённому ключу. */
  aiModels: async (provider: string): Promise<string[]> => {
    try {
      return (await api<{ models: string[] }>("GET", EP.aiModels(provider))).models;
    } catch (e) {
      st().say(e instanceof ApiError ? e.message : "Не удалось загрузить список моделей.");
      return [];
    }
  },
  /** Проверка связи с провайдером: запрос списка моделей по сохранённому ключу (лимит не тратит). */
  aiProbe: async (provider: string): Promise<{ ok: boolean; ms: number; message: string }> => {
    const t0 = performance.now();
    try {
      await api("GET", EP.aiModels(provider));
      return { ok: true, ms: Math.round(performance.now() - t0), message: "" };
    } catch (e) {
      return { ok: false, ms: Math.round(performance.now() - t0), message: e instanceof ApiError ? e.message : "нет связи с сервером" };
    }
  },
  /** Найти свой chat id: кто писал боту. Токен берётся из поля или из сохранённого. */
  findTelegramChat: async (token: string, code: string): Promise<{ id: string; name: string }[] | null> => {
    try {
      return (await api<{ chats: { id: string; name: string }[] }>("POST", EP.telegramChat, { code, ...(token ? { token } : {}) })).chats;
    } catch (e) {
      st().say(e instanceof ApiError ? e.message : "Не удалось связаться с Telegram.");
      return null;
    }
  },
  /** Вход в Telegram для чтения каналов: «Войти» → код → облачный пароль, если он включён. */
  tgLogin: async (path: "login" | "code" | "password", body?: Record<string, string>): Promise<TgStep | null> => {
    try {
      const r = await api<TgStep>("POST", EP.telethon(path), body);
      if (r.step === "done") {
        await st().loadSecrets();
        st().say("Вход в Telegram выполнен — бот подключится к каналам в течение минуты.");
      }
      return r;
    } catch (e) {
      st().say(e instanceof ApiError ? e.message : "Не удалось связаться с Telegram.");
      return null;
    }
  },
  tgStatus: (): Promise<TgStatus | null> => api<TgStatus>("GET", EP.telethonStatus).catch(() => null),
  deleteSecret: (service: string) =>
    st().act(() => api("DELETE", EP.secret(service)).then(() => st().loadSecrets()), "Ключ удалён"),
};
