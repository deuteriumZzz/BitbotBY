import * as platform from "./platform";
import type { BotEvent, CommandResult } from "./types";

export const isMock =
  new URLSearchParams(location.search).has("mock") || import.meta.env.VITE_MOCK === "1";

// Мок грузится только в демо-режиме — в обычную сборку фикстуры не попадают.
const loadMock = () => import("../mock/mock").then((m) => m.mock);

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public fields?: Record<string, string>,
  ) {
    super(message);
  }
}

interface ErrorBody {
  error?: { code?: string; message?: string; fields?: Record<string, string> };
}

export async function api<T>(method: string, path: string, body?: unknown): Promise<T> {
  if (isMock) return (await loadMock()).request(method, path, body) as Promise<T>;
  const token = platform.getToken();
  let res: Response;
  try {
    res = await fetch(platform.apiBase() + path, {
      method,
      headers: {
        "Content-Type": "application/json",
        "X-Client": platform.kind,
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "network", "Нет связи с сервером бота.");
  }
  const data: unknown = await res.json().catch(() => ({}));
  if (!res.ok && res.status !== 202) {
    const err = (data as ErrorBody).error;
    throw new ApiError(
      res.status,
      err?.code ?? `http_${res.status}`,
      err?.message ?? "Сервер не смог выполнить запрос.",
      err?.fields,
    );
  }
  return data as T;
}

const newId = (): string =>
  globalThis.crypto?.randomUUID?.() ??
  `r${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;

interface CommandResponse {
  status: string;
  result?: CommandResult;
}

/**
 * Команда боту через очередь: повтор с тем же request_id не исполнится дважды.
 * Ждём результат до ~12 с; бот исполняет команды в своём цикле.
 */
export async function command(
  method: "POST" | "PATCH",
  path: string,
  body: Record<string, unknown> = {},
): Promise<CommandResult> {
  const requestId = newId();
  let resp = await api<CommandResponse>(method, path, { ...body, request_id: requestId });
  for (let i = 0; resp.status !== "done" && i < 24; i++) {
    await new Promise((r) => setTimeout(r, 500));
    resp = await api<CommandResponse>("GET", `/api/commands/${requestId}`);
  }
  if (resp.status !== "done" || !resp.result) {
    throw new ApiError(504, "timeout", "Бот не ответил вовремя. Проверьте, что он запущен.");
  }
  if (!resp.result.ok) {
    throw new ApiError(
      422,
      resp.result.error?.code ?? "failed",
      resp.result.error?.message ?? "Команда не выполнилась.",
    );
  }
  return resp.result;
}

/** Поток событий бота с авто-переподключением. Возвращает функцию отписки. */
export function subscribe(
  onEvent: (e: BotEvent) => void,
  onStatus: (up: boolean) => void,
): () => void {
  if (isMock) {
    let off = (): void => undefined;
    let gone = false;
    void loadMock().then((m) => {
      if (gone) return;
      onStatus(true);
      off = m.subscribe(onEvent);
    });
    return () => {
      gone = true;
      off();
    };
  }
  let ws: WebSocket | null = null;
  let closed = false;
  let retry = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;

  const url = (): string => `${(platform.apiBase() || location.origin).replace(/^http/, "ws")}/ws`;

  const open = (): void => {
    ws = new WebSocket(url());
    ws.onopen = () => {
      retry = 0;
      onStatus(true);
      const token = platform.getToken();
      if (token) ws?.send(JSON.stringify({ type: "auth", token }));
    };
    ws.onmessage = (m) => {
      try {
        onEvent(JSON.parse(String(m.data)) as BotEvent);
      } catch {
        /* битое сообщение — пропускаем */
      }
    };
    ws.onclose = () => {
      onStatus(false);
      if (!closed) timer = setTimeout(open, Math.min(30_000, 1000 * 2 ** retry++));
    };
  };
  open();
  return () => {
    closed = true;
    clearTimeout(timer);
    ws?.close();
  };
}
