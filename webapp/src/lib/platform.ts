// Единственное место, которое знает, где запущено приложение:
// Telegram MiniApp, десктоп (Tauri) или обычный браузер.

interface TelegramWebApp {
  initData: string;
  ready(): void;
  expand(): void;
  HapticFeedback?: {
    impactOccurred(style: "light" | "medium" | "heavy"): void;
    notificationOccurred(type: "success" | "warning" | "error"): void;
  };
  openLink?(url: string): void;
}

declare global {
  interface Window {
    Telegram?: { WebApp?: TelegramWebApp };
    __TAURI__?: unknown;
  }
}

export type PlatformKind = "miniapp" | "desktop" | "web";

const tg = (): TelegramWebApp | undefined => window.Telegram?.WebApp;

export const kind: PlatformKind = tg()?.initData
  ? "miniapp"
  : window.__TAURI__
    ? "desktop"
    : "web";

export function init(): void {
  if (kind === "miniapp") {
    tg()?.ready();
    tg()?.expand();
  }
}

export const initData = (): string => tg()?.initData ?? "";

export function haptic(type: "tap" | "success" | "warning" | "error"): void {
  const h = tg()?.HapticFeedback;
  if (!h) return;
  if (type === "tap") h.impactOccurred("light");
  else h.notificationOccurred(type);
}

export function openLink(url: string): void {
  const t = tg();
  if (t?.openLink) t.openLink(url);
  else window.open(url, "_blank", "noopener,noreferrer");
}

/** Адрес сервера бота. В браузере и MiniApp — тот же origin; в Tauri — из настроек. */
export function apiBase(): string {
  if (kind !== "desktop") return "";
  try {
    return localStorage.getItem("bitbot.server") ?? "";
  } catch {
    return "";
  }
}

export function setApiBase(url: string): void {
  try {
    localStorage.setItem("bitbot.server", url.replace(/\/+$/, ""));
  } catch {
    /* хранилище недоступно — адрес придётся ввести заново */
  }
}

// Токен сессии. В MiniApp — только в памяти: initData даёт новый при каждом
// запуске. На десктопе и в браузере — в sessionStorage, чтобы не входить заново
// после перезагрузки. Системная связка ключей для Tauri — следующим шагом.
let memoryToken: string | null = null;

export function getToken(): string | null {
  if (memoryToken || kind === "miniapp") return memoryToken;
  try {
    return sessionStorage.getItem("bitbot.token");
  } catch {
    return null;
  }
}

export function setToken(token: string | null): void {
  memoryToken = token;
  if (kind === "miniapp") return;
  try {
    if (token) sessionStorage.setItem("bitbot.token", token);
    else sessionStorage.removeItem("bitbot.token");
  } catch {
    /* без хранилища токен живёт до перезагрузки */
  }
}

export const reducedMotion = (): boolean =>
  window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
