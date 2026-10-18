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
  version?: string;
  isVersionAtLeast?(version: string): boolean;
  setHeaderColor?(color: string): void;
  setBackgroundColor?(color: string): void;
  setBottomBarColor?(color: string): void;
  disableVerticalSwipes?(): void;
  BackButton?: { show(): void; hide(): void; onClick(cb: () => void): void; offClick(cb: () => void): void };
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
  if (kind !== "miniapp") return;
  const t = tg();
  t?.ready();
  t?.expand();
  document.documentElement.classList.add("miniapp");
  // Шапка и низ Telegram в цвет приложения, чтобы окно выглядело одним целым.
  // Каждый вызов в try: старые версии Telegram не знают новых методов.
  const bg = getComputedStyle(document.documentElement).getPropertyValue("--bg").trim() || "#100F0D";
  const panel = getComputedStyle(document.documentElement).getPropertyValue("--panel").trim() || bg;
  for (const call of [() => t?.setHeaderColor?.(bg), () => t?.setBackgroundColor?.(bg), () => t?.setBottomBarColor?.(panel)]) {
    try {
      call();
    } catch {
      /* метод недоступен в этой версии */
    }
  }
  // Свайп вниз при прокрутке не должен закрывать приложение.
  try {
    if (t?.isVersionAtLeast?.("7.7")) t.disableVerticalSwipes?.();
  } catch {
    /* не критично */
  }
}

let backHandler: (() => void) | null = null;

/** Кнопка «Назад» самого Telegram: показывается, пока есть куда вернуться. */
export function setBack(handler: (() => void) | null): void {
  const b = kind === "miniapp" ? tg()?.BackButton : undefined;
  if (!b) return;
  try {
    if (backHandler) b.offClick(backHandler);
    backHandler = handler;
    if (handler) {
      b.onClick(handler);
      b.show();
    } else b.hide();
  } catch {
    /* старая версия Telegram без BackButton */
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

/** Токен запуска из ссылки лаунчера `#local=…`; другие фрагменты игнорируются. */
export function parseLocalHash(hash: string): string | null {
  const m = /^#local=([A-Za-z0-9_-]{32,})$/.exec(hash);
  return m ? m[1] : null;
}

/** Забирает токен из адреса и сразу стирает его из строки браузера и истории. */
export function takeLocalToken(): string | null {
  const token = parseLocalHash(window.location.hash);
  if (token) window.history.replaceState(null, "", window.location.pathname + window.location.search);
  return token;
}

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
