import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Приложение раздаётся FastAPI-дашбордом на /app; в разработке — прокси к нему на :8080.
// Сборку для Tauri (desktop/) запускает его CLI с TAURI_ENV_PLATFORM: там файлы лежат
// в корне, поэтому пути относительные — они же работают и под /app/ на сервере.
export default defineConfig(({ command }) => ({
  base: command === "build" && process.env.TAURI_ENV_PLATFORM ? "./" : "/app/",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8080",
      "/ws": { target: "ws://localhost:8080", ws: true },
    },
  },
  test: { environment: "jsdom" },
}));
