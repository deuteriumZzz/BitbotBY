import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Приложение раздаётся FastAPI-дашбордом на /app; в разработке — прокси к нему на :8080.
export default defineConfig({
  base: "/app/",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8080",
      "/ws": { target: "ws://localhost:8080", ws: true },
    },
  },
  test: { environment: "jsdom" },
});
