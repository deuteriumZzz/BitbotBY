# BitbotBY Desktop

A Tauri 2 shell around the React frontend in `..`. It has no Rust logic,
plugins or IPC permissions of its own. The webview talks straight to the bot
server whose address you enter on the login screen, and you sign in with the
6-digit code that the bot sends to Telegram.

Requirements: Rust (cargo), Node/npm, and on macOS the Xcode Command Line Tools.

```bash
cd desktop && npm install
```

## Development

```bash
cd desktop && npm run dev
```

This starts `npm --prefix .. run dev` (Vite on :5173) and opens a window
at `http://localhost:5173/app/`. On the login screen, enter
`http://localhost:5173` as the server. Vite then proxies `/api` and `/ws` to the
bot on :8080, so the requests stay on one origin and CORS never comes up.

## Build

```bash
cd desktop && npm run build
```

Debug build, faster, with devtools enabled:

```bash
cd desktop && npx tauri build --debug
```

Build through the Tauri CLI, not with a bare `cargo build`. Only the CLI turns
on the `custom-protocol` feature that embeds `frontend/dist`. Without it the
binary expects the dev server at `devUrl`.

Output: `src-tauri/target/{release,debug}/bundle/macos/BitbotBY.app`. Only the
`.app` bundle is built (`bundle.targets`). Pass `--bundles dmg` for a disk image.

The build runs `npm --prefix .. run build` with `TAURI_ENV_PLATFORM` set.
Under that flag `frontend/vite.config.ts` uses `base: "./"`, because Tauri serves
the files from the root and not from `/app/`. Relative paths also work when
FastAPI serves the same `frontend/dist` at `/app/`.

## Server access

- CSP (`tauri.conf.json`): `connect-src` allows `http:`, `https:`, `ws:` and
  `wss:` to any host. Scripts load only from the bundle. Styles come from the
  bundle plus Google Fonts.
- `telegram-web-app.js` is deliberately blocked by the CSP. The desktop app does
  not need it, and a third-party script should not see the session token.
- **CORS:** the packaged app's origin is `tauri://localhost` (macOS/Linux) or
  `http://tauri.localhost` (Windows). The bot server has to allow these origins
  for `/api` (the `Authorization` header triggers a preflight). WebSocket is not
  subject to CORS.

## Not done yet

The session token still lives in `sessionStorage` (`frontend/src/lib/platform.ts`).
To move it to the OS keychain, add a Rust command built on the `keyring` crate
and make `getToken`/`setToken` asynchronous through
`window.__TAURI__.core.invoke`, which also changes the code that calls them.
